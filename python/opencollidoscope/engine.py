# Copyright (C) 2016 Queen Mary University of London
# Author: Fiore Martin
# Python port for current CPython / Raspberry Pi OS.
#
# This file is part of Collidoscope.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import NUM_WAVES, Config
from .dsp import MIN_GRAIN_DURATION, BiquadLowpass, PGranular, midi_note_ratio, normalize_recording, wave_inputs

NO_MIDI_NOTE = -50
LOOP_ID = -1
RAMP_TIME = 0.02


@dataclass
class RecordChunk:
    index: int
    bottom: float
    top: float


@dataclass
class CursorEvent:
    cmd: str  # "update" | "end"
    synth_id: int


@dataclass
class WaveState:
    selection_start_chunk: int = 0
    selection_size_chunks: int = 8
    grain_coeff: float = 1.0
    filter_midi: float = 127.0
    looping: bool = False
    chunks: list[tuple[float, float]] = field(default_factory=list)
    chunk_animate: list[float] = field(default_factory=list)
    cursors: dict[int, int] = field(default_factory=dict)
    cursor_started: dict[int, float] = field(default_factory=dict)
    oscilloscope: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    selection_alpha: float = 1.0
    applied_record_gain: float = 1.0


class WaveEngine:
    def __init__(self, config: Config, rng: random.Random) -> None:
        self.config = config
        n = config.buffer_frames()
        self.buffer = np.zeros(n, dtype=np.float32)
        self._rng = rng
        max_offset = max(1, config.sample_rate // 100)

        def rand_offset() -> int:
            return self._rng.randint(0, max_offset - 1)

        def on_trigger(kind: str, grain_id: int) -> None:
            cmd = "update" if kind == "t" else "end"
            self.cursor_events.append(CursorEvent(cmd, grain_id))

        self.loop = PGranular(self.buffer, config.sample_rate, rand_offset, on_trigger, LOOP_ID)
        self.voices = [
            PGranular(self.buffer, config.sample_rate, rand_offset, on_trigger, i)
            for i in range(config.max_keyboard_voices)
        ]
        self.midi_notes = [NO_MIDI_NOTE] * config.max_keyboard_voices
        self.filter = BiquadLowpass(config.sample_rate, config.max_filter_cutoff_hz)
        self.cursor_events: deque[CursorEvent] = deque(maxlen=config.cursor_trigger_buf_size)
        self.record_chunks: deque[RecordChunk] = deque(maxlen=config.num_chunks + 4)
        self.recording = False
        self.write_pos = 0
        self.chunk_index = 0
        self.chunk_counter = 0
        self.chunk_min = 1.0
        self.chunk_max = -1.0
        self.samples_per_chunk = config.samples_per_chunk()
        self.env_ramp = 0.0
        self.env_ramp_len = int(RAMP_TIME * config.sample_rate)
        self.env_ramp_rate = 1.0 / self.env_ramp_len if self.env_ramp_len else 0.0
        self.env_decay_start = n - self.env_ramp_len
        self.monitor = np.zeros(config.block_size, dtype=np.float32)
        self.record_peak = 0.0
        self.record_gain = 1.0
        self.temp = np.zeros(config.block_size, dtype=np.float32)
        self.mix = np.zeros(config.block_size, dtype=np.float32)
        start = 0
        size = 8 * self.samples_per_chunk
        self.loop.set_selection_start(start)
        self.loop.set_selection_size(size)
        for voice in self.voices:
            voice.set_selection_start(start)
            voice.set_selection_size(size)

    def start_record(self) -> None:
        self.recording = True
        self.write_pos = 0
        self.chunk_index = 0
        self.chunk_counter = 0
        self.chunk_min = 1.0
        self.chunk_max = -1.0
        self.env_ramp = 0.0
        self.record_peak = 0.0
        self.record_gain = 1.0
        self.record_chunks.append(RecordChunk(-1, 0.0, 0.0))  # WAVE_START sentinel

    def set_selection_start_samples(self, start: int) -> None:
        start = int(np.clip(start, 0, max(0, self.buffer.shape[0] - 1)))
        self.loop.set_selection_start(start)
        for voice in self.voices:
            voice.set_selection_start(start)

    def set_selection_size_samples(self, size: int) -> None:
        self.loop.set_selection_size(size)
        for voice in self.voices:
            voice.set_selection_size(size)

    def set_grain_coeff(self, coeff: float) -> None:
        coeff = float(np.clip(coeff, 1.0, self.config.max_grain_duration_coeff))
        self.loop.set_grains_duration_coeff(coeff)
        for voice in self.voices:
            voice.set_grains_duration_coeff(coeff)

    def set_filter_cutoff(self, hz: float) -> None:
        self.filter.set_cutoff(hz)

    def loop_on(self) -> None:
        self.loop.note_on(1.0)

    def loop_off(self) -> None:
        self.loop.note_off()

    def note_on(self, midi_note: int) -> None:
        rate = midi_note_ratio(midi_note)
        for i, assigned in enumerate(self.midi_notes):
            if assigned == midi_note:
                self.voices[i].note_on(rate)
                return
        for i, assigned in enumerate(self.midi_notes):
            if assigned == NO_MIDI_NOTE:
                self.voices[i].note_on(rate)
                self.midi_notes[i] = midi_note
                return

    def note_off(self, midi_note: int) -> None:
        for i, assigned in enumerate(self.midi_notes):
            if assigned == midi_note and not self.voices[i].is_idle():
                self.voices[i].note_off()
                return

    def process(self, inp: np.ndarray) -> np.ndarray:
        frames = inp.shape[0]
        if self.mix.shape[0] != frames:
            self.mix = np.zeros(frames, dtype=np.float32)
            self.temp = np.zeros(frames, dtype=np.float32)
            self.monitor = np.zeros(frames, dtype=np.float32)
        if self.recording:
            self._record(inp)
        self.mix.fill(0.0)
        if not self.loop.is_idle():
            self.loop.process(self.mix, self.temp, frames)
        for i, voice in enumerate(self.voices):
            if voice.is_idle():
                continue
            voice.process(self.mix, self.temp, frames)
            if voice.is_idle():
                self.midi_notes[i] = NO_MIDI_NOTE
        filtered = self.filter.process(self.mix)
        np.nan_to_num(filtered, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(filtered, -1.0, 1.0, out=filtered)
        self.monitor = filtered.astype(np.float32, copy=True)
        return self.monitor

    def _record(self, inp: np.ndarray) -> None:
        n = self.buffer.shape[0]
        remaining = n - self.write_pos
        if remaining <= 0:
            self.recording = False
            return
        block = inp[:remaining].copy()
        write_pos = self.write_pos
        if write_pos < self.env_ramp_len:
            for i in range(min(self.env_ramp_len, block.shape[0])):
                block[i] *= self.env_ramp
                self.env_ramp = min(1.0, self.env_ramp + self.env_ramp_rate)
        elif write_pos + block.shape[0] > self.env_decay_start:
            start_i = max(write_pos, self.env_decay_start) - write_pos
            for i in range(start_i, block.shape[0]):
                block[i] *= self.env_ramp
                self.env_ramp = max(0.0, self.env_ramp - self.env_ramp_rate)
        self.buffer[write_pos : write_pos + block.shape[0]] = block
        for sample in block:
            if sample < self.chunk_min:
                self.chunk_min = float(sample)
            if sample > self.chunk_max:
                self.chunk_max = float(sample)
            at_end = write_pos + 1 >= n
            if self.chunk_counter >= self.samples_per_chunk or at_end:
                self.record_chunks.append(
                    RecordChunk(self.chunk_index, self.chunk_min, self.chunk_max)
                )
                self.chunk_index += 1
                self.chunk_min = 1.0
                self.chunk_max = -1.0
                self.chunk_counter = 0
            else:
                self.chunk_counter += 1
            write_pos += 1
        self.write_pos = write_pos
        if self.write_pos >= n:
            self.recording = False
            peak, gain = normalize_recording(self.buffer)
            self.record_peak = peak
            self.record_gain = gain


class AudioEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        rng = random.Random(0xC0111D05)
        self._stereo = np.zeros((config.block_size, 2), dtype=np.float32)
        self.waves = [WaveEngine(config, rng) for _ in range(NUM_WAVES)]

    def process_stereo(self, indata: np.ndarray) -> np.ndarray:
        frames = indata.shape[0]
        if self._stereo.shape[0] != frames:
            self._stereo = np.zeros((frames, 2), dtype=np.float32)
        signals = wave_inputs(indata, len(self.waves))
        for i, wave in enumerate(self.waves):
            self._stereo[:, i] = wave.process(np.asarray(signals[i], dtype=np.float32))
        return self._stereo
