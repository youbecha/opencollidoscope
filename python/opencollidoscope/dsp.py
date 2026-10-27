# Copyright (C) 2016 Queen Mary University of London
# Author: Fiore Martin, based on SuperCollider TGrains and Ross Bencina
# Python port for current CPython / Raspberry Pi OS.
#
# This file is part of Collidoscope.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import enum
import math
from collections.abc import Callable

import numpy as np

CHROMATIC_RATIOS = np.array(
    [
        1.0,
        1.0594630943591,
        1.1224620483089,
        1.1892071150019,
        1.2599210498937,
        1.3348398541685,
        1.4142135623711,
        1.4983070768743,
        1.5874010519653,
        1.6817928305039,
        1.7817974362766,
        1.8877486253586,
    ],
    dtype=np.float64,
)

DEFAULT_ATTENUATION = 0.25118864315096  # -12 dB
MIN_GRAIN_DURATION = 640
MAX_GRAINS = 32


class EnvState(enum.Enum):
    ATTACK = "attack"
    SUSTAIN = "sustain"
    RELEASE = "release"
    IDLE = "idle"


class EnvASR:
    """Linear ASR envelope, matching EnvASR.h (STK-style)."""

    def __init__(
        self,
        sustain_level: float,
        attack_time: float,
        release_time: float,
        sample_rate: int,
    ) -> None:
        attack_time = max(attack_time, 0.001)
        release_time = max(release_time, 0.001)
        self.sustain_level = float(sustain_level)
        self.attack_rate = 1.0 / (attack_time * sample_rate)
        self.release_rate = 1.0 / (release_time * sample_rate)
        self.state = EnvState.IDLE
        self.value = 0.0

    def tick(self) -> float:
        if self.state is EnvState.IDLE:
            self.value = 0.0
        elif self.state is EnvState.ATTACK:
            self.value += self.attack_rate
            if self.value >= self.sustain_level:
                self.value = self.sustain_level
                self.state = EnvState.SUSTAIN
        elif self.state is EnvState.RELEASE:
            self.value -= self.release_rate
            if self.value <= 0.0:
                self.value = 0.0
                self.state = EnvState.IDLE
        return self.value

    def process_block(self, n: int, out: np.ndarray) -> tuple[int, bool]:
        """Fill ``out[:n]``. Returns (samples_written, became_idle)."""
        became_idle = False
        written = 0
        for i in range(n):
            out[i] = self.tick()
            written += 1
            if self.state is EnvState.IDLE and i > 0:
                became_idle = True
                break
            if self.state is EnvState.IDLE and i == 0 and out[i] == 0.0:
                became_idle = True
                break
        return written, became_idle


def midi_note_ratio(midi_note: int) -> float:
    """Pitch ratio vs MIDI 60 (middle C), matching AudioEngine.cpp."""
    distance = midi_note - 60
    if distance < 0:
        amount = -distance
        octaves, intervals = divmod(amount, 12)
        return (0.5**octaves) / float(CHROMATIC_RATIOS[intervals])
    octaves, intervals = divmod(distance, 12)
    return (2.0**octaves) * float(CHROMATIC_RATIOS[intervals])


def filter_cutoff_from_midi(midi_val: float, min_hz: float, max_hz: float) -> float:
    """Original mapping: pow(maxCutoff/200, midi/127) * minCutoff."""
    return (max_hz / 200.0) ** (midi_val / 127.0) * min_hz


def capture_gain(level: float, target: float = 0.45, max_gain: float = 80.0) -> float:
    """Boost a quiet microphone toward `target`. Leave an already-hot signal alone."""
    if not math.isfinite(level) or level < 1e-5:
        return 1.0
    return float(min(max_gain, max(1.0, target / level)))


def normalize_recording(
    buffer: np.ndarray, target: float = 0.8, max_gain: float = 80.0
) -> tuple[float, float]:
    """Scale a finished recording up to `target`. Returns the pre-gain peak and the gain."""
    peak = float(np.max(np.abs(buffer))) if buffer.size else 0.0
    if not math.isfinite(peak) or peak < 1e-4:
        return peak, 1.0
    gain = min(max_gain, target / peak)
    if gain > 1.01:
        buffer *= np.float32(gain)
        return peak, float(gain)
    return peak, 1.0


def wave_inputs(indata: np.ndarray, waves: int = 2) -> list[np.ndarray]:
    """One input channel per wave. A mono device feeds every wave."""
    if indata.ndim == 1 or indata.shape[1] == 1:
        mono = indata if indata.ndim == 1 else indata[:, 0]
        return [mono for _ in range(waves)]
    channels = indata.shape[1]
    return [indata[:, index if index < channels else 0] for index in range(waves)]


class BiquadLowpass:
    """RBJ low-pass, Q = 0.707, replacing Cinder FilterLowPassNode."""

    def __init__(self, sample_rate: int, cutoff_hz: float, q: float = 0.707) -> None:
        self.sample_rate = sample_rate
        self.q = q
        self._z1 = 0.0
        self._z2 = 0.0
        self._coeff = (1.0, 0.0, 0.0, 0.0, 0.0)
        self.set_cutoff(cutoff_hz)

    def set_cutoff(self, cutoff_hz: float) -> None:
        nyquist = self.sample_rate * 0.5 - 1.0
        cutoff_hz = float(np.clip(cutoff_hz, 20.0, nyquist))
        if not math.isfinite(cutoff_hz):
            return
        w0 = 2.0 * math.pi * cutoff_hz / self.sample_rate
        alpha = math.sin(w0) / (2.0 * self.q)
        cos_w0 = math.cos(w0)
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        # One assignment so the audio thread never mixes coefficients from two cutoffs.
        # A torn set (loop playing while the filter moves) makes this recurrence explode.
        self._coeff = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        z1, z2 = self._z1, self._z2
        b0, b1, b2, a1, a2 = self._coeff
        for i in range(x.shape[0]):
            x_n = float(x[i])
            out = b0 * x_n + z1
            z1 = b1 * x_n - a1 * out + z2
            z2 = b2 * x_n - a2 * out
            y[i] = out
        if y.size and not (
            math.isfinite(float(y[-1])) and math.isfinite(z1) and math.isfinite(z2)
        ):
            y.fill(0)
            z1 = 0.0
            z2 = 0.0
        self._z1, self._z2 = z1, z2
        return y


class PGrain:
    __slots__ = ("phase", "rate", "alive", "age", "duration")

    def __init__(self) -> None:
        self.phase = 0.0
        self.rate = 1.0
        self.alive = False
        self.age = 0
        self.duration = 1


class PGranular:
    """Selection-loop granular engine from PGranular.h."""

    def __init__(
        self,
        buffer: np.ndarray,
        sample_rate: int,
        rand_offset: Callable[[], int],
        trigger_callback: Callable[[str, int], None],
        grain_id: int,
    ) -> None:
        self.buffer = buffer
        self.buffer_len = int(buffer.shape[0])
        self.rand_offset = rand_offset
        self.trigger_callback = trigger_callback
        self.grain_id = grain_id
        self.env = EnvASR(1.0, 0.01, 0.05, sample_rate)
        self.attenuation = DEFAULT_ATTENUATION
        self.grains_rate = 1.0
        self.trigger = 0
        self.trigger_rate = 0
        self.grains_start = 0
        self.grains_duration = MIN_GRAIN_DURATION
        self.grains_duration_coeff = 1.0
        self.grains = [PGrain() for _ in range(MAX_GRAINS)]
        self.num_alive = 0
        self._scratch_n = 0
        self._ramp = np.empty(0, dtype=np.float64)
        self._phase = np.empty(0, dtype=np.float64)
        self._frac = np.empty(0, dtype=np.float64)
        self._sample = np.empty(0, dtype=np.float64)
        self._window = np.empty(0, dtype=np.float64)
        self._left = np.empty(0, dtype=np.float32)
        self._right = np.empty(0, dtype=np.float32)
        self._idx = np.empty(0, dtype=np.intp)
        self._nxt = np.empty(0, dtype=np.intp)

    def is_idle(self) -> bool:
        return self.env.state is EnvState.IDLE

    def set_grains_duration_coeff(self, coeff: float) -> None:
        self.grains_duration_coeff = coeff
        self.grains_duration = max(
            MIN_GRAIN_DURATION, int(round(self.trigger_rate * coeff))
        )

    def set_grains_rate(self, rate: float) -> None:
        self.grains_rate = rate

    def set_selection_start(self, start: int) -> None:
        self.grains_start = start

    def set_selection_size(self, size: int) -> None:
        size = max(size, MIN_GRAIN_DURATION)
        self.trigger_rate = size
        self.grains_duration = max(
            MIN_GRAIN_DURATION, int(round(size * self.grains_duration_coeff))
        )

    def note_on(self, rate: float) -> None:
        if self.env.state is EnvState.IDLE:
            if self.trigger_rate < MIN_GRAIN_DURATION:
                self.trigger_rate = MIN_GRAIN_DURATION
            self.set_grains_rate(rate)
            self.env.state = EnvState.ATTACK

    def note_off(self) -> None:
        if self.env.state is not EnvState.IDLE:
            self.env.state = EnvState.RELEASE

    def process(self, audio_out: np.ndarray, temp: np.ndarray, num_samples: int) -> None:
        env = self.env
        if env.state is EnvState.SUSTAIN:
            temp[:num_samples] = env.value
            self._process_grains(audio_out, temp, num_samples)
            return
        env_samples = 0
        became_idle = False
        tick = env.tick
        for i in range(num_samples):
            temp[i] = tick()
            env_samples += 1
            if env.state is EnvState.IDLE:
                became_idle = True
                break
            if env.state is EnvState.SUSTAIN:
                rest = i + 1
                if rest < num_samples:
                    temp[rest:num_samples] = env.value
                env_samples = num_samples
                break
        self._process_grains(audio_out, temp, env_samples)
        if became_idle:
            self.trigger_callback("e", self.grain_id)
            self._reset()

    def _ensure_scratch(self, n: int) -> None:
        if n <= self._scratch_n:
            return
        self._scratch_n = n
        self._ramp = np.arange(n, dtype=np.float64)
        self._phase = np.empty(n, dtype=np.float64)
        self._frac = np.empty(n, dtype=np.float64)
        self._sample = np.empty(n, dtype=np.float64)
        self._window = np.empty(n, dtype=np.float64)
        self._left = np.empty(n, dtype=np.float32)
        self._right = np.empty(n, dtype=np.float32)
        self._idx = np.empty(n, dtype=np.intp)
        self._nxt = np.empty(n, dtype=np.intp)

    def _process_grains(
        self, audio_out: np.ndarray, envelope: np.ndarray, num_samples: int
    ) -> None:
        grain_idx = 0
        while grain_idx < self.num_alive:
            grain = self.grains[grain_idx]
            self._synthesize_grain(grain, audio_out, envelope, num_samples)
            if not grain.alive:
                self.grains[grain_idx] = self.grains[self.num_alive - 1]
                self.grains[self.num_alive - 1] = grain
                self.num_alive -= 1
            else:
                grain_idx += 1

        if self.trigger_rate == 0:
            return

        rand_offset = self.rand_offset()
        new_triggered = False
        while self.trigger < num_samples:
            if self.num_alive < MAX_GRAINS:
                grain = self.grains[self.num_alive]
                self.num_alive += 1
                phase = self.grains_start + float(rand_offset)
                if phase >= self.buffer_len:
                    phase -= self.buffer_len
                grain.phase = phase
                grain.rate = self.grains_rate
                grain.alive = True
                grain.age = 0
                grain.duration = self.grains_duration
                onset = self.trigger
                self._synthesize_grain(
                    grain,
                    audio_out[onset:],
                    envelope[onset:],
                    num_samples - onset,
                )
                if not grain.alive:
                    self.num_alive -= 1
                new_triggered = True
            self.trigger += self.trigger_rate
        self.trigger -= num_samples
        if new_triggered:
            self.trigger_callback("t", self.grain_id)

    def _synthesize_grain(
        self,
        grain: PGrain,
        audio_out: np.ndarray,
        envelope: np.ndarray,
        num_samples: int,
    ) -> None:
        n = min(num_samples, grain.duration - grain.age)
        if n <= 0:
            grain.alive = False
            return
        self._ensure_scratch(n)
        ramp = self._ramp[:n]
        phase = self._phase[:n]
        frac = self._frac[:n]
        sample = self._sample[:n]
        window = self._window[:n]
        idx = self._idx[:n]
        nxt = self._nxt[:n]
        np.multiply(ramp, grain.rate, out=phase)
        phase += grain.phase
        np.remainder(phase, self.buffer_len, out=phase)
        np.copyto(idx, phase, casting="unsafe")
        np.subtract(phase, idx, out=frac)
        np.add(idx, 1, out=nxt)
        np.remainder(nxt, self.buffer_len, out=nxt)
        np.take(self.buffer, idx, out=self._left[:n])
        np.take(self.buffer, nxt, out=self._right[:n])
        last_phase = float(phase[n - 1])
        np.subtract(1.0, frac, out=sample)
        sample *= self._left[:n]
        np.multiply(frac, self._right[:n], out=phase)
        sample += phase
        np.add(ramp, grain.age + 1, out=window)
        window *= math.pi / grain.duration
        np.sin(window, out=window)
        sample *= window
        np.multiply(sample, envelope[:n], out=sample)
        sample *= self.attenuation
        audio_out[:n] += sample
        grain.age += n
        grain.phase = (last_phase + grain.rate) % self.buffer_len
        if grain.age >= grain.duration:
            grain.alive = False

    def _reset(self) -> None:
        self.trigger = 0
        for grain in self.grains[: self.num_alive]:
            grain.alive = False
        self.num_alive = 0
