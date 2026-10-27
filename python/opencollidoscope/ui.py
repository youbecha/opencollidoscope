# Copyright (C) 2016 Queen Mary University of London
# Author: Fiore Martin
# Python port for current CPython / Raspberry Pi OS.
#
# This file is part of Collidoscope.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
import wave
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .config import NUM_WAVES, WAVE_COLORS, Config
from .dsp import capture_gain, filter_cutoff_from_midi
from .engine import AudioEngine, WaveState
from .midi_io import (
    CC_NAMES,
    MIDIInput,
    MIDIMessage,
    pitchbend_level,
    pitchbend_value,
    selection_chunk_from_pitchbend,
)

if TYPE_CHECKING:
    import pygame

log = logging.getLogger("opencollidoscope")

# Computer-keyboard chromatic map from C3, same idea as a MIDI keyboard substitute.
PC_KEYS = "zsxdcvgbhnjm"
PC_BASE = 48
CHUNK_GAP = 2
CHUNK_WIDTH = 7
SHRINK = 3.0 / 5.0


def _soft_device_rank(name: str, hostapi: str) -> int | None:
    """Lower is better. None means this is not a desktop routing device."""
    blob = f"{hostapi} {name}".lower()
    if "pipewire" in blob:
        return 0
    if "pulse" in blob:
        return 1
    lowered = name.lower()
    if lowered == "default" or lowered.startswith("default "):
        return 2
    return None


def choose_audio_devices(
    devices: list[dict],
    default_input: int | None,
    default_output: int | None,
) -> tuple[int | None, int | None]:
    """Pick input and output devices.

    File players on Raspberry Pi OS go through PipeWire or Pulse. A duplex
    PortAudio stream skips that and opens a raw card (often HDMI), which is
    silent when the speakers are the desktop default.
    """

    def pick(kind: str, fallback: int | None) -> int | None:
        channels_key = "max_input_channels" if kind == "in" else "max_output_channels"
        best: tuple[int, int] | None = None
        for dev in devices:
            if int(dev.get(channels_key, 0)) < 1:
                continue
            rank = _soft_device_rank(str(dev.get("name", "")), str(dev.get("hostapi", "")))
            if rank is None:
                continue
            candidate = (rank, int(dev["index"]))
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return best[1]
        if fallback is not None and fallback >= 0:
            for dev in devices:
                if int(dev["index"]) == int(fallback) and int(dev.get(channels_key, 0)) >= 1:
                    return int(fallback)
        for dev in devices:
            if int(dev.get(channels_key, 0)) >= 1:
                return int(dev["index"])
        return None

    return pick("in", default_input), pick("out", default_output)


def find_audio_device(devices: list[dict], query: str, kind: str) -> int | None:
    """Match a PortAudio device by index or by a case-insensitive name fragment."""
    channels_key = "max_input_channels" if kind == "in" else "max_output_channels"
    text = query.strip()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        for dev in devices:
            if int(dev["index"]) == index and int(dev.get(channels_key, 0)) >= 1:
                return index
        return None
    needle = text.lower()
    for dev in devices:
        if int(dev.get(channels_key, 0)) < 1:
            continue
        blob = f"{dev.get('hostapi', '')} {dev.get('name', '')}".lower()
        if needle in blob:
            return int(dev["index"])
    return None


def hardware_output_channels(max_channels: int, name: str, hostapi: str) -> int:
    """How many playback channels to open.

    A Scarlett 2i4 is a fixed 4-channel USB device. Opening it as stereo makes
    ALSA insert a converter, which raises CPU use and then drops the stream.
    Pulse and PipeWire accept a stereo request, so those stay at two channels.
    """
    available = max(1, int(max_channels))
    if _soft_device_rank(name, hostapi) is None:
        return min(available, 8)
    return 2 if available >= 2 else 1


def audio_stream_timing(name: str, hostapi: str) -> tuple[int, float | str]:
    """Hardware cards need a longer callback than PipeWire or Pulse.

    A 256-frame block is about 5 ms at 48 kHz. Python grain synthesis often
    misses that deadline on a USB device, PortAudio then aborts the stream,
    and the retry traffic stalls the UI. Desktop routers already buffer.
    """
    if _soft_device_rank(name, hostapi) is None:
        return 1024, 0.1
    return 256, "high"


def format_audio_devices(devices: list[dict]) -> str:
    lines = []
    for dev in devices:
        lines.append(
            f"[{dev['index']}] {dev.get('hostapi', '')}  "
            f"in {int(dev.get('max_input_channels', 0))} "
            f"out {int(dev.get('max_output_channels', 0))}  "
            f"{dev.get('name', '')}"
        )
    return "\n".join(lines)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> float:
    """Write float audio in -1..1 to a 16-bit WAV. Returns the peak level."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    pcm = np.clip(np.nan_to_num(audio), -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return peak


def lerp(value: float, in0: float, in1: float, out0: float, out1: float) -> float:
    if in1 == in0:
        return out0
    t = (value - in0) / (in1 - in0)
    return out0 + t * (out1 - out0)


def parse_kernel_devices(text: str) -> tuple[list[str], list[str]]:
    """Keyboard and mouse names from /proc/bus/input/devices."""
    keyboards: list[str] = []
    mice: list[str] = []
    name = ""
    handlers = ""

    def finish() -> None:
        nonlocal name, handlers
        tokens = handlers.split()
        if name and "kbd" in tokens:
            keyboards.append(name)
        if name and any(token == "mouse" or token.startswith("mouse") for token in tokens):
            mice.append(name)
        name = ""
        handlers = ""

    for line in text.splitlines():
        if line.startswith("N: Name="):
            finish()
            raw = line.split("=", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
                raw = raw[1:-1]
            name = raw
        elif line.startswith("H: Handlers="):
            handlers = line.split("=", 1)[1].strip()
        elif not line.strip():
            finish()
    finish()
    return keyboards, mice


def parse_kernel_keyboards(text: str) -> list[str]:
    """Names of kernel input devices whose handlers include a keyboard."""
    return parse_kernel_devices(text)[0]


def read_kernel_inputs() -> tuple[str, list[str], list[str]]:
    """('ok', keyboards, mice) on Linux, or ('unavailable', [], []) elsewhere."""
    try:
        with open("/proc/bus/input/devices", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return "unavailable", [], []
    keyboards, mice = parse_kernel_devices(text)
    return "ok", keyboards, mice


class CollidoscopeApp:
    def __init__(
        self,
        width: int,
        height: int,
        config: Config | None = None,
        debug: bool = False,
        debug_log: str | None = None,
        input_query: str | None = None,
        output_query: str | None = None,
    ) -> None:
        self.config = config or Config()
        self.width = width
        self.height = height
        self.engine = AudioEngine(self.config)
        self.midi = MIDIInput()
        self.states = [
            WaveState(
                chunks=[(0.0, 0.0)] * self.config.num_chunks,
                chunk_animate=[0.0] * self.config.num_chunks,
                oscilloscope=np.zeros(max(8, self.config.block_size // self.config.oscilloscope_divider)),
            )
            for _ in range(NUM_WAVES)
        ]
        for i, wave in enumerate(self.engine.waves):
            size = self.states[i].selection_size_chunks * self.config.samples_per_chunk()
            start = self.states[i].selection_start_chunk * self.config.samples_per_chunk()
            wave.set_selection_size_samples(size)
            wave.set_selection_start_samples(start)
        self.active_wave = 0
        self.held_notes: dict[int, int] = {}
        self.stream = None
        self._input_stream = None
        self._output_stream = None
        self._input_device: int | None = None
        self._output_device: int | None = None
        self.audio_input_name = "none"
        self.audio_output_name = "none"
        self.audio_output_hostapi = ""
        self._input_query = input_query
        self._output_query = output_query
        self._audio_latency: float | str = "high"
        self._audio_duplex = False
        self._audio_running = False
        self._stream_finished = False
        self._audio_restarts = 0
        self._audio_restart_at = 0.0
        self._callback_logged = False
        self._out_peak = 0.0
        self._in_peak = 0.0
        self._in_envelope = 0.0
        self._in_gain = 1.0
        self._out_callbacks = 0
        self._out_callbacks_seen = 0
        self._out_callback_seen_at = time.monotonic()
        self._wav_lock = threading.Lock()
        self._capture_output = False
        self._output_chunks: list[np.ndarray] = []
        self._wav_note = ""
        self.fullscreen = False
        self.clock = None
        self.screen = None
        self.font = None
        self.debug_font = None
        self.pygame: object | None = None
        self.debug_overlay = debug
        self.debug_log_enabled = debug
        self.debug_log_path = debug_log
        self._debug_handler: logging.Handler | None = None
        self._trace_lines: deque[str] = deque(maxlen=8)
        self._key_last: tuple[float, str] | None = None
        self._key_events = 0
        self._key_focus: bool | None = None
        self._kernel_kbds: list[str] | None = None
        self._kernel_mice: list[str] | None = None
        self._kernel_status = "unavailable"
        self._kernel_poll_at = 0.0
        self._heartbeat_at = 0.0
        self.audio_xruns = 0
        self._audio_status: object | None = None
        self._event_log_at: dict[str, float] = {}
        self._trace_last: dict[str, str] = {}
        self._quiet_events: set[int] = set()

    def run(self) -> int:
        pygame = _import_pygame()
        self.pygame = pygame
        pygame.display.init()
        pygame.font.init()
        if hasattr(pygame, "mixer"):
            pygame.mixer.quit()
        pygame.display.set_caption("OpenCollidoscope")
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.debug_font = pygame.font.Font(None, 18)
        self._quiet_events = {
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEWHEEL,
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.QUIT,
            pygame.VIDEORESIZE,
            pygame.TEXTINPUT,
            getattr(pygame, "TEXTEDITING", -1),
            getattr(pygame, "WINDOWEXPOSED", -1),
            getattr(pygame, "WINDOWMOVED", -1),
            getattr(pygame, "WINDOWENTER", -1),
            getattr(pygame, "WINDOWLEAVE", -1),
            getattr(pygame, "WINDOWSHOWN", -1),
        }
        if self.debug_log_enabled:
            self._ensure_debug_log()
        self.midi.setup()
        if self.debug_log_enabled:
            ports = ", ".join(self.midi.ports) or "none"
            errors = "; ".join(self.midi.errors) or "none"
            log.info("MIDI ports: %s", ports)
            log.info("MIDI errors: %s", errors)
        self._configure_sample_rate()
        self.engine = AudioEngine(self.config)
        for i, wave in enumerate(self.engine.waves):
            size = self.states[i].selection_size_chunks * self.config.samples_per_chunk()
            start = self.states[i].selection_start_chunk * self.config.samples_per_chunk()
            wave.set_selection_size_samples(size)
            wave.set_selection_start_samples(start)
        self._start_audio()
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        self._trace_key("down", event)
                        self._keydown(event)
                    elif event.type == pygame.KEYUP:
                        self._trace_key("up", event)
                        self._keyup(event)
                    elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                        self._mouse_drag(event.pos)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._trace(f"mouse down {tuple(event.pos)}")
                        self._mouse_drag(event.pos)
                    elif event.type == pygame.MOUSEWHEEL:
                        self._trace(f"wheel {event.y}")
                        self._nudge_selection_size(event.y)
                    elif event.type == pygame.VIDEORESIZE:
                        self.width, self.height = event.w, event.h
                        self.screen = pygame.display.set_mode(
                            (self.width, self.height), pygame.RESIZABLE
                        )
                    elif event.type not in self._quiet_events:
                        self._trace_event(pygame.event.event_name(event.type))
                self._poll_focus()
                self._poll_kernel_inputs()
                self._poll_midi()
                self._heartbeat()
                self._poll_audio_health()
                self._drain_audio_ui()
                self._draw()
                self.clock.tick(60)
        finally:
            self._stop_audio()
            self.midi.close()
            if self._debug_handler is not None:
                self._debug_handler.flush()
            pygame.display.quit()
            pygame.font.quit()
        return 0

    def _configure_sample_rate(self) -> None:
        try:
            import sounddevice as sd

            hostapis = sd.query_hostapis()
            devices = []
            for dev in sd.query_devices():
                index = int(dev["index"])
                host = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else ""
                devices.append({**dev, "index": index, "hostapi": host})
            default_in, default_out = sd.default.device
            in_dev, out_dev = choose_audio_devices(devices, default_in, default_out)
            if self._output_query:
                matched = find_audio_device(devices, self._output_query, "out")
                if matched is None:
                    raise SystemExit(
                        f"No output device matches {self._output_query!r}.\n"
                        f"{format_audio_devices(devices)}"
                    )
                out_dev = matched
            if self._input_query:
                matched = find_audio_device(devices, self._input_query, "in")
                if matched is None:
                    raise SystemExit(
                        f"No input device matches {self._input_query!r}.\n"
                        f"{format_audio_devices(devices)}"
                    )
                in_dev = matched
            self._input_device = in_dev
            self._output_device = out_dev
            by_index = {dev["index"]: dev for dev in devices}
            if in_dev is not None and in_dev in by_index:
                self.audio_input_name = str(by_index[in_dev]["name"])
            if out_dev is not None and out_dev in by_index:
                chosen = by_index[out_dev]
                self.audio_output_name = str(chosen["name"])
                rate = int(chosen["default_samplerate"])
                if rate > 0:
                    self.config.sample_rate = rate
                self.audio_output_hostapi = str(chosen.get("hostapi", ""))
                block, latency = audio_stream_timing(
                    self.audio_output_name, self.audio_output_hostapi
                )
                self.config.block_size = block
                self._audio_latency = latency
            log.info(
                "Audio output: [%s] %s block %s latency %s",
                out_dev,
                self.audio_output_name,
                self.config.block_size,
                self._audio_latency,
            )
            log.info("Audio input: [%s] %s", in_dev, self.audio_input_name)
        except Exception as exc:
            log.warning("Using %s Hz (device probe failed: %s)", self.config.sample_rate, exc)
            self._input_device = None
            self._output_device = None

    def _mark_stream_finished(self) -> None:
        self._stream_finished = True

    def _note_callback_error(self) -> None:
        self.audio_xruns += 1
        self._audio_status = "callback error"
        if self._callback_logged:
            return
        self._callback_logged = True
        log.exception("audio callback failed")

    def _start_audio(self) -> None:
        import sounddevice as sd

        cfg = self.config
        out_info = sd.query_devices(self._output_device, "output")
        out_channels = hardware_output_channels(
            int(out_info["max_output_channels"]),
            self.audio_output_name,
            self.audio_output_hostapi,
        )
        in_channels = 1
        if self._input_device is not None:
            in_info = sd.query_devices(self._input_device, "input")
            in_channels = 2 if int(in_info["max_input_channels"]) >= 2 else 1
        same_device = (
            self._input_device is not None
            and self._output_device is not None
            and int(self._input_device) == int(self._output_device)
        )
        self._audio_duplex = same_device
        self._capture = np.zeros((cfg.block_size, in_channels), dtype=np.float32)
        self._stream_finished = False

        def take_input(indata, frames) -> None:  # noqa: ANN001
            cap = self._capture
            n = min(frames, cap.shape[0])
            ch = min(indata.shape[1], cap.shape[1])
            cap[:n, :ch] = indata[:n, :ch]
            if not indata.size:
                return
            raw = float(np.max(np.abs(indata)))
            if raw > self._in_peak:
                self._in_peak = raw
            if raw > self._in_envelope:
                self._in_envelope = raw
            elif self._in_envelope > 0.0:
                self._in_envelope *= 0.999
            gain = capture_gain(self._in_envelope)
            self._in_gain = gain
            if gain != 1.0:
                cap[:n, :ch] *= np.float32(gain)

        def write_output(outdata, frames) -> None:  # noqa: ANN001
            cap = self._capture
            if frames > cap.shape[0]:
                stereo_in = np.zeros((frames, cap.shape[1]), dtype=np.float32)
                stereo_in[: cap.shape[0]] = cap
            else:
                stereo_in = cap[:frames]
            stereo = self.engine.process_stereo(stereo_in)
            if outdata.shape[1] == 1:
                outdata[:, 0] = stereo[:, 0]
            elif NUM_WAVES < 2:
                outdata[:, 0] = stereo[:, 0]
                outdata[:, 1] = stereo[:, 0]
                if outdata.shape[1] > 2:
                    outdata[:, 2:] = 0
            else:
                outdata[:, 0] = stereo[:, 0]
                outdata[:, 1] = stereo[:, 1]
                if outdata.shape[1] > 2:
                    outdata[:, 2:] = 0
            if outdata.size:
                peak = float(np.max(np.abs(outdata)))
                if peak > self._out_peak:
                    self._out_peak = peak
            self._out_callbacks += 1
            if self._capture_output:
                self._keep_output_block(outdata)

        def note_status(status) -> None:  # noqa: ANN001
            if status:
                self.audio_xruns += 1
                self._audio_status = status

        def in_callback(indata, frames, _time, status) -> None:  # noqa: ANN001
            try:
                note_status(status)
                take_input(indata, frames)
            except Exception:
                self._note_callback_error()

        def out_callback(outdata, frames, _time, status) -> None:  # noqa: ANN001
            try:
                note_status(status)
                write_output(outdata, frames)
            except Exception:
                outdata.fill(0)
                self._note_callback_error()

        def duplex_callback(indata, outdata, frames, _time, status) -> None:  # noqa: ANN001
            try:
                note_status(status)
                take_input(indata, frames)
                write_output(outdata, frames)
            except Exception:
                outdata.fill(0)
                self._note_callback_error()

        attempts: list[tuple[int, float | str, int]] = []

        def add_attempt(block: int, latency: float | str, channels: int) -> None:
            item = (block, latency, channels)
            if item not in attempts:
                attempts.append(item)

        add_attempt(cfg.block_size, self._audio_latency, out_channels)
        if out_channels > 2:
            add_attempt(cfg.block_size, self._audio_latency, 2)
        if cfg.block_size != 256:
            add_attempt(256, "high", min(out_channels, 2))
        opened = False
        open_error: Exception | None = None
        for block, latency, channels in attempts:
            cfg.block_size = block
            self._audio_latency = latency
            out_channels = channels
            self._capture = np.zeros((cfg.block_size, in_channels), dtype=np.float32)
            common = {
                "samplerate": cfg.sample_rate,
                "blocksize": cfg.block_size,
                "dtype": "float32",
                "latency": self._audio_latency,
                "finished_callback": self._mark_stream_finished,
            }
            try:
                if same_device:
                    self._output_stream = sd.Stream(
                        device=self._output_device,
                        channels=(in_channels, out_channels),
                        callback=duplex_callback,
                        **common,
                    )
                else:
                    self._output_stream = sd.OutputStream(
                        device=self._output_device,
                        channels=out_channels,
                        callback=out_callback,
                        **common,
                    )
                self._output_stream.start()
                opened = True
                break
            except Exception as exc:
                open_error = exc
                failed = self._output_stream
                self._output_stream = None
                if failed is not None:
                    try:
                        failed.close()
                    except Exception:
                        log.debug("failed audio stream close", exc_info=True)
                log.warning(
                    "audio open failed block %s channels %s: %s",
                    block,
                    channels,
                    exc,
                )
        if not opened:
            raise open_error or RuntimeError("audio output did not open")
        self.stream = self._output_stream
        self._audio_running = True
        mode = "duplex" if same_device else "output"
        log.info(
            "Audio %s stream %s Hz, %s ch, block %s, latency %s, device %s",
            mode,
            cfg.sample_rate,
            out_channels,
            cfg.block_size,
            self._audio_latency,
            self.audio_output_name,
        )
        if same_device or self._input_device is None:
            if self._input_device is None:
                log.warning("No audio input device; recording will be silent")
            return
        try:
            self._input_stream = sd.InputStream(
                device=self._input_device,
                channels=in_channels,
                callback=in_callback,
                samplerate=common["samplerate"],
                blocksize=common["blocksize"],
                dtype=common["dtype"],
                latency=common["latency"],
            )
            self._input_stream.start()
            log.info(
                "Audio input stream %s Hz, %s ch, device %s",
                cfg.sample_rate,
                in_channels,
                self.audio_input_name,
            )
        except Exception as exc:
            log.warning("Audio input not opened (%s); recording will be silent", exc)
            self._input_stream = None

    def _poll_audio_health(self) -> None:
        if not self._audio_running or self._output_stream is None or self._audio_restarts >= 4:
            return
        try:
            active = bool(self._output_stream.active)
        except Exception:
            active = False
        if active and not self._stream_finished:
            return
        now = time.monotonic()
        if now - self._audio_restart_at < 0.75:
            return
        self._audio_restart_at = now
        self._audio_restarts += 1
        log.warning(
            "audio stream stopped (%s); restart %s",
            self._audio_status or "no status",
            self._audio_restarts,
        )
        self._audio_running = False
        self._close_audio_streams()
        try:
            self._start_audio()
        except Exception:
            log.exception("audio restart failed")

    def _close_audio_streams(self) -> None:
        seen: list[object] = []
        for stream in (self._input_stream, self._output_stream):
            if stream is None or stream in seen:
                continue
            seen.append(stream)
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.debug("audio stream close failed", exc_info=True)
        self._input_stream = None
        self._output_stream = None
        self.stream = None

    def _stop_audio(self) -> None:
        self._audio_running = False
        self._close_audio_streams()
        if self._capture_output:
            self._finish_output_capture()

    def _keep_output_block(self, outdata: np.ndarray) -> None:
        limit = max(1, int(self.config.sample_rate * 60 / max(1, self.config.block_size)))
        with self._wav_lock:
            if not self._capture_output or len(self._output_chunks) >= limit:
                return
            self._output_chunks.append(np.array(outdata, dtype=np.float32, copy=True))

    def _save_recordings(self) -> None:
        notes: list[str] = []
        for index, wave in enumerate(self.engine.waves):
            path = f"opencollidoscope-record-{index + 1}.wav"
            peak = write_wav(path, wave.buffer, self.config.sample_rate)
            mic_peak = getattr(wave, "record_peak", 0.0)
            mic_gain = getattr(wave, "record_gain", 1.0)
            notes.append(f"{path} peak {peak:.3f} (mic {mic_peak:.3f}, gain x{mic_gain:.0f})")
            log.info("wrote %s (%.3f peak)", path, peak)
        self._wav_note = "saved " + "  ".join(notes)

    def _toggle_output_capture(self) -> None:
        with self._wav_lock:
            if self._capture_output:
                chunks = self._output_chunks
                self._output_chunks = []
                self._capture_output = False
            else:
                self._output_chunks = []
                self._capture_output = True
                chunks = None
        if chunks is None:
            self._wav_note = "capturing opencollidoscope-output.wav — press F3 to finish"
            log.info("capturing generated audio to opencollidoscope-output.wav")
            return
        self._write_output_chunks(chunks)

    def _finish_output_capture(self) -> None:
        with self._wav_lock:
            chunks = self._output_chunks
            self._output_chunks = []
            self._capture_output = False
        self._write_output_chunks(chunks)

    def _write_output_chunks(self, chunks: list[np.ndarray]) -> None:
        if not chunks:
            self._wav_note = "output capture was empty"
            log.info("output capture was empty")
            return
        audio = np.concatenate(chunks, axis=0)
        path = "opencollidoscope-output.wav"
        peak = write_wav(path, audio, self.config.sample_rate)
        seconds = audio.shape[0] / float(self.config.sample_rate)
        self._wav_note = f"wrote {path} peak {peak:.3f} {seconds:.1f}s"
        log.info("wrote %s (%.3f peak, %.1fs)", path, peak, seconds)

    def _keydown(self, event: pygame.event.Event) -> None:
        pygame = self.pygame
        key = event.key
        if key == pygame.K_F1:
            self.debug_overlay = not self.debug_overlay
            if self.debug_overlay:
                self.debug_log_enabled = True
                self._ensure_debug_log()
            state = "on" if self.debug_overlay else "off"
            self._trace(f"debug overlay {state}")
            if self.debug_overlay:
                names = self._kernel_kbds or ["none"]
                mice = self._kernel_mice or ["none"]
                self._trace(f"kernel keyboards ({self._kernel_status}): {names}")
                self._trace(f"kernel mice ({self._kernel_status}): {mice}")
                self._trace(f"pygame key focus {self._key_focus}")
            return
        if key == pygame.K_F2:
            self._save_recordings()
            return
        if key == pygame.K_F3:
            self._toggle_output_capture()
            return
        if key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return
        if key == pygame.K_TAB:
            self.active_wave = (self.active_wave + 1) % NUM_WAVES
            return
        if key == pygame.K_f:
            self.fullscreen = not self.fullscreen
            flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            return
        if key == pygame.K_r:
            self.engine.waves[self.active_wave].start_record()
            return
        if key == pygame.K_t:
            self.engine.waves[1 if NUM_WAVES > 1 else 0].start_record()
            return
        if key == pygame.K_SPACE:
            self._toggle_loop(self.active_wave)
            return
        if key == pygame.K_LEFT or key == pygame.K_a:
            self._nudge_selection_start(-1)
            return
        if key == pygame.K_RIGHT or key == pygame.K_d:
            self._nudge_selection_start(1)
            return
        if key == pygame.K_DOWN or key == pygame.K_s:
            self._nudge_selection_size(-1)
            return
        if key == pygame.K_UP or key == pygame.K_w:
            self._nudge_selection_size(1)
            return
        if key in (pygame.K_9, pygame.K_COMMA):
            self._nudge_grain(-1)
            return
        if key in (pygame.K_0, pygame.K_PERIOD):
            self._nudge_grain(1)
            return
        name = event.unicode.lower() if event.unicode else ""
        if name in PC_KEYS:
            midi = PC_BASE + PC_KEYS.index(name)
            self.held_notes[key] = midi
            self.engine.waves[self.active_wave].note_on(midi)

    def _keyup(self, event: pygame.event.Event) -> None:
        midi = self.held_notes.pop(event.key, None)
        if midi is not None:
            self.engine.waves[self.active_wave].note_off(midi)

    def _toggle_loop(self, wave_idx: int) -> None:
        state = self.states[wave_idx]
        wave = self.engine.waves[wave_idx]
        state.looping = not state.looping
        if state.looping:
            wave.loop_on()
        else:
            wave.loop_off()

    def _nudge_selection_start(self, delta: int) -> None:
        state = self.states[self.active_wave]
        start = int(np.clip(state.selection_start_chunk + delta, 0, self.config.num_chunks - 1))
        self._apply_selection_start(self.active_wave, start)

    def _nudge_selection_size(self, delta: int) -> None:
        state = self.states[self.active_wave]
        size = int(np.clip(state.selection_size_chunks + delta, 1, self.config.max_selection_chunks))
        self._apply_selection_size(self.active_wave, size)

    def _nudge_grain(self, delta: int) -> None:
        state = self.states[self.active_wave]
        coeff = float(np.clip(round(state.grain_coeff) + delta, 1, 8))
        state.grain_coeff = coeff
        self.engine.waves[self.active_wave].set_grain_coeff(coeff)

    def _apply_selection_start(self, wave_idx: int, start_chunk: int) -> None:
        state = self.states[wave_idx]
        max_start = self.config.num_chunks - 1
        start_chunk = int(np.clip(start_chunk, 0, max_start))
        prev_size = state.selection_size_chunks
        state.selection_start_chunk = start_chunk
        end = min(self.config.num_chunks - 1, start_chunk + prev_size - 1)
        state.selection_size_chunks = max(1, end - start_chunk + 1)
        spc = self.config.samples_per_chunk()
        wave = self.engine.waves[wave_idx]
        wave.set_selection_start_samples(start_chunk * spc)
        if state.selection_size_chunks != prev_size:
            wave.set_selection_size_samples(state.selection_size_chunks * spc)

    def _apply_selection_size(self, wave_idx: int, num_chunks: int) -> None:
        state = self.states[wave_idx]
        num_chunks = int(np.clip(num_chunks, 1, self.config.max_selection_chunks))
        state.selection_size_chunks = num_chunks
        spc = self.config.samples_per_chunk()
        self.engine.waves[wave_idx].set_selection_size_samples(num_chunks * spc)

    def _apply_filter(self, wave_idx: int, midi_val: float) -> None:
        state = self.states[wave_idx]
        state.filter_midi = float(np.clip(midi_val, 0.0, 127.0))
        cutoff = filter_cutoff_from_midi(
            state.filter_midi,
            self.config.min_filter_cutoff_hz,
            self.config.max_filter_cutoff_hz,
        )
        self.engine.waves[wave_idx].set_filter_cutoff(cutoff)
        state.selection_alpha = float(lerp(state.filter_midi, 0.0, 127.0, 0.0, 1.0))

    def _mouse_drag(self, pos: tuple[int, int]) -> None:
        x, y = pos
        wave_idx = 0 if y >= self.height / 2 else 1
        self.active_wave = wave_idx
        start = int(lerp(x, 0, self.width, 0, self.config.num_chunks - 1))
        self._apply_selection_start(wave_idx, start)
        midi_val = lerp(y if wave_idx == 0 else (self.height - y), 0, self.height / 2, 0, 127)
        # Horizontal drag sets grain window; vertical position in the tier sets filter.
        self._apply_filter(wave_idx, midi_val)

    def _poll_midi(self) -> None:
        for msg in self.midi.poll():
            detail = self._handle_midi(msg)
            if not detail:
                continue
            port = f" [{msg.port}]" if msg.port else ""
            line = f"midi{port} {detail}"
            continuous = msg.voice == "pitchbend" or (msg.voice == "cc" and msg.data1 in (1, 2, 7))
            if continuous:
                self._trace(line, key=f"midi:{msg.channel}:{msg.voice}:{msg.data1}", min_interval=0.25)
            else:
                self._trace(line)

    def _handle_midi(self, msg: MIDIMessage) -> str:
        wave_idx = self.config.wave_for_midi_channel(msg.channel)
        channel = f"ch{msg.channel + 1}"
        if wave_idx >= NUM_WAVES:
            return f"{channel} ignored (no wave)"
        wave = self.engine.waves[wave_idx]
        state = self.states[wave_idx]
        wave_name = f"wave{wave_idx + 1}"
        mapped = "" if msg.channel == wave_idx else f" mapped->{wave_name}"
        if msg.voice == "noteon":
            wave.note_on(msg.data1)
            return f"{channel} noteon {msg.data1} vel {msg.data2} -> {wave_name}{mapped}"
        if msg.voice == "noteoff":
            wave.note_off(msg.data1)
            return f"{channel} noteoff {msg.data1} -> {wave_name}{mapped}"
        if msg.voice == "pitchbend":
            value = pitchbend_value(msg.data1, msg.data2)
            level = pitchbend_level(msg.data1, msg.data2)
            self._apply_selection_start(
                wave_idx,
                selection_chunk_from_pitchbend(msg.data1, msg.data2, self.config.num_chunks),
            )
            start = self.states[wave_idx].selection_start_chunk
            return f"{channel} pitchbend {value} level {level}/127 -> {wave_name} start {start}"
        if msg.voice == "cc":
            label = CC_NAMES.get(msg.data1, f"CC{msg.data1}")
            if msg.data1 == 1:
                num = int(lerp(msg.data2, 0, 127, 1, self.config.max_selection_chunks))
                self._apply_selection_size(wave_idx, num)
                size = self.states[wave_idx].selection_size_chunks
                return f"{channel} {label} {msg.data2} -> {wave_name} size {size}"
            if msg.data1 == 4:
                want = msg.data2 > 0
                changed = want != state.looping
                if changed:
                    self._toggle_loop(wave_idx)
                looping = self.states[wave_idx].looping
                how = "changed" if changed else "unchanged"
                return f"{channel} {label} {msg.data2} -> {wave_name} looping {looping} ({how})"
            if msg.data1 == 5:
                wave.start_record()
                return f"{channel} {label} {msg.data2} -> {wave_name} record"
            if msg.data1 == 2:
                coeff = lerp(msg.data2, 0, 127, 1.0, self.config.max_grain_duration_coeff)
                state.grain_coeff = float(coeff)
                wave.set_grain_coeff(state.grain_coeff)
                return f"{channel} {label} {msg.data2} -> {wave_name} grain {state.grain_coeff:.2f}"
            if msg.data1 == 7:
                self._apply_filter(wave_idx, msg.data2)
                return (
                    f"{channel} {label} {msg.data2} -> {wave_name} "
                    f"cutoff midi {state.filter_midi:.0f}"
                )
            return f"{channel} IGNORED {label} value {msg.data2}"
        return (
            f"{channel} IGNORED {msg.voice} {msg.data1} {msg.data2} "
            f"status 0x{msg.status:02X}"
        )

    def _drain_audio_ui(self) -> None:
        for i, wave in enumerate(self.engine.waves):
            state = self.states[i]
            while wave.record_chunks:
                chunk = wave.record_chunks.popleft()
                if chunk.index < 0:
                    state.chunks = [(0.0, 0.0)] * self.config.num_chunks
                    state.chunk_animate = [0.0] * self.config.num_chunks
                    state.applied_record_gain = 1.0
                elif chunk.index < self.config.num_chunks:
                    state.chunks[chunk.index] = (chunk.bottom, chunk.top)
                    state.chunk_animate[chunk.index] = 0.01
            record_gain = getattr(wave, "record_gain", 1.0)
            if record_gain != state.applied_record_gain and state.applied_record_gain > 0:
                scale = record_gain / state.applied_record_gain
                state.chunks = [(bottom * scale, top * scale) for bottom, top in state.chunks]
                state.applied_record_gain = record_gain
            while wave.cursor_events:
                ev = wave.cursor_events.popleft()
                if ev.cmd == "update":
                    state.cursor_started[ev.synth_id] = time.monotonic()
                else:
                    state.cursor_started.pop(ev.synth_id, None)
            seconds_per_chunk = self.config.samples_per_chunk() / float(self.config.sample_rate)
            now = time.monotonic()
            selection_end = min(
                self.config.num_chunks - 1,
                state.selection_start_chunk + state.selection_size_chunks - 1,
            )
            state.cursors.clear()
            for synth_id, started in list(state.cursor_started.items()):
                pos = state.selection_start_chunk + int((now - started) / seconds_per_chunk)
                if pos < state.selection_start_chunk or pos > selection_end:
                    continue
                state.cursors[synth_id] = pos
            mon = wave.monitor
            pts = state.oscilloscope.shape[0]
            if mon.size:
                idx = np.linspace(0, max(0, mon.size - 1), pts).astype(np.intp)
                sampled = np.asarray(mon[idx], dtype=np.float32)
                if sampled.size and float(np.max(np.abs(sampled))) > 0.002:
                    state.oscilloscope = sampled
            for j, anim in enumerate(state.chunk_animate):
                if anim < 1.0 and (state.chunks[j][0] != 0.0 or state.chunks[j][1] != 0.0):
                    state.chunk_animate[j] = min(1.0, anim + 0.3333)

    def _draw(self) -> None:
        pygame = self.pygame
        screen = self.screen
        screen.fill((0, 0, 0))
        for i in range(NUM_WAVES):
            self._draw_wave(i)
        if self._capture_output or self._wav_note:
            note = self._wav_note
            if self._capture_output:
                with self._wav_lock:
                    frames = sum(chunk.shape[0] for chunk in self._output_chunks)
                seconds = frames / float(self.config.sample_rate)
                note = f"CAPTURING opencollidoscope-output.wav {seconds:.1f}s — F3 finishes"
            screen.blit(self.font.render(note, True, (255, 220, 120)), (10, 8))
        if self.debug_overlay:
            self._draw_debug()
        pygame.display.flip()

    def _output_level_line(self) -> tuple[tuple[int, int, int], str]:
        """What the output callback is writing, separate from the recorded waveform."""
        out_peak = self._out_peak
        in_peak = self._in_peak
        self._out_peak *= 0.85
        self._in_peak *= 0.85
        callbacks = self._out_callbacks
        callback_delta = callbacks - self._out_callbacks_seen
        self._out_callbacks_seen = callbacks
        if callback_delta > 0:
            self._out_callback_seen_at = time.monotonic()
        stalled = time.monotonic() - self._out_callback_seen_at > 0.5
        playing = False
        parts: list[str] = []
        for index, wave in enumerate(self.engine.waves):
            notes = sum(1 for voice in wave.voices if not voice.is_idle())
            looping = not wave.loop.is_idle()
            if looping or notes:
                playing = True
            grain = self.states[index].grain_coeff
            parts.append(
                f"w{index + 1} {'loop' if looping else 'idle'} notes {notes} grain {grain:.1f}"
            )
        stream = self._output_stream
        if stream is None:
            state = "NO OUTPUT STREAM"
            color = (255, 110, 110)
        elif not stream.active or stalled:
            state = "OUTPUT STOPPED"
            color = (255, 110, 110)
        elif out_peak < 0.0008 and playing:
            state = "PLAYING BUT SILENT"
            color = (255, 186, 80)
        elif out_peak < 0.0008:
            state = "SILENCE"
            color = (180, 180, 180)
        else:
            state = "GENERATING"
            color = (140, 220, 160)
        text = (
            f"audio {state}  out {out_peak:.3f}  in {in_peak:.3f} x{self._in_gain:.0f}  "
            f"device {self.audio_output_name}  " + "  ".join(parts)
        )
        return color, _clip(text, 180)

    def _ensure_debug_log(self) -> None:
        if self._debug_handler is not None:
            return
        path = self.debug_log_path or "opencollidoscope-debug.log"
        self.debug_log_path = path
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("opencollidoscope").addHandler(handler)
        self._debug_handler = handler
        log.info("debug log %s", path)

    def _trace(self, line: str, *, key: str | None = None, min_interval: float = 0.0, keep: bool = False) -> None:
        watching = self.debug_overlay or self.debug_log_enabled
        if not watching and not keep:
            return
        now = time.monotonic()
        if key is not None:
            last_at = self._event_log_at.get(key, 0.0)
            if line == self._trace_last.get(key) or now - last_at < min_interval:
                return
            self._event_log_at[key] = now
            self._trace_last[key] = line
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        self._trace_lines.append(stamped)
        if self.debug_log_enabled:
            log.info("%s", line)

    def _trace_key(self, kind: str, event: pygame.event.Event) -> None:
        name = self.pygame.key.name(event.key)
        text = event.unicode or ""
        line = f"key {kind} {name} unicode={text!r}"
        self._key_last = (time.monotonic(), line)
        self._key_events += 1
        self._trace(line)

    def _trace_event(self, name: str) -> None:
        self._trace(f"event {name}", key=f"event:{name}", min_interval=1.0)

    def _poll_focus(self) -> None:
        focused = bool(self.pygame.key.get_focused())
        if focused == self._key_focus:
            return
        previous = self._key_focus
        self._key_focus = focused
        if previous is not None:
            self._trace(f"pygame key focus {previous} -> {focused}", keep=True)

    def _poll_kernel_inputs(self) -> None:
        now = time.monotonic()
        if now - self._kernel_poll_at < 0.5:
            return
        self._kernel_poll_at = now
        status, keyboards, mice = read_kernel_inputs()
        self._kernel_status = status
        if self._kernel_kbds is None:
            self._kernel_kbds = keyboards
            self._kernel_mice = mice
            self._trace(f"kernel keyboards ({status}): {keyboards or ['none']}", keep=True)
            self._trace(f"kernel mice ({status}): {mice or ['none']}", keep=True)
            return
        if keyboards != self._kernel_kbds:
            self._trace(
                f"kernel keyboards changed from {self._kernel_kbds or ['none']} to {keyboards or ['none']}",
                keep=True,
            )
            self._kernel_kbds = keyboards
        if mice != self._kernel_mice:
            self._trace(
                f"kernel mice changed from {self._kernel_mice or ['none']} to {mice or ['none']}",
                keep=True,
            )
            self._kernel_mice = mice

    def _heartbeat(self) -> None:
        if not self.debug_log_enabled:
            return
        now = time.monotonic()
        if now - self._heartbeat_at < 5.0:
            return
        self._heartbeat_at = now
        stats = self.midi.snapshot()
        if self._key_last is None:
            age = "none"
        else:
            age = f"{now - self._key_last[0]:.1f}s"
        waves = " ".join(
            (
                f"w{index + 1}:start={state.selection_start_chunk},"
                f"size={state.selection_size_chunks},grain={state.grain_coeff:.2f},"
                f"filt={state.filter_midi:.0f},loop={state.looping},"
                f"rec={self.engine.waves[index].recording}"
            )
            for index, state in enumerate(self.states)
        )
        log.info(
            "heartbeat focus=%s keyboards=%s mice=%s midi_rx=%s dropped=%s cc=%s bend=%s other=%s xruns=%s last=%s last_key=%s %s",
            self._key_focus,
            self._kernel_kbds,
            self._kernel_mice,
            stats.received,
            stats.dropped,
            dict(stats.cc_counts),
            dict(stats.bend_counts),
            stats.last_other or "none",
            self.audio_xruns,
            self._audio_status or "none",
            age,
            waves,
        )

    def _kernel_rows(
        self,
        dim: tuple[int, int, int],
        warn: tuple[int, int, int],
        bad: tuple[int, int, int],
        ok: tuple[int, int, int],
    ) -> list[tuple[tuple[int, int, int], str]]:
        checked = time.monotonic() - self._kernel_poll_at if self._kernel_poll_at else 0.0
        stamp = f"  checked {checked:.1f}s ago"
        rows: list[tuple[tuple[int, int, int], str]] = []
        if self._kernel_status != "ok":
            rows.append((warn, "kernel inputs: unavailable (no /proc/bus/input/devices)"))
            return rows
        keyboards = self._kernel_kbds or []
        mice = self._kernel_mice or []
        if keyboards:
            kbd_color, kbd_text = ok, "kernel keyboards: " + " | ".join(keyboards)
        else:
            kbd_color, kbd_text = bad, "kernel keyboards: NONE (USB keyboard not present)"
        if mice:
            mouse_color, mouse_text = ok, "kernel mice: " + " | ".join(mice)
        else:
            mouse_color, mouse_text = bad, "kernel mice: NONE (USB mouse not present)"
        rows.append((kbd_color, _clip(kbd_text + stamp)))
        rows.append((mouse_color, _clip(mouse_text)))
        return rows

    def _debug_rows(self) -> list[tuple[tuple[int, int, int], str]]:
        dim = (210, 210, 210)
        warn = (255, 186, 80)
        bad = (255, 110, 110)
        ok = (150, 220, 170)
        stats = self.midi.snapshot()
        fps = self.clock.get_fps() if self.clock is not None else 0.0
        focus = "yes" if self._key_focus else "NO"
        if self._key_last is None:
            key_text = "last key: none"
        else:
            age = time.monotonic() - self._key_last[0]
            key_text = f"last key {age:.1f}s: {self._key_last[1]}"
        xrun_reason = f" {self._audio_status}" if self._audio_status else ""
        rows: list[tuple[tuple[int, int, int], str]] = [
            (dim, f"DEBUG  F1 hides overlay   log {self.debug_log_path or 'off'}"),
            (
                dim,
                _clip(f"out {self.audio_output_name}   in {self.audio_input_name}"),
            ),
            (
                bad if self._key_focus is False else (warn if self.audio_xruns else dim),
                _clip(
                    f"fps {fps:.0f}  xruns {self.audio_xruns}{xrun_reason}  key-focus {focus}  "
                    f"key-events {self._key_events}  {key_text}"
                ),
            ),
        ]
        rows.extend(self._kernel_rows(dim, warn, bad, ok))
        ports = ", ".join(stats.ports) or "none"
        midi_color = warn if stats.errors or not stats.ports else dim
        err = f"  errors: {'; '.join(stats.errors)}" if stats.errors else ""
        rows.append(
            (
                midi_color,
                _clip(
                    f"MIDI rx {stats.received} drop {stats.dropped} sense {stats.active_sense} "
                    f"other {stats.other_count}  ports: {ports}{err}"
                ),
            )
        )
        if stats.last_other:
            rows.append((warn, _clip(f"last non-note MIDI: {stats.last_other}")))
        channels = {0, 1}
        channels.update(channel for channel, _cc in stats.cc_counts)
        channels.update(stats.bend_counts)
        short = {1: "len", 2: "dur", 4: "loop", 5: "rec", 7: "filt"}
        for channel in sorted(channels):
            parts = [f"bend {stats.bend_counts.get(channel, 0)}"]
            for cc, label in short.items():
                parts.append(f"{label} {stats.cc_counts.get((channel, cc), 0)}")
            extras = [
                f"#{cc}={count}"
                for (found, cc), count in sorted(stats.cc_counts.items())
                if found == channel and cc not in short
            ]
            extra = f"  other {' '.join(extras)}" if extras else ""
            rows.append((dim, _clip(f"ch{channel + 1} hits {' '.join(parts)}{extra}")))
        for index, state in enumerate(self.states):
            recording = self.engine.waves[index].recording
            rows.append(
                (
                    dim,
                    f"wave{index + 1} start {state.selection_start_chunk} "
                    f"size {state.selection_size_chunks} grain {state.grain_coeff:.2f} "
                    f"filt {state.filter_midi:.0f} loop {state.looping} rec {recording}",
                )
            )
        if self._trace_lines:
            rows.append((dim, "recent:"))
            for line in self._trace_lines:
                color = warn if ("IGNORED" in line or "changed from" in line or "focus False" in line) else dim
                rows.append((color, _clip(line)))
        return rows

    def _draw_debug(self) -> None:
        pygame = self.pygame
        rows = self._debug_rows()
        line_h = 18
        max_rows = max(4, (self.height // 2) // line_h)
        rows = rows[:max_rows]
        panel_h = 8 + line_h * len(rows)
        panel = pygame.Surface((self.width, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        for index, (color, text) in enumerate(rows):
            panel.blit(self.debug_font.render(text, True, color), (10, 4 + index * line_h))
        self.screen.blit(panel, (0, max(0, self.height - panel_h)))

    def _draw_wave(self, wave_idx: int) -> None:
        pygame = self.pygame
        state = self.states[wave_idx]
        color = WAVE_COLORS[wave_idx]
        tier_h = self.height / NUM_WAVES
        center_y = (
            self.height - (self.height / (2 * NUM_WAVES))
            if wave_idx == 0
            else self.height / (NUM_WAVES * 2)
        )
        max_h = tier_h * SHRINK
        n = self.config.num_chunks
        total_w = n * (CHUNK_GAP + CHUNK_WIDTH)
        scale_x = self.width / total_w
        start = state.selection_start_chunk
        end = min(n - 1, start + state.selection_size_chunks - 1)
        dest_y = int(center_y - tier_h / 2)
        played = set(state.cursors.values())
        recorded = (128, 128, 128)
        selection_gain = 0.5 + 0.5 * state.selection_alpha
        selected = tuple(int(channel * selection_gain) for channel in color)

        scope = state.oscilloscope
        if scope.size > 1:
            peak = float(np.max(np.abs(scope)))
            if peak > 1e-4:
                gain = min(6.0, 0.55 / peak)
                pts = []
                for j, sample in enumerate(scope):
                    x = int(j / (scope.size - 1) * (self.width - 1))
                    level = float(sample)
                    if not math.isfinite(level):
                        level = 0.0
                    y = int(center_y - level * gain * tier_h * 0.42)
                    pts.append((x, y))
                pygame.draw.aalines(self.screen, (255, 255, 255), False, pts)

        for edge in (start, end):
            x = int((1 + edge * (CHUNK_GAP + CHUNK_WIDTH)) * scale_x)
            w = max(1, int(CHUNK_WIDTH * scale_x))
            pygame.draw.rect(self.screen, color, (x, dest_y, w, int(tier_h)))

        for idx, (bottom, top) in enumerate(state.chunks):
            anim = state.chunk_animate[idx]
            x = int((1 + idx * (CHUNK_GAP + CHUNK_WIDTH)) * scale_x)
            w = max(1, int(CHUNK_WIDTH * scale_x))
            top_h = anim * top * max_h
            bot_h = anim * abs(bottom) * max_h
            if idx in played and start <= idx <= end:
                chunk_color = (255, 255, 255)
            elif start <= idx <= end:
                chunk_color = selected
            else:
                chunk_color = recorded
            pygame.draw.rect(
                self.screen,
                chunk_color,
                (x, int(center_y - top_h), w, max(1, int(top_h + bot_h))),
            )
        label = self.font.render(f"grain {state.grain_coeff:.1f}", True, color)
        self.screen.blit(label, (self.width - label.get_width() - 12, int(dest_y + 8)))


def _import_pygame():
    try:
        import pygame
    except ImportError as exc:
        print(
            "pygame is required. On Raspberry Pi OS:\n"
            "  sudo apt install python3-pygame python3-numpy libportaudio2\n"
            "  python3 -m pip install --user sounddevice python-rtmidi",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return pygame


def _clip(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="opencollidoscope")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the input overlay and append opencollidoscope-debug.log",
    )
    parser.add_argument(
        "--debug-log",
        metavar="PATH",
        help="write the debug log to PATH (implies --debug)",
    )
    parser.add_argument(
        "--output",
        metavar="NAME",
        help="output device index or name fragment, for example usb",
    )
    parser.add_argument(
        "--input",
        metavar="NAME",
        help="input device index or name fragment",
    )
    parser.add_argument(
        "--list-audio",
        action="store_true",
        help="print PortAudio devices and exit",
    )
    parser.add_argument("width", nargs="?", type=int)
    parser.add_argument("height", nargs="?", type=int)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_audio:
        return _print_audio_devices()
    if (args.width is None) != (args.height is None):
        parser.error("pass both window_width and window_height, or neither")
    width = 1024 if args.width is None else args.width
    height = 768 if args.height is None else args.height
    debug = bool(args.debug or args.debug_log)
    return CollidoscopeApp(
        width,
        height,
        debug=debug,
        debug_log=args.debug_log,
        input_query=args.input,
        output_query=args.output,
    ).run()


def _print_audio_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError as exc:
        print("sounddevice is required to list audio devices", file=sys.stderr)
        raise SystemExit(1) from exc
    hostapis = sd.query_hostapis()
    devices = []
    for dev in sd.query_devices():
        index = int(dev["index"])
        host = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else ""
        devices.append({**dev, "index": index, "hostapi": host})
    default_in, default_out = sd.default.device
    print(format_audio_devices(devices))
    print(f"PortAudio default input {default_in}, output {default_out}")
    return 0
