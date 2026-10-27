# Copyright (C) 2016 Queen Mary University of London
# Author: Fiore Martin
# Python port for current CPython / Raspberry Pi OS.
#
# This file is part of Collidoscope.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("opencollidoscope.midi")

# Teensy firmware: cc_length, cc_duration, cc_play, cc_record, cc_filter.
CC_NAMES = {
    1: "length",
    2: "duration",
    4: "loop",
    5: "record",
    7: "filter",
}


@dataclass(slots=True)
class MIDIMessage:
    voice: str
    channel: int
    data1: int
    data2: int
    status: int = 0
    port: str = ""
    raw: tuple[int, ...] = ()


@dataclass
class MIDIStats:
    available: bool
    ports: tuple[str, ...]
    errors: tuple[str, ...]
    received: int
    dropped: int
    active_sense: int
    other_count: int
    last_other: str
    voice_counts: dict[str, int]
    cc_counts: dict[tuple[int, int], int]
    bend_counts: dict[int, int]


def pitchbend_value(data1: int, data2: int) -> int:
    """14-bit pitch bend from the MIDI data bytes (0–16383)."""
    return (data1 & 0x7F) | ((data2 & 0x7F) << 7)


def pitchbend_level(data1: int, data2: int) -> int:
    """Selection position on the MIDI 0–127 scale.

    A controller that sends 0–127 puts that value in the low byte. A standard
    pitch wheel sends a 14-bit value whose coarse byte is already 0–127.
    """
    raw = pitchbend_value(data1, data2)
    if raw <= 127:
        return raw
    return data2 & 0x7F


def selection_chunk_from_pitchbend(data1: int, data2: int, num_chunks: int) -> int:
    """Map pitch bend 0–127 across the wave. 127 is the last chunk."""
    last = max(0, num_chunks - 1)
    level = pitchbend_level(data1, data2)
    if last == 0 or level <= 0:
        return 0
    return (level * last + 63) // 127


class MIDIInput:
    """RtMidi-backed input. Missing python-rtmidi is a soft failure, like the C++ app."""

    def __init__(self) -> None:
        self._messages: deque[MIDIMessage] = deque(maxlen=1024)
        self._pitch: dict[int, MIDIMessage] = {}
        self._filter: dict[int, MIDIMessage] = {}
        self._lock = threading.Lock()
        self._inputs: list[Any] = []
        self.available = False
        self.ports: list[str] = []
        self.errors: list[str] = []
        self.received = 0
        self.dropped = 0
        self.active_sense = 0
        self.other_count = 0
        self.last_other = ""
        self.voice_counts: dict[str, int] = {}
        self.cc_counts: dict[tuple[int, int], int] = {}
        self.bend_counts: dict[int, int] = {}

    def setup(self) -> None:
        try:
            import rtmidi
        except ImportError:
            message = "python-rtmidi not installed; hardware MIDI disabled"
            self.errors.append(message)
            log.warning(message)
            return
        try:
            probe = rtmidi.MidiIn(name="OpenCollidoscopeProbe")
            names = [probe.get_port_name(i) for i in range(probe.get_port_count())]
            del probe
        except Exception as exc:
            message = f"MIDI port scan failed: {exc}"
            self.errors.append(message)
            log.warning(message)
            return
        names = [name for name in names if "OpenCollidoscope" not in name]
        if not names:
            message = "No MIDI input ports"
            self.errors.append(message)
            log.warning(message)
            return
        # Open by name. Creating a MidiIn client adds an ALSA port, so indexes
        # taken from an earlier client can point at the wrong device.
        for name in names:
            try:
                port = rtmidi.MidiIn(name="OpenCollidoscope")
                index = _port_index(port, name)
                if index is None:
                    self.errors.append(f"missing port {name}")
                    log.warning("MIDI port disappeared: %s", name)
                    continue
                port.open_port(index)
                port.ignore_types(True, True, False)
                port.set_callback(self._make_callback(name))
                self._inputs.append(port)
                self.ports.append(name)
                log.info("MIDI in: %s", name)
            except Exception as exc:
                message = f"{name}: {exc}"
                self.errors.append(message)
                log.warning("MIDI open failed for %s", message)
        self.available = bool(self.ports)

    def _make_callback(self, port_name: str):
        def callback(event: tuple[list[int], float], _data: object = None) -> None:
            try:
                message, _delta = event
                self._ingest(list(message), port_name)
            except Exception:
                log.exception("MIDI callback failed on %s", port_name)

        return callback

    def _ingest(self, message: list[int], port_name: str) -> None:
        parsed = self._parse(message)
        if parsed is None:
            return
        parsed.port = port_name
        with self._lock:
            if parsed.voice == "activesense":
                self.active_sense += 1
                return
            if parsed.voice in ("system", "other"):
                self.other_count += 1
                self.last_other = (
                    f"{port_name} status 0x{parsed.status:02X} bytes {list(parsed.raw)}"
                )
                return
            self.received += 1
            self.voice_counts[parsed.voice] = self.voice_counts.get(parsed.voice, 0) + 1
            if parsed.voice == "cc":
                key = (parsed.channel, parsed.data1)
                self.cc_counts[key] = self.cc_counts.get(key, 0) + 1
            elif parsed.voice == "pitchbend":
                self.bend_counts[parsed.channel] = self.bend_counts.get(parsed.channel, 0) + 1
            if parsed.voice == "pitchbend":
                self._pitch[parsed.channel] = parsed
                return
            if parsed.voice == "cc" and parsed.data1 == 7:
                self._filter[parsed.channel] = parsed
                return
            if len(self._messages) == self._messages.maxlen:
                self.dropped += 1
            self._messages.append(parsed)

    def _parse(self, message: list[int]) -> MIDIMessage | None:
        if not message:
            return None
        status = message[0]
        raw = tuple(int(byte) & 0xFF for byte in message)
        if status >= 0xF8:
            voice = "activesense" if status == 0xFE else "system"
            return MIDIMessage(voice, 0, 0, 0, status=status, raw=raw)
        kind = status & 0xF0
        channel = status & 0x0F
        d1 = message[1] if len(message) > 1 else 0
        d2 = message[2] if len(message) > 2 else 0
        if kind == 0x90:
            voice = "noteoff" if d2 == 0 else "noteon"
            return MIDIMessage(voice, channel, d1, d2, status=status, raw=raw)
        if kind == 0x80:
            return MIDIMessage("noteoff", channel, d1, d2, status=status, raw=raw)
        if kind == 0xE0:
            return MIDIMessage("pitchbend", channel, d1, d2, status=status, raw=raw)
        if kind == 0xB0:
            return MIDIMessage("cc", channel, d1, d2, status=status, raw=raw)
        return MIDIMessage("other", channel, d1, d2, status=status, raw=raw)

    def poll(self) -> list[MIDIMessage]:
        with self._lock:
            out = list(self._messages)
            self._messages.clear()
            out.extend(self._pitch.values())
            self._pitch.clear()
            out.extend(self._filter.values())
            self._filter.clear()
            return out

    def snapshot(self) -> MIDIStats:
        with self._lock:
            return MIDIStats(
                available=self.available,
                ports=tuple(self.ports),
                errors=tuple(self.errors),
                received=self.received,
                dropped=self.dropped,
                active_sense=self.active_sense,
                other_count=self.other_count,
                last_other=self.last_other,
                voice_counts=dict(self.voice_counts),
                cc_counts=dict(self.cc_counts),
                bend_counts=dict(self.bend_counts),
            )

    def close(self) -> None:
        for port in self._inputs:
            try:
                port.cancel_callback()
            except Exception:
                log.debug("MIDI cancel_callback failed", exc_info=True)
            try:
                port.close_port()
            except Exception:
                log.debug("MIDI close failed", exc_info=True)
        self._inputs = []
        self.available = False


def _port_index(port: Any, name: str) -> int | None:
    for index in range(port.get_port_count()):
        if port.get_port_name(index) == name:
            return index
    return None
