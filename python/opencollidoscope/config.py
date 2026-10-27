# Copyright (C) 2016 Queen Mary University of London
# Author: Fiore Martin
# Python port for current CPython / Raspberry Pi OS.
#
# This file is part of Collidoscope.
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field

NUM_WAVES = 2

WAVE_COLORS = (
    (243, 6, 62),
    (255, 204, 0),
)


@dataclass
class Config:
    """Runtime values from the original C++ Config defaults."""

    num_chunks: int = 150
    wave_len: float = 2.0
    sample_rate: int = 48000
    block_size: int = 256
    max_grain_duration_coeff: float = 8.0
    max_filter_cutoff_hz: float = 22050.0
    min_filter_cutoff_hz: float = 200.0
    max_keyboard_voices: int = 6
    max_selection_chunks: int = 37
    oscilloscope_divider: int = 4
    cursor_trigger_buf_size: int = 512
    midi_channels: tuple[int, int] = field(default_factory=lambda: (0, 1))

    def samples_per_chunk(self) -> int:
        return max(1, round((self.wave_len * self.sample_rate) / self.num_chunks))

    def buffer_frames(self) -> int:
        return int(self.wave_len * self.sample_rate)

    def wave_for_midi_channel(self, channel_idx: int) -> int:
        if 0 <= channel_idx < NUM_WAVES:
            return channel_idx
        return 0
