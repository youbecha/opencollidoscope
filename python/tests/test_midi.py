# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from opencollidoscope.midi_io import (
    MIDIInput,
    MIDIMessage,
    pitchbend_level,
    pitchbend_value,
    selection_chunk_from_pitchbend,
)
from opencollidoscope.ui import (
    CollidoscopeApp,
    choose_audio_devices,
    find_audio_device,
    find_audio_devices,
    parse_audio_config,
    require_audio_device,
    save_audio_config,
    audio_stream_timing,
    hardware_output_channels,
    load_audio_config,
    parse_kernel_devices,
    parse_kernel_keyboards,
    write_wav,
)


PROC_SAMPLE = """
I: Bus=0003 Vendor=046d Product=c31c Version=0110
N: Name="Logitech USB Keyboard"
P: Phys=usb-0000:01:00.0-1.2/input0
H: Handlers=sysrq kbd event0 leds
B: EV=120013

I: Bus=0003 Vendor=046d Product=c077 Version=0111
N: Name="Logitech USB Optical Mouse"
H: Handlers=mouse0 event1
B: EV=17

N: Name="AT Translated Set 2 keyboard"
H: Handlers=sysrq kbd event2 leds
"""


class PitchbendTests(unittest.TestCase):
    def test_unsigned_14_bit(self) -> None:
        self.assertEqual(pitchbend_value(0x0A, 0x01), 0x8A)
        self.assertEqual(pitchbend_value(0, 0), 0)
        self.assertEqual(pitchbend_value(0, 64), 8192)

    def test_seven_bit_value_uses_127_as_the_top(self) -> None:
        self.assertEqual(pitchbend_level(0, 0), 0)
        self.assertEqual(pitchbend_level(127, 0), 127)
        self.assertEqual(selection_chunk_from_pitchbend(0, 0, 150), 0)
        self.assertEqual(selection_chunk_from_pitchbend(127, 0, 150), 149)

    def test_standard_wheel_uses_the_coarse_byte(self) -> None:
        self.assertEqual(pitchbend_level(0, 64), 64)
        self.assertEqual(pitchbend_level(127, 127), 127)
        self.assertEqual(selection_chunk_from_pitchbend(0, 127, 150), 149)
        self.assertEqual(selection_chunk_from_pitchbend(0, 64, 150), 75)


class MIDIInputTests(unittest.TestCase):
    def test_note_on_velocity_zero_is_note_off(self) -> None:
        midi = MIDIInput()
        midi._ingest([0x90, 60, 0], "keys")
        messages = midi.poll()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].voice, "noteoff")
        self.assertEqual(messages[0].data1, 60)
        self.assertEqual(messages[0].port, "keys")

    def test_filter_cc_keeps_latest_but_counts_each(self) -> None:
        midi = MIDIInput()
        midi._ingest([0xB0, 7, 10], "teensy")
        midi._ingest([0xB0, 7, 80], "teensy")
        messages = midi.poll()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].data2, 80)
        self.assertEqual(midi.snapshot().cc_counts[(0, 7)], 2)

    def test_duration_cc_is_counted_on_its_channel(self) -> None:
        midi = MIDIInput()
        midi._ingest([0xB1, 2, 40], "teensy")
        message = midi.poll()[0]
        self.assertEqual(message.voice, "cc")
        self.assertEqual(message.channel, 1)
        self.assertEqual(message.data1, 2)
        self.assertEqual(midi.snapshot().cc_counts[(1, 2)], 1)

    def test_unknown_status_is_visible_without_reaching_the_engine_queue(self) -> None:
        midi = MIDIInput()
        midi._ingest([0xC0, 3], "teensy")
        self.assertEqual(midi.poll(), [])
        stats = midi.snapshot()
        self.assertEqual(stats.other_count, 1)
        self.assertIn("0xC0", stats.last_other)

    def test_active_sense_is_counted(self) -> None:
        midi = MIDIInput()
        midi._ingest([0xFE], "teensy")
        self.assertEqual(midi.poll(), [])
        self.assertEqual(midi.snapshot().active_sense, 1)

    def test_pitchbend_is_coalesced_per_channel(self) -> None:
        midi = MIDIInput()
        midi._ingest([0xE0, 10, 0], "teensy")
        midi._ingest([0xE0, 20, 0], "teensy")
        messages = midi.poll()
        self.assertEqual(len(messages), 1)
        self.assertEqual(pitchbend_value(messages[0].data1, messages[0].data2), 20)
        self.assertEqual(midi.snapshot().bend_counts[0], 2)


class MidiApplyTests(unittest.TestCase):
    def test_pitchbend_127_reaches_the_last_chunk(self) -> None:
        app = CollidoscopeApp(100, 100)
        detail = app._handle_midi(MIDIMessage("pitchbend", 0, 127, 0))
        self.assertIn("level 127/127", detail)
        self.assertEqual(app.states[0].selection_start_chunk, 149)

    def test_centered_pitch_wheel_maps_from_its_coarse_byte(self) -> None:
        app = CollidoscopeApp(100, 100)
        detail = app._handle_midi(MIDIMessage("pitchbend", 0, 0, 64))
        self.assertIn("level 64/127", detail)
        self.assertEqual(app.states[0].selection_start_chunk, 75)

    def test_duration_cc_updates_grain(self) -> None:
        app = CollidoscopeApp(100, 100)
        detail = app._handle_midi(MIDIMessage("cc", 1, 2, 127))
        self.assertIn("wave2", detail)
        self.assertAlmostEqual(app.states[1].grain_coeff, 8.0)
        self.assertAlmostEqual(app.states[0].grain_coeff, 1.0)

    def test_unknown_cc_does_not_change_selection(self) -> None:
        app = CollidoscopeApp(100, 100)
        detail = app._handle_midi(MIDIMessage("cc", 0, 99, 40))
        self.assertIn("IGNORED CC99", detail)
        self.assertEqual(app.states[0].selection_size_chunks, 8)


class AudioDeviceTests(unittest.TestCase):
    def test_prefers_pipewire_over_hdmi(self) -> None:
        devices = [
            {"index": 0, "name": "vc4-hdmi-0", "hostapi": "ALSA", "max_input_channels": 0, "max_output_channels": 8},
            {"index": 1, "name": "bcm2835 Headphones", "hostapi": "ALSA", "max_input_channels": 0, "max_output_channels": 2},
            {"index": 2, "name": "pulse", "hostapi": "ALSA", "max_input_channels": 32, "max_output_channels": 32},
            {"index": 3, "name": "USB Audio", "hostapi": "ALSA", "max_input_channels": 2, "max_output_channels": 0},
        ]
        chosen_in, chosen_out = choose_audio_devices(devices, 3, 0)
        self.assertEqual(chosen_out, 2)
        self.assertEqual(chosen_in, 2)

    def test_uses_portaudio_default_when_no_desktop_device(self) -> None:
        devices = [
            {"index": 0, "name": "vc4-hdmi-0", "hostapi": "ALSA", "max_input_channels": 0, "max_output_channels": 8},
            {"index": 1, "name": "bcm2835 Headphones", "hostapi": "ALSA", "max_input_channels": 0, "max_output_channels": 2},
        ]
        _chosen_in, chosen_out = choose_audio_devices(devices, None, 1)
        self.assertEqual(chosen_out, 1)

    def test_name_fragment_selects_usb_output(self) -> None:
        devices = [
            {"index": 0, "name": "vc4-hdmi-0", "hostapi": "ALSA", "max_input_channels": 0, "max_output_channels": 8},
            {"index": 1, "name": "USB Audio Device", "hostapi": "ALSA", "max_input_channels": 2, "max_output_channels": 2},
        ]
        self.assertEqual(find_audio_device(devices, "usb", "out"), 1)
        self.assertEqual(find_audio_device(devices, "0", "out"), 0)
        self.assertIsNone(find_audio_device(devices, "headphones", "out"))

    def test_shared_usb_name_is_rejected(self) -> None:
        devices = [
            {"index": 2, "name": "KTMicro KT USB Audio", "hostapi": "ALSA", "max_input_channels": 1, "max_output_channels": 2},
            {"index": 4, "name": "Scarlett 2i4 USB", "hostapi": "ALSA", "max_input_channels": 2, "max_output_channels": 4},
        ]
        self.assertEqual(find_audio_devices(devices, "usb", "in"), [2, 4])
        self.assertEqual(require_audio_device(devices, "Scarlett", "in"), 4)
        with self.assertRaises(SystemExit) as raised:
            require_audio_device(devices, "usb", "in")
        self.assertIn("KTMicro", str(raised.exception))
        self.assertIn("Scarlett", str(raised.exception))

    def test_audio_config_round_trip(self) -> None:
        pinned_in, pinned_out = parse_audio_config(
            "# comment\ninput = Scarlett\noutput =\nnoise = 1\n"
        )
        self.assertEqual((pinned_in, pinned_out), ("Scarlett", None))
        with tempfile.TemporaryDirectory() as folder:
            path = save_audio_config("Scarlett", "hdmi", path=Path(folder) / "audio.conf")
            self.assertEqual(load_audio_config(path), ("Scarlett", "hdmi"))

    def test_hardware_output_uses_a_longer_block(self) -> None:
        self.assertEqual(audio_stream_timing("USB Audio Device", "ALSA"), (1024, 0.1))
        self.assertEqual(audio_stream_timing("pulse", "ALSA"), (256, "high"))

    def test_scarlett_opens_all_four_outputs(self) -> None:
        self.assertEqual(hardware_output_channels(4, "Scarlett 2i4 USB", "ALSA"), 4)
        self.assertEqual(hardware_output_channels(32, "pulse", "ALSA"), 2)


class KernelKeyboardTests(unittest.TestCase):
    def test_lists_only_kbd_handlers(self) -> None:
        names = parse_kernel_keyboards(PROC_SAMPLE)
        self.assertEqual(names, ["Logitech USB Keyboard", "AT Translated Set 2 keyboard"])

    def test_lists_mice_separately(self) -> None:
        keyboards, mice = parse_kernel_devices(PROC_SAMPLE)
        self.assertEqual(mice, ["Logitech USB Optical Mouse"])
        self.assertEqual(keyboards[0], "Logitech USB Keyboard")

    def test_empty_proc_text(self) -> None:
        self.assertEqual(parse_kernel_keyboards(""), [])


class WavWriteTests(unittest.TestCase):
    def test_round_trip_peak_and_channels(self) -> None:
        samples = np.array([0.0, 0.5, -0.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/tone.wav"
            peak = write_wav(path, samples, 48000)
            self.assertAlmostEqual(peak, 0.5)
            with wave.open(path, "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), 48000)
                self.assertEqual(handle.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
