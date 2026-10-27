# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
import unittest

import numpy as np

from opencollidoscope.config import Config
from opencollidoscope.dsp import (
    DEFAULT_ATTENUATION,
    BiquadLowpass,
    EnvASR,
    EnvState,
    PGranular,
    capture_gain,
    filter_cutoff_from_midi,
    midi_note_ratio,
    normalize_recording,
    wave_inputs,
)
from opencollidoscope.engine import AudioEngine


class MidiRatioTests(unittest.TestCase):
    def test_middle_c_is_unity(self) -> None:
        self.assertAlmostEqual(midi_note_ratio(60), 1.0)

    def test_octave_up(self) -> None:
        self.assertAlmostEqual(midi_note_ratio(72), 2.0)

    def test_octave_down(self) -> None:
        self.assertAlmostEqual(midi_note_ratio(48), 0.5)

    def test_semitone_up(self) -> None:
        self.assertAlmostEqual(midi_note_ratio(61), 2 ** (1 / 12), places=9)


class EnvelopeTests(unittest.TestCase):
    def test_idle_then_attack_reaches_sustain(self) -> None:
        env = EnvASR(1.0, 0.01, 0.05, 48000)
        self.assertIs(env.state, EnvState.IDLE)
        env.state = EnvState.ATTACK
        peak = 0.0
        for _ in range(48000):
            peak = max(peak, env.tick())
            if env.state is EnvState.SUSTAIN:
                break
        self.assertIs(env.state, EnvState.SUSTAIN)
        self.assertAlmostEqual(peak, 1.0, places=5)


class FilterMapTests(unittest.TestCase):
    def test_midi_127_is_near_max(self) -> None:
        cfg = Config()
        hi = filter_cutoff_from_midi(127, cfg.min_filter_cutoff_hz, cfg.max_filter_cutoff_hz)
        self.assertGreater(hi, 10_000)


class CaptureGainTests(unittest.TestCase):
    def test_quiet_mic_is_boosted_and_hot_mic_is_not(self) -> None:
        self.assertAlmostEqual(capture_gain(0.01), 45.0)
        self.assertEqual(capture_gain(0.9), 1.0)
        self.assertEqual(capture_gain(0.0), 1.0)
        self.assertLessEqual(capture_gain(1e-6), 80.0)

    def test_normalize_lifts_a_quiet_buffer(self) -> None:
        buffer = np.full(100, 0.01, dtype=np.float32)
        peak, gain = normalize_recording(buffer)
        self.assertAlmostEqual(peak, 0.01)
        self.assertAlmostEqual(gain, 80.0)
        self.assertAlmostEqual(float(buffer[0]), 0.8, places=5)

    def test_two_waves_keep_their_own_inputs(self) -> None:
        left = np.array([0.5, -0.25], dtype=np.float32)
        right = np.array([0.0001, -0.0001], dtype=np.float32)
        signals = wave_inputs(np.column_stack([left, right]), 2)
        self.assertTrue(np.allclose(signals[0], left))
        self.assertTrue(np.allclose(signals[1], right))

    def test_single_wave_uses_only_the_first_input(self) -> None:
        left = np.zeros(4, dtype=np.float32)
        right = np.array([0.2, -0.4, 0.1, -0.2], dtype=np.float32)
        signals = wave_inputs(np.column_stack([left, right]), 1)
        self.assertEqual(len(signals), 1)
        self.assertTrue(np.allclose(signals[0], left))

    def test_mono_input_feeds_every_wave(self) -> None:
        mono = np.array([0.2, -0.2], dtype=np.float32)
        signals = wave_inputs(mono.reshape(-1, 1), 2)
        self.assertTrue(np.allclose(signals[0], mono))
        self.assertTrue(np.allclose(signals[1], mono))


class BiquadTests(unittest.TestCase):
    def test_nonfinite_state_resets_instead_of_spreading(self) -> None:
        filt = BiquadLowpass(48000, 1000.0)
        filt._z1 = float("inf")
        y = filt.process(np.ones(32, dtype=np.float32))
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(math.isfinite(filt._z1))
        self.assertTrue(math.isfinite(filt._z2))

    def test_cutoff_sweep_stays_finite(self) -> None:
        filt = BiquadLowpass(48000, 200.0)
        block = np.sign(np.sin(np.linspace(0, 40, 256))).astype(np.float32)
        for midi in range(128):
            filt.set_cutoff(filter_cutoff_from_midi(midi, 200.0, 22050.0))
            y = filt.process(block)
            self.assertTrue(np.isfinite(y).all())


class GranularTests(unittest.TestCase):
    def test_note_on_produces_audio(self) -> None:
        buf = np.ones(48000, dtype=np.float32)
        events: list[tuple[str, int]] = []

        def cb(kind: str, grain_id: int) -> None:
            events.append((kind, grain_id))

        g = PGranular(buf, 48000, lambda: 0, cb, 0)
        g.set_selection_start(0)
        g.set_selection_size(2000)
        g.note_on(1.0)
        out = np.zeros(2048, dtype=np.float32)
        tmp = np.zeros(2048, dtype=np.float32)
        g.process(out, tmp, 2048)
        self.assertFalse(math.isclose(float(np.max(np.abs(out))), 0.0))
        self.assertLessEqual(float(np.max(np.abs(out))), DEFAULT_ATTENUATION * 1.1)
        self.assertTrue(any(kind == "t" for kind, _ in events))


class EngineTests(unittest.TestCase):
    def test_stereo_process_shape(self) -> None:
        cfg = Config(sample_rate=48000, block_size=256)
        engine = AudioEngine(cfg)
        engine.waves[0].start_record()
        inp = np.zeros((256, 2), dtype=np.float32)
        inp[:, 0] = 0.2
        out = engine.process_stereo(inp)
        self.assertEqual(out.shape, (256, 2))
        engine.waves[0].note_on(60)
        out2 = engine.process_stereo(inp)
        self.assertEqual(out2.shape, (256, 2))


if __name__ == "__main__":
    unittest.main()
