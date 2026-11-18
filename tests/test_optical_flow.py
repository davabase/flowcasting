"""Known-motion checks for boundaries, alignment, reliability, evolution and halos."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from flowcast.optical_flow import (DEFAULT_CONFIG, V2_CONFIG, OpticalFlowConfig, _bands, _combine_motion,
                                   _growth_trend, _motion_history, _pair_confidence,
                                   config_from_result, configuration, extrapolate,
                                   legacy_config, load_config, optical_flow, optical_flow_one)


class OpticalFlowBoundaryTests(unittest.TestCase):
    def test_legacy_interpolation_matches_original(self):
        field = np.random.default_rng(42).integers(0, 256, (16, 16), dtype=np.uint8)
        motion = np.empty((16, 16, 2), np.float32)
        motion[:] = (-0.5, 0.25)
        prediction = extrapolate(field, motion, 3, config=legacy_config())
        x, y = np.meshgrid(np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32))
        for frame in prediction:
            x, y = x - 0.5, y + 0.25
            expected = cv2.remap(field, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            np.testing.assert_array_equal(frame, expected)

    def test_translation_extends_edge_values(self):
        field = np.arange(1, 31, dtype=np.uint8).reshape(5, 6)
        original = field.copy()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
            with self.subTest(dx=dx, dy=dy):
                motion = np.empty((*field.shape, 2), dtype=np.float32)
                motion[:] = (dx, dy)
                prediction = extrapolate(field, motion, steps=8)
                y, x = np.indices(field.shape)
                for lead, frame in enumerate(prediction, start=1):
                    expected = field[np.clip(y + lead * dy, 0, 4),
                                     np.clip(x + lead * dx, 0, 5)]
                    np.testing.assert_array_equal(frame, expected)
        np.testing.assert_array_equal(field, original)

    def test_fractional_motion_keeps_uniform_field(self):
        field = np.full((5, 6), 137, dtype=np.uint8)
        motion = np.empty((*field.shape, 2), dtype=np.float32)
        motion[:] = (-0.5, 0.25)
        prediction = extrapolate(field, motion, steps=12)
        np.testing.assert_array_equal(prediction, np.broadcast_to(field, prediction.shape))


class OpticalFlowMotionTests(unittest.TestCase):
    def test_old_vectors_follow_latest_storm_location(self):
        frames = np.zeros((3, 32, 32), np.uint8)
        newest = np.zeros((32, 32, 2), np.float32)
        newest[..., 0] = -2
        older = np.zeros_like(newest)
        older[..., 0] = np.arange(32)[None, :]
        config = replace(legacy_config(), align_motion=True, pairs=2)
        with patch('flowcast.optical_flow._dis') as dis:
            dis.return_value.calc.side_effect = [newest, older]
            flows, _, _, _ = _motion_history(frames, config)
        self.assertEqual(flows[1, 16, 12, 0], 10)
        self.assertEqual(flows[0, 16, 12, 0], -2)

    def test_unreliable_pair_does_not_dominate_average(self):
        flows = np.zeros((2, 32, 32, 2), np.float32)
        flows[0, ..., 0], flows[1, ..., 0] = -2, 20
        confidence = np.ones((2, 32, 32), np.float32)
        confidence[1] = 0.001
        motion = _combine_motion(flows, confidence, replace(DEFAULT_CONFIG, smooth_motion=False))
        self.assertLess(np.abs(motion[..., 0] + 2).max(), 0.03)

    def test_low_confidence_hole_uses_neighboring_motion(self):
        flows = np.zeros((1, 32, 32, 2), np.float32)
        flows[..., 0] = -2
        flows[0, 16, 16] = (40, 40)
        confidence = np.ones((1, 32, 32), np.float32)
        confidence[0, 16, 16] = 0
        motion = _combine_motion(flows, confidence, DEFAULT_CONFIG)
        np.testing.assert_allclose(motion[16, 16], (-2, 0), atol=1e-5)
        np.testing.assert_allclose(motion[8, 8], (-2, 0), atol=1e-5)

    def test_inconsistent_motion_gets_lower_confidence(self):
        field = np.random.default_rng(42).integers(0, 256, (32, 32), dtype=np.uint8)
        still = np.zeros((32, 32, 2), np.float32)
        good = _pair_confidence(field, field, still, still, DEFAULT_CONFIG)
        bad = _pair_confidence(field, field, still, np.full_like(still, 10), DEFAULT_CONFIG)
        self.assertLess(bad.mean(), good.mean() / 10)
        wrong_image = _pair_confidence(field, np.zeros_like(field), still, still, DEFAULT_CONFIG)
        self.assertLess(wrong_image.mean(), good.mean() / 2)

    def test_known_translation_with_real_dis(self):
        field = np.zeros((96, 96), np.uint8)
        field[16:80, 16:72] = np.random.default_rng(42).integers(30, 200, (64, 56), dtype=np.uint8)
        frames = np.stack([np.roll(field, t, axis=1) for t in range(4)])
        original = frames.copy()
        prediction, motion = optical_flow_one(frames, steps=3)
        np.testing.assert_array_equal(frames, original)
        self.assertAlmostEqual(float(np.median(motion[24:72, 24:64, 0])), -1, delta=0.25)
        self.assertAlmostEqual(float(np.median(motion[24:72, 24:64, 1])), 0, delta=0.25)
        for lead, forecast in enumerate(prediction, 1):
            expected = np.roll(frames[-1], lead, axis=1)
            self.assertLess(np.abs(forecast.astype(float) - expected).mean(), 10)


class OpticalFlowEvolutionTests(unittest.TestCase):
    def test_aligned_translation_is_not_growth(self):
        field = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        frames = np.stack([np.roll(field, t, axis=1) for t in range(4)])
        flow = np.zeros((32, 32, 2), np.float32)
        flow[..., 0] = -1
        config = replace(legacy_config(), align_motion=True)
        with patch('flowcast.optical_flow._dis') as dis:
            dis.return_value.calc.return_value = flow
            _, _, residuals, valid = _motion_history(frames, config)
        trend = _growth_trend(residuals, valid, config)
        np.testing.assert_allclose(trend[:, 12:24], 0, atol=1e-4)

    def test_growth_and_decay_are_estimated_and_bounded(self):
        for delta in (-2, 2):
            with self.subTest(delta=delta):
                frames = np.stack([np.full((32, 32), 100 + delta * t, np.uint8) for t in range(4)])
                config = replace(DEFAULT_CONFIG, multiscale=False)
                with patch('flowcast.optical_flow._dis') as dis:
                    dis.return_value.calc.return_value = np.zeros((32, 32, 2), np.float32)
                    prediction, _ = optical_flow_one(frames, 100, config=config)
                self.assertEqual(int(prediction[0, 16, 16]), round(100 + delta * 3 + delta * 3 * (1 - np.exp(-1 / 3))))
                self.assertEqual(int(prediction[-1, 16, 16]), 100 + delta * 6)

    def test_missing_history_and_isolated_error_do_not_create_trend(self):
        residuals = np.full((3, 32, 32), 2, np.float32)
        residuals[0] = 80
        valid = np.ones_like(residuals, dtype=bool)
        np.testing.assert_allclose(_growth_trend(residuals, valid, DEFAULT_CONFIG), 2, atol=1e-5)
        np.testing.assert_array_equal(_growth_trend(residuals, ~valid, DEFAULT_CONFIG), 0)

    def test_multiscale_reconstruction_and_detail_damping(self):
        y, x = np.indices((64, 64))
        field = (100 + 40 * ((x + y) % 2 * 2 - 1)).astype(np.float32)
        bands, coarse = _bands(field, DEFAULT_CONFIG.scale_sigmas)
        np.testing.assert_allclose(sum(bands, coarse), field, atol=1e-5)
        prediction = extrapolate(field, np.zeros((64, 64, 2), np.float32), 36,
                                 config=replace(DEFAULT_CONFIG, growth_decay=False))
        first, last = prediction[0, 12:-12, 12:-12], prediction[-1, 12:-12, 12:-12]
        self.assertLess(last.std(), first.std() / 2)
        self.assertAlmostEqual(float(last.mean()), 100, delta=0.1)

    def test_empty_field_and_batched_contract(self):
        inputs = np.zeros((2, 4, 32, 32), np.uint8)
        prediction, motion = optical_flow(inputs, steps=3, workers=2)
        self.assertEqual(prediction.shape, (2, 3, 32, 32))
        self.assertEqual(motion.shape, (2, 32, 32, 2))
        self.assertEqual(prediction.dtype, np.uint8)
        self.assertEqual(motion.dtype, np.float32)
        self.assertFalse(prediction.any())
        self.assertFalse(motion.any())

    def test_large_trend_is_clipped_without_uint8_wraparound(self):
        for intensity, trend, expected in [(250, 100, 255), (5, -100, 0)]:
            prediction = extrapolate(np.full((8, 8), intensity, np.uint8), np.zeros((8, 8, 2), np.float32),
                                     12, trend=np.full((8, 8), trend, np.float32),
                                     config=replace(DEFAULT_CONFIG, multiscale=False))
            np.testing.assert_array_equal(prediction[-1], expected)


class OpticalFlowConfigurationTests(unittest.TestCase):
    def test_v2_scores_keep_original_motion_settings(self):
        recorded = configuration(V2_CONFIG)
        recorded['version'] = 2
        recorded['parameters'].pop('motion_gaps')
        recorded['parameters'].pop('integration')
        self.assertEqual(config_from_result({'configuration': recorded}), V2_CONFIG)

    def test_long_gap_normalizes_motion_and_growth(self):
        frames = np.stack([np.full((32, 32), 100 + 2*t, np.uint8) for t in range(7)])
        flow = np.full((32, 32, 2), (-6, 0), np.float32)
        config = replace(legacy_config(boundary='edge'), motion_gaps=(3,), growth_decay=True)
        with patch('flowcast.optical_flow._dis') as dis:
            dis.return_value.calc.return_value = flow
            prediction, motion = optical_flow_one(frames, 1, config=config)
        np.testing.assert_allclose(motion[8:24, 8:24, 0], -2)
        np.testing.assert_array_equal(prediction[:, 8:24, 8:24], 114)

    def test_selected_preset_matches_defaults(self):
        self.assertEqual(load_config(Path(__file__).resolve().parents[1] /
                                     'configs/optical_flow_v3.json'), DEFAULT_CONFIG)

    def test_real_observation_halo_can_feed_crop(self):
        frames = np.zeros((4, 32, 32), np.uint8)
        frames[:, 8:24, 4:8] = 150  # echo initially outside the requested crop
        flow = np.zeros((32, 32, 2), np.float32)
        flow[..., 0] = -2
        with patch('flowcast.optical_flow._dis') as dis:
            dis.return_value.calc.return_value = flow
            prediction, motion = optical_flow_one(frames, 3, config=legacy_config(boundary='edge'),
                                                  output_region=(8, 8, 16, 16))
        self.assertEqual(prediction.shape, (3, 16, 16))
        self.assertEqual(motion.shape, (16, 16, 2))
        self.assertTrue((prediction[0, :, :2] == 150).all())
        self.assertTrue((prediction[0, :, 2:] == 0).all())

    def test_configs_round_trip_and_unrecorded_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(configuration()), encoding='utf-8')
            self.assertEqual(load_config(path), DEFAULT_CONFIG)
        self.assertEqual(config_from_result({'configuration': configuration()}), DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            config_from_result({})
        with self.assertRaises(ValueError):
            config_from_result({'configuration': {'version': -1, 'parameters': {}}})

    def test_invalid_settings_and_regions_are_rejected(self):
        for values in ({'pairs': 0}, {'recency_decay': 0}, {'scale_lifetimes': (2,)},
                       {'scale_sigmas': (8, 2)}, {'boundary': 'wrap'}, {'growth_cap': -1},
                       {'motion_gaps': (0, 2)}, {'motion_gaps': (2, 2)}, {'integration': 'bad'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                OpticalFlowConfig(**values)
        with self.assertRaises(ValueError):
            optical_flow_one(np.zeros((4, 32, 32), np.uint8), output_region=(0, 0, 40, 40))


if __name__ == '__main__':
    unittest.main()
