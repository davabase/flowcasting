"""Check the adapter's motion convention and boundary handling on known translations."""
import unittest

import numpy as np

from flowcast.pysteps import pysteps


class PystepsBaselineTests(unittest.TestCase):
    def test_empty_field(self):
        inputs = np.zeros((1, 13, 96, 96), dtype=np.uint8)
        prediction, velocity = pysteps(inputs, steps=3, workers=1)
        self.assertEqual(prediction.shape, (1, 3, 96, 96))
        self.assertEqual(velocity.shape, (1, 96, 96, 2))
        self.assertEqual(prediction.dtype, np.uint8)
        self.assertFalse(prediction.any())
        self.assertFalse(velocity.any())

    def test_eastward_translation(self):
        # Multiple textured echoes give LK identifiable corners and a known +x drift.
        rng = np.random.default_rng(42)
        field = np.zeros((128, 128), np.uint8)
        field[24:104, 24:88] = rng.integers(30, 220, (80, 64), dtype=np.uint8)
        frames = np.stack([np.roll(field, t, axis=1) for t in range(4)])
        original = frames.copy()
        prediction, velocity = pysteps(frames[None], steps=3, workers=1)
        np.testing.assert_array_equal(frames, original)
        self.assertAlmostEqual(float(np.median(velocity[0, 32:96, 32:88, 0])), 1, delta=0.25)
        self.assertAlmostEqual(float(np.median(velocity[0, 32:96, 32:88, 1])), 0, delta=0.25)
        for lead in range(3):
            expected = np.roll(frames[-1], lead + 1, axis=1)
            self.assertLess(np.abs(prediction[0, lead].astype(float) - expected).mean(), 8)
            self.assertFalse(prediction[0, lead, :, :lead + 1].any())


if __name__ == '__main__':
    unittest.main()
