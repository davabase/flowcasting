"""CUDA integration checks; skip when the native library or a GPU is unavailable."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from flowcast.optical_flow import DEFAULT_CONFIG, extrapolate, legacy_config, optical_flow_one
from flowcast.optical_flow_cuda import CudaOpticalFlow


class CudaOpticalFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        try:
            cls.gpu = CudaOpticalFlow()
            cls.hybrid = CudaOpticalFlow(motion_backend='opencv')
        except (OSError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc))

    @classmethod
    def tearDownClass(cls):
        cls.gpu.close()
        cls.hybrid.close()

    def test_constant_translation_and_boundaries(self):
        field = np.arange(32*32, dtype=np.float32).reshape(32,32) % 220
        for boundary in ('edge', 'zero'):
            for vector in ((1,0), (-1,1), (-.5,.25)):
                with self.subTest(boundary=boundary, vector=vector):
                    motion = np.full((32,32,2), vector, np.float32)
                    config = legacy_config(boundary=boundary)
                    expected = extrapolate(field, motion, 12, config=config)
                    actual = self.gpu.extrapolate(field, motion, 12, config=config)
                    np.testing.assert_array_equal(actual, expected)

    def test_evolution_and_midpoint_match_cpu(self):
        rng = np.random.default_rng(14)
        field = rng.integers(0,256,(48,48),dtype=np.uint8)
        motion = np.full((48,48,2),(-.5,.25), np.float32)
        trend = rng.uniform(-3,3,field.shape).astype(np.float32)
        for integration in ('euler','midpoint'):
            config = replace(DEFAULT_CONFIG,integration=integration)
            expected = extrapolate(field,motion,12,trend=trend,config=config)
            actual = self.gpu.extrapolate(field,motion,12,trend=trend,config=config)
            delta = np.abs(actual.astype(float)-expected)
            self.assertLessEqual(delta.max(),1)
            self.assertLess(delta.mean(),.01)

    def test_midpoint_with_rotating_motion_matches_reference(self):
        y,x = np.indices((64,64),dtype=np.float32)
        field = (80+70*np.exp(-((x-40)**2+(y-32)**2)/80)).astype(np.float32)
        motion = np.stack((-(y-32)*.04,(x-32)*.04),axis=-1)
        config = replace(DEFAULT_CONFIG,integration='midpoint',growth_decay=False)
        expected = extrapolate(field,motion,12,config=config)
        actual = self.gpu.extrapolate(field,motion,12,config=config)
        self.assertLess(np.abs(actual.astype(float)-expected).mean(),.01)
        euler = extrapolate(field,motion,12,config=replace(config,integration='euler'))
        self.assertGreater(np.abs(euler.astype(float)-expected).mean(),.01)

    def test_uniform_growth_and_minimal_history(self):
        frames = np.stack([np.full((32,32),100+2*t,np.uint8) for t in range(2)])
        expected,_ = optical_flow_one(frames,12)
        actual,motion = self.gpu.predict(frames[None],12)
        np.testing.assert_array_equal(actual[0],expected)
        self.assertFalse(motion.any())

    def test_full_pipeline_translation_empty_and_reuse(self):
        field = np.random.default_rng(42).integers(20,200,(96,96),dtype=np.uint8)
        frames = np.stack([np.roll(field,t,axis=1) for t in range(13)])
        for backend in (self.gpu,self.hybrid):
            empty, motion = backend.predict(np.zeros((2,4,32,32),np.uint8),3)
            self.assertFalse(empty.any())
            self.assertFalse(motion.any())
            actual, motion = backend.predict(frames[None],3)
            self.assertEqual(actual.shape,(1,3,96,96))
            self.assertEqual(actual.dtype,np.uint8)
            self.assertEqual(motion.dtype,np.float32)
            np.testing.assert_allclose(np.median(motion[0,24:72,24:72],axis=(0,1)),(-1,0),atol=.3)
            for lead in range(1,4):
                self.assertLess(np.abs(actual[0,lead-1].astype(float)-np.roll(frames[-1],lead,axis=1)).mean(),12)
            again, again_motion = backend.predict(frames[None],3)
            np.testing.assert_array_equal(actual,again)
            np.testing.assert_array_equal(motion,again_motion)

    def test_opencv_motion_isolates_postprocessing_port(self):
        field = np.random.default_rng(17).integers(20,200,(96,96),dtype=np.uint8)
        field = cv2.GaussianBlur(field,(0,0),2)
        frames = np.stack([np.roll(field,t,axis=1) for t in range(13)])
        expected, motion = optical_flow_one(frames,12)
        actual, gpu_motion = self.hybrid.predict(frames[None],12)
        self.assertLess(np.abs(gpu_motion[0]-motion).mean(),.002)
        self.assertLess(np.abs(actual[0].astype(float)-expected).mean(),.15)

    def test_crop_and_invalid_inputs(self):
        inputs = np.zeros((1,4,32,32),np.uint8)
        prediction,motion = self.gpu.predict(inputs,3,output_region=(4,4,16,16))
        self.assertEqual(prediction.shape,(1,3,16,16))
        self.assertEqual(motion.shape,(1,16,16,2))
        for value in (inputs.astype(float),inputs[:, :1],inputs[:0]):
            with self.assertRaises(ValueError): self.gpu.predict(value)
        with self.assertRaises(ValueError): self.gpu.predict(inputs,0)
        with self.assertRaises(ValueError): self.gpu.predict(inputs,config=replace(DEFAULT_CONFIG,pairs=9))


if __name__ == '__main__':
    unittest.main()
