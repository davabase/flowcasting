"""Matched accuracy and synchronized end-to-end timing for Python and native CUDA.

Does not overwrite the main results/metrics.json. Default quality run uses all official
test windows; repeat timings use a fixed evenly spaced subset. No parameter selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import h5py
import numpy as np
import torch

from flowcast.data import IN_FRAMES, OUT_FRAMES, TEST_FILE, iter_test_blocks, read_test_samples
from flowcast.metrics import Scores, summary_row
from flowcast.optical_flow import DEFAULT_CONFIG, configuration, optical_flow
from flowcast.optical_flow_cuda import CudaOpticalFlow


def statistics(seconds, batch_sizes):
    times = np.asarray(seconds, dtype=float)
    per_window = times / np.asarray(batch_sizes)
    return {'calls': len(times), 'windows': sum(batch_sizes), 'total_seconds': float(times.sum()),
            'mean_ms_per_window': float(times.sum() / sum(batch_sizes) * 1000),
            'median_ms_per_window': float(np.median(per_window) * 1000),
            'p95_ms_per_window': float(np.quantile(per_window, .95) * 1000),
            'windows_per_second': float(sum(batch_sizes) / times.sum()),
            'seconds': times.tolist(), 'batch_sizes': batch_sizes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-windows', type=int, default=1872)
    parser.add_argument('--timing-windows', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    if min(args.quality_windows, args.timing_windows, args.repeats, args.batch, args.workers) < 1:
        parser.error('All counts must be positive')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(TEST_FILE, 'r') as source:
        total = source['IN_vil'].shape[0]
    quality_ids = np.linspace(0, total-1, min(total, args.quality_windows), dtype=int).tolist()
    timing_ids = np.linspace(0, total-1, min(total, args.timing_windows), dtype=int).tolist()
    methods = {}
    creation = {}
    for name, backend in [('CUDA', 'cuda'), ('OpenCV DIS + CUDA', 'opencv')]:
        started = time.perf_counter()
        methods[name] = CudaOpticalFlow(motion_backend=backend)
        creation[name] = (time.perf_counter()-started)*1000
    names = ['Python / OpenCV', 'OpenCV DIS + CUDA', 'CUDA']

    def predict(name, inputs):
        if name == 'Python / OpenCV':
            return optical_flow(inputs, OUT_FRAMES, args.workers)
        return methods[name].predict(inputs, OUT_FRAMES)

    # Warm DLL loading, workspace allocation, CUDA module initialization and CPU DIS.
    warm = read_test_samples([quality_ids[0]])[:, :IN_FRAMES]
    first_call = {}
    for name in names:
        start = time.perf_counter()
        predict(name, warm)
        first_call[name] = (time.perf_counter()-start)*1000

    records = {name: {'configuration': configuration() if name == names[0] else methods[name].configuration(),
                      'first_call_ms': first_call[name], 'context_creation_ms': creation.get(name, 0)}
               for name in names}
    # Fresh paired scores, without reusing historical Python metrics.
    scores = {name: Scores(OUT_FRAMES) for name in names}
    differences = {name: {'absolute_sum': 0., 'pixels': 0, 'max': 0., 'motion_absolute_sum': 0.,
                           'motion_values': 0, 'equal_pixels': 0} for name in names[1:]}
    digest = hashlib.sha256()
    def quality_batches():
        if len(quality_ids) == total:
            for offset, block in iter_test_blocks():
                for local in range(0, len(block), args.batch):
                    part = block[local:local+args.batch]
                    yield list(range(offset+local, offset+local+len(part))), part
        else:
            for start in range(0, len(quality_ids), args.batch):
                ids = quality_ids[start:start+args.batch]
                yield ids, read_test_samples(ids)

    completed = 0
    for ids, block in quality_batches():
        digest.update(np.asarray(ids, dtype='<i8').tobytes())
        digest.update(block.tobytes())
        inputs, target = block[:, :IN_FRAMES], torch.from_numpy(block[:, IN_FRAMES:]).to('cuda')
        reference = None
        for name in names:
            prediction, motion = predict(name, inputs)
            scores[name].update(torch.from_numpy(prediction).to('cuda'), target)
            if reference is None:
                reference = prediction, motion
            else:
                diff = np.abs(prediction.astype(np.int16)-reference[0].astype(np.int16))
                d = differences[name]
                d['absolute_sum'] += float(diff.sum(dtype=np.float64))
                d['pixels'] += diff.size
                d['max'] = max(d['max'], float(diff.max()))
                d['equal_pixels'] += int((diff == 0).sum())
                d['motion_absolute_sum'] += float(np.abs(motion-reference[1]).sum(dtype=np.float64))
                d['motion_values'] += motion.size
        completed += len(ids)
        if completed % 117 == 0 or completed == len(quality_ids):
            print(f'Quality: {completed}/{len(quality_ids)} windows', flush=True)
    for name in names:
        result = scores[name].result()
        result['windows'] = len(quality_ids)
        records[name]['metrics'] = result
        print(name, summary_row(result), flush=True)
        if name in differences:
            d = differences[name]
            records[name]['difference_from_python'] = {
                'forecast_mae_encoded': d['absolute_sum']/d['pixels'], 'forecast_max_encoded': d['max'],
                'equal_pixel_fraction': d['equal_pixels']/d['pixels'],
                'motion_mae_pixels_per_interval': d['motion_absolute_sum']/d['motion_values']}

    # All timing inputs in RAM; score computation has finished before clocks start.
    timed_inputs = read_test_samples(timing_ids)[:, :IN_FRAMES].copy()
    timing_hash = hashlib.sha256(timed_inputs.tobytes()).hexdigest()
    for mode, batch in [('single', 1), ('batch', args.batch)]:
        seconds = {name: [] for name in names}
        counts = {name: [] for name in names}
        stages = {name: [] for name in names[1:]}
        # Batch-specific warmup also removes Python thread-pool cold effects from repeats.
        for name in names:
            predict(name, timed_inputs[:batch])
        for repeat in range(args.repeats):
            # Rotate method order to distribute thermal and desktop load effects.
            order = names[repeat % len(names):]+names[:repeat % len(names)]
            for start in range(0,len(timed_inputs),batch):
                inputs = timed_inputs[start:start+batch]
                for name in order:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    predict(name,inputs)
                    elapsed = time.perf_counter()-tick  # native predict synchronizes before returning
                    seconds[name].append(elapsed)
                    counts[name].append(len(inputs))
                    if name in methods:
                        stages[name].extend(methods[name].stages)
            print(f'Timing {mode}: repeat {repeat+1}/{args.repeats}',flush=True)
        for name in names:
            records[name][mode] = statistics(seconds[name], counts[name])
            if name in stages:
                records[name][mode]['cuda_event_mean_ms'] = {
                    k: float(np.mean([v[k] for v in stages[name]])) for k in stages[name][0]}

    sources = ['flowcast/optical_flow.py','flowcast/optical_flow_cuda.py',
               'flowcasting_cuda/src/flowcast.cu','flowcasting_cuda/src/flowcast.h']
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'protocol': {'quality_windows': len(quality_ids), 'official_test_windows': total,
                            'quality_indices': quality_ids, 'quality_input_target_sha256': digest.hexdigest(),
                            'timing_indices': timing_ids, 'timing_input_sha256': timing_hash,
                            'repeats': args.repeats, 'batch': args.batch, 'cpu_workers': args.workers,
                            'opencv_threads_per_worker': 1, 'input_frames': IN_FRAMES, 'leads': OUT_FRAMES,
                            'shape': list(timed_inputs.shape[2:]),
                            'timing': 'Warm perf_counter wall time, NumPy input to NumPy forecast AND motion; '
                                      'includes transfers, allocation reuse, dispatch and synchronization; '
                                      'excludes disk I/O, scoring, build and context initialization. '
                                      'Single: batch 1; batch: CPU workers across windows, CUDA processes windows sequentially.'},
               'hardware': {'gpu': methods['CUDA'].device_name, 'cpu': platform.processor(),
                            'logical_cpus': os.cpu_count(), 'os': platform.platform(),
                            'opencv': cv2.__version__, 'numpy': np.__version__, 'torch': torch.__version__,
                            'cuda_runtime': methods['CUDA'].lib.fc_runtime_version()},
               'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
               'library_sha256': hashlib.sha256(methods['CUDA'].library_path.read_bytes()).hexdigest(),
               'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'methods': records}
    path = out/'cuda_comparison.json'
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('Wrote',path.resolve(),flush=True)
    # A matched visual; context and targets are identical for every implementation.
    from flowcast import viz
    example = read_test_samples([302])
    rows = {'Ground truth': example[0,IN_FRAMES:]}
    for name in names:
        rows[name] = predict(name,example[:,:IN_FRAMES])[0][0]
    (out/'figures').mkdir(exist_ok=True)
    viz.forecast_grid(out/'figures/cuda_comparison.png',example[0,:IN_FRAMES],rows,
                      title='SEVIR test #302: Python and CUDA on the same observations')
    for method in methods.values():
        method.close()


if __name__ == '__main__':
    main()
