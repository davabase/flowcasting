"""SEVIR VIL datasets: random training windows from raw events and the official test set."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent / 'data' / 'sevir'
TRAIN_DIR = ROOT / 'vil_train'
TEST_FILE = ROOT / 'data' / 'processed' / 'nowcast_testing_000.h5'
IN_FRAMES, OUT_FRAMES = 13, 12          # 65 min of context -> 60 min of forecast (5-min frames)
WINDOW = IN_FRAMES + OUT_FRAMES


def load_manifest(split=None, require_done=True, exclude_missing=True, directory=TRAIN_DIR):
    """Downloaded events from events.csv.

    exclude_missing drops events whose SEVIR catalog `pct_missing` is > 0: their gaps are filled with
    VIL 255 (whole frames read as an extreme storm), which corrupts both targets and scores.
    """
    directory = Path(directory)
    events = pd.read_csv(directory / 'events.csv')
    if exclude_missing:
        events = events[events.pct_missing == 0]
    if require_done:
        done_path = directory / 'events.done.npy'
        done = np.load(done_path) if done_path.exists() else np.zeros(len(events), bool)
        events = events[done[events.slot.values]]
    if split is not None:
        events = events[events.split == split]
    return events.reset_index(drop=True)


class _EventArray:
    """Reads windows from events.npy (N, 49, 384, 384) with one contiguous read each.

    A window of consecutive frames is a single contiguous block on disk, so a plain readinto is much
    faster than memmap page faults. The file is opened lazily per DataLoader worker (spawned on Windows).
    """

    _file = None
    _offset = None
    directory = TRAIN_DIR
    FRAME = 384 * 384

    def read(self, slot, start, frames):
        if self._file is None:
            header = np.load(Path(self.directory) / 'events.npy', mmap_mode='r')
            self._offset = header.offset
            del header
            self._file = open(Path(self.directory) / 'events.npy', 'rb', buffering=0)
        out = np.empty((frames, 384, 384), np.uint8)
        self._file.seek(self._offset + (int(slot) * 49 + int(start)) * self.FRAME)
        self._file.readinto(memoryview(out).cast('B'))
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state


class TrainWindows(_EventArray, torch.utils.data.Dataset):
    """Random 25-frame windows from random training events, with dihedral augmentation."""

    def __init__(self, samples_per_epoch=20000, augment=True, frames=WINDOW, seed=0, refresh_every=500):
        self.slots = load_manifest('train').slot.values
        self.samples_per_epoch, self.augment, self.frames, self.seed = samples_per_epoch, augment, frames, seed
        self.refresh_every, self.calls = refresh_every, 0

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # Pick up events that finished downloading since the last refresh (download order is shuffled).
        self.calls += 1
        if self.refresh_every and self.calls % self.refresh_every == 0:
            self.slots = load_manifest('train').slot.values
        rng = np.random.default_rng((self.seed, index, torch.initial_seed() % (2 ** 31)))
        slot = self.slots[rng.integers(len(self.slots))]
        start = rng.integers(0, 49 - self.frames + 1)
        x = self.read(slot, start, self.frames)
        if self.augment:
            x = np.rot90(x, k=rng.integers(4), axes=(1, 2))
            if rng.random() < 0.5:
                x = x[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x))


class ValWindows(_EventArray, torch.utils.data.Dataset):
    """Fixed windows from held-out events. Default starts mirror the test set's 3 windows per event."""

    def __init__(self, starts=(0, 12, 24), frames=WINDOW, max_events=None):
        slots = load_manifest('val').slot.values
        if max_events is not None:
            slots = slots[:max_events]
        self.items = [(s, t) for s in slots for t in starts if t + frames <= 49]
        self.frames = frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        slot, start = self.items[index]
        return torch.from_numpy(self.read(slot, start, self.frames))


def read_event(slot, directory=TRAIN_DIR):
    """Full 4-hour event (49, 384, 384) uint8 from an event store (training store by default)."""
    key = str(Path(directory).resolve())
    if key not in _READERS:
        _READERS[key] = _EventArray()
        _READERS[key].directory = Path(directory)
    return _READERS[key].read(slot, 0, 49)


_READERS = {}


def iter_test_blocks(path=TEST_FILE, block=117):
    """Yield (start index, uint8 array (n, 25, 384, 384)) from the official SEVIR nowcast test set.

    The HDF5 chunks span 117 samples, so reading block-aligned slices is ~100x faster than per sample.
    """
    with h5py.File(path, 'r') as source:
        count = source['IN_vil'].shape[0]
        for start in range(0, count, block):
            stop = min(count, start + block)
            x = np.concatenate([source['IN_vil'][start:stop], source['OUT_vil'][start:stop]], axis=-1)
            yield start, np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def read_test_samples(indices, path=TEST_FILE):
    with h5py.File(path, 'r') as source:
        out = []
        for i in indices:
            x = np.concatenate([source['IN_vil'][i], source['OUT_vil'][i]], axis=-1)
            out.append(x.transpose(2, 0, 1))
    return np.stack(out)
