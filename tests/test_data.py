"""Concurrent experiment workers must never read another event's input or target."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from flowcast.data import _EventArray


class DelayedReader:
    def __init__(self, file):
        self.file = file

    def seek(self, offset):
        self.file.seek(offset)
        time.sleep(0.002)  # expose an unlocked shared seek/read race deterministically

    def readinto(self, buffer):
        return self.file.readinto(buffer)


class EventReaderTests(unittest.TestCase):
    def test_parallel_windows_preserve_event_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.npy'
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                              shape=(3, 49, 384, 384))
            for slot in range(3):
                array[slot] = (slot * 60 + np.arange(49, dtype=np.uint8))[:, None, None]
            offset = array.offset
            array.flush()
            del array
            with path.open('rb', buffering=0) as file:
                reader = _EventArray()
                reader._file, reader._offset = DelayedReader(file), offset
                jobs = [(slot, start) for _ in range(3) for slot in range(3) for start in (0, 12, 24)]
                with ThreadPoolExecutor(4) as pool:
                    windows = list(pool.map(lambda job: reader.read(job[0], job[1], 4), jobs))
                for (slot, start), window in zip(jobs, windows):
                    expected = np.broadcast_to((slot * 60 + start + np.arange(4))[:, None, None], window.shape)
                    np.testing.assert_array_equal(window, expected)

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryFile() as file:
            reader = _EventArray()
            reader._file, reader._offset = file, 0
            with self.assertRaises(EOFError):
                reader.read(0, 0, 1)


if __name__ == '__main__':
    unittest.main()
