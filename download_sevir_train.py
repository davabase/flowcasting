"""Download a training subset of raw SEVIR VIL storm/random events (resumable).

The raw SEVIR VIL files store a contiguous, uncompressed uint8 dataset
`vil` shaped (events, 384, 384, 49). That lets us fetch any single 4-hour
event with one HTTP byte-range request instead of downloading 10-17 GB files.

Selected events are written, time-first, into one memory-mapped array:

    data/sevir/vil_train/events.npy   (N, 49, 384, 384) uint8
    data/sevir/vil_train/events.csv   one row per slot (id, time, split, ...)

Only events before 2019-06-01 are eligible, so nothing overlaps the official
SEVIR nowcast test set (events from 2019-06-11 to 2019-11-30).
"""
import argparse
import concurrent.futures
import io
import pathlib
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent / 'data' / 'sevir'
BUCKET = 'https://sevir.s3.us-west-2.amazonaws.com/'
H = W = 384
T = 49
EVENT_BYTES = H * W * T
CUTOFF = '2019-06-01'


class RangeFile(io.RawIOBase):
    """Minimal read-only, block-cached HTTP file so h5py can parse remote headers."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        head = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
        self.size = int(head.headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, index):
        if index not in self.cache:
            start = index * self.block
            end = min(self.size, start + self.block) - 1
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            self.cache[index] = urllib.request.urlopen(request, timeout=120).read()
        return self.cache[index]

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        while len(out) < n:
            index, offset = divmod(self.pos + len(out), self.block)
            out += self._block(index)[offset:offset + n - len(out)]
        buffer[:n] = out
        self.pos += n
        return n


def remote_layout(file_name):
    """Return (byte offset of `vil`, number of events, event ids) for one raw file."""
    raw = RangeFile(BUCKET + 'data/' + file_name)
    reader = io.BufferedReader(raw, 1 << 20)
    source = h5py.File(reader, 'r')
    try:
        vil = source['vil']
        if vil.chunks is not None or vil.compression is not None or vil.dtype != np.uint8:
            raise RuntimeError(f'{file_name}: vil is not a contiguous uint8 dataset')
        if vil.shape[1:] != (H, W, T):
            raise RuntimeError(f'{file_name}: unexpected vil shape {vil.shape}')
        layout = vil.id.get_offset(), vil.shape[0], [i.decode() for i in source['id'][:]]
    finally:
        source.close()
        reader.close()
    return layout


def select_events(catalog, budget_gb, val_events, seed, start=None, end=CUTOFF, max_storm=None):
    vil = catalog[catalog.img_type == 'vil'].copy()
    vil['time_utc'] = pd.to_datetime(vil.time_utc)
    vil = vil[(vil.time_utc < end) & (vil.time_utc >= (start or '1900-01-01'))].drop_duplicates('id')
    vil = vil[vil.pct_missing == 0] if start else vil   # evaluation sets: skip events with missing-data fill
    vil['kind'] = np.where(vil.id.str.startswith('S'), 'storm', 'random')
    rng = np.random.default_rng(seed)
    total = int(budget_gb * 1e9 // EVENT_BYTES)
    storms = vil[vil.kind == 'storm']
    if max_storm is not None:
        storms = storms.sample(n=min(len(storms), max_storm), random_state=seed)
    randoms = vil[vil.kind == 'random']
    n_random = max(0, min(len(randoms), total - len(storms)))
    storms = storms.sample(n=min(len(storms), total), random_state=seed)
    randoms = randoms.sample(n=n_random, random_state=seed)
    chosen = pd.concat([storms, randoms])
    # Shuffle so a partially finished download is still a representative subset.
    chosen = chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)
    chosen['split'] = 'train'
    val = rng.choice(len(chosen), size=min(val_events, len(chosen)), replace=False)
    chosen.loc[val, 'split'] = 'val'
    chosen.insert(0, 'slot', np.arange(len(chosen)))
    keep = ['slot', 'id', 'file_name', 'file_index', 'time_utc', 'kind', 'split',
            'event_type', 'data_max', 'pct_missing', 'llcrnrlat', 'llcrnrlon', 'urcrnrlat', 'urcrnrlon']
    return chosen[keep]


def fetch(url, offset, index):
    start = offset + int(index) * EVENT_BYTES
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + EVENT_BYTES - 1}'})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise RuntimeError('Server did not honor byte range')
                data = response.read()
            if len(data) != EVENT_BYTES:
                raise RuntimeError('Incomplete byte range')
            # (384, 384, 49) -> (49, 384, 384): frames become contiguous for fast window reads.
            return np.frombuffer(data, np.uint8).reshape(H, W, T).transpose(2, 0, 1)
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget-gb', type=float, default=40.0, help='Download budget (each event is 7.2 MB)')
    parser.add_argument('--val-events', type=int, default=160, help='Events held out for validation')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='vil_train', help='Folder under data/sevir')
    parser.add_argument('--start', default=None, help='Earliest event time (inclusive), e.g. 2019-06-11')
    parser.add_argument('--end', default=CUTOFF, help='Latest event time (exclusive)')
    parser.add_argument('--max-storm', type=int, default=None, help='Cap on storm events (rest of budget is random)')
    args = parser.parse_args()

    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT / 'CATALOG.csv'
    if not catalog_path.exists():
        urllib.request.urlretrieve(BUCKET + 'CATALOG.csv', catalog_path)
    manifest_path, array_path, done_path = OUT / 'events.csv', OUT / 'events.npy', OUT / 'events.done.npy'

    if manifest_path.exists():
        events = pd.read_csv(manifest_path)
        print(f'Resuming: {len(events)} events in existing manifest')
    else:
        events = select_events(pd.read_csv(catalog_path, low_memory=False), args.budget_gb, args.val_events, args.seed,
                               start=args.start, end=args.end, max_storm=args.max_storm)
        events.to_csv(manifest_path, index=False)
    n = len(events)
    print(f'{n} events ({(events.kind == "storm").sum()} storm, {(events.kind == "random").sum()} random), '
          f'{n * EVENT_BYTES / 1e9:.1f} GB, {(events.split == "val").sum()} held out for validation', flush=True)

    layouts = {}
    for file_name in sorted(events.file_name.unique()):
        offset, count, ids = remote_layout(file_name)
        rows = events[events.file_name == file_name]
        if rows.file_index.max() >= count or any(ids[i] != e for i, e in zip(rows.file_index, rows.id)):
            raise RuntimeError(f'{file_name}: catalog file_index does not match the event ids in the file')
        layouts[file_name] = offset
        print(f'  verified {file_name}: vil @ byte {offset}, {len(rows)} of {count} events selected', flush=True)

    if array_path.exists():
        array = np.lib.format.open_memmap(array_path, mode='r+')
        if array.shape != (n, T, H, W):
            raise RuntimeError(f'{array_path} has shape {array.shape}, expected {(n, T, H, W)}')
    else:
        array = np.lib.format.open_memmap(array_path, mode='w+', dtype=np.uint8, shape=(n, T, H, W))
    done = np.load(done_path) if done_path.exists() else np.zeros(n, bool)
    todo = [int(s) for s in np.flatnonzero(~done)]
    print(f'{len(todo)} events left to download', flush=True)

    def job(slot):
        row = events.iloc[slot]
        array[slot] = fetch(BUCKET + 'data/' + row.file_name, layouts[row.file_name], row.file_index)
        return slot

    start, last = time.monotonic(), time.monotonic()
    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(job, s) for s in todo]):
            done[future.result()] = True
            finished += 1
            if time.monotonic() - last > 30 or finished == len(todo):
                array.flush()
                np.save(done_path, done)
                rate = finished * EVENT_BYTES / 1e6 / (time.monotonic() - start)
                eta = (len(todo) - finished) * EVENT_BYTES / 1e6 / max(rate, 1e-6) / 60
                print(f'{done.sum()}/{n} events  {done.sum() * EVENT_BYTES / 1e9:.1f} GB  '
                      f'{rate:.0f} MB/s  ETA {eta:.0f} min', flush=True)
                last = time.monotonic()
    array.flush()
    np.save(done_path, done)
    print('Complete' if done.all() else 'Incomplete: rerun to resume', flush=True)


if __name__ == '__main__':
    main()
