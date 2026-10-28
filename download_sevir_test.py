"""Download and extract the official SEVIR nowcasting test set (resumable)."""
import concurrent.futures
import pathlib
import time
import urllib.request
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent / 'data' / 'sevir'
URL = 'https://sevir.s3.us-west-2.amazonaws.com/data/processed/'
NAME = 'nowcast_testing_000.h5.tar.gz'
SIZE = 5119067375
CHUNK = 8 * 1024 * 1024

def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = ROOT / '.test-parts' / str(index)
    if path.exists() and path.stat().st_size == end - start + 1:
        return
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL + NAME, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError('Server did not honor byte range')
                data = response.read()
            if len(data) != end - start + 1:
                raise RuntimeError('Incomplete byte range')
            path.write_bytes(data)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(attempt + 1)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / '.test-parts').mkdir(exist_ok=True)
    archive = ROOT / NAME
    if not archive.exists() or archive.stat().st_size != SIZE:
        count = (SIZE + CHUNK - 1) // CHUNK
        last = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(fetch, i) for i in range(count)]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                if time.monotonic() - last > 20 or done == count:
                    total = sum(p.stat().st_size for p in (ROOT / '.test-parts').iterdir())
                    print(f'Downloaded {total / 1e9:.2f} / {SIZE / 1e9:.2f} GB', flush=True)
                    last = time.monotonic()
        with archive.open('wb') as output:
            for i in range(count):
                with (ROOT / '.test-parts' / str(i)).open('rb') as source:
                    while block := source.read(CHUNK):
                        output.write(block)
        if archive.stat().st_size != SIZE:
            raise RuntimeError('Archive size mismatch')
        for part in (ROOT / '.test-parts').iterdir():
            part.unlink()
        (ROOT / '.test-parts').rmdir()
    urllib.request.urlretrieve(URL + 'nowcast_testing_000_META.csv', ROOT / 'nowcast_testing_000_META.csv')
    print('Extracting and checking gzip integrity...', flush=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            print(f'{member.name}: {member.size:,} bytes', flush=True)
            tar.extract(member, ROOT, filter='data')
    print('Complete', flush=True)

if __name__ == '__main__':
    main()
