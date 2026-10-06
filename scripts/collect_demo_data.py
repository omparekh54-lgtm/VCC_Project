"""Collect a short, staged sample of actual server load for a demo model."""
import concurrent.futures
import random
import time
import urllib.request

def worker(until, pause):
    count = 0
    while time.monotonic() < until:
        try:
            intensity = random.randint(1, 10)
            with urllib.request.urlopen(
                f'http://127.0.0.1:8000/api/simulate?intensity={intensity}', timeout=20) as response:
                response.read()
            count += 1
        except Exception:
            pass
        if pause:
            time.sleep(pause)
    return count

if __name__ == '__main__':
    for concurrency, pause in [(1, 0.1), (4, 0.03), (12, 0), (32, 0), (2, 0.1)]:
        until = time.monotonic() + 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            counts = list(pool.map(lambda _: worker(until, pause), range(concurrency)))
        print({'concurrency': concurrency, 'requests': sum(counts)}, flush=True)
