"""
Collect real server load data for a demo model, without Locust.

Runs repeated load cycles against /api/simulate: quiet, a gradual ramp up to
--peak parallel clients, a short hold, a ramp down, quiet again. Gradual
ramps matter: the model forecasts latency 30 s ahead (ml/train.py), so it
needs to see load building up before latency degrades, not just jumps.
Several cycles give compare_scaling.py more than one slowdown to score.

Run it on the app server (or anywhere that can reach it):
    python3 scripts/collect_demo_data.py --cycles 3
which takes about 8 minutes per cycle with the defaults.
"""
import argparse
import random
import threading
import time
import urllib.request


def load_profile(t: float, ramp_s: float, hold_s: float, quiet_s: float, peak: int) -> int:
    """Parallel clients wanted at `t` seconds into one cycle."""
    if t < quiet_s:
        return 1
    t -= quiet_s
    if t < ramp_s:
        return 1 + round((peak - 1) * t / ramp_s)
    t -= ramp_s
    if t < hold_s:
        return peak
    t -= hold_s
    if t < ramp_s / 2:
        return max(1, round(peak * (1 - t / (ramp_s / 2))))
    return 1


def worker(index: int, wanted: list, stop: threading.Event, host: str, counts: list) -> None:
    while not stop.is_set():
        if index >= wanted[0]:
            time.sleep(0.2)
            continue
        try:
            intensity = random.randint(1, 10)
            with urllib.request.urlopen(f"{host}/api/simulate?intensity={intensity}", timeout=30) as r:
                r.read()
            counts[index] += 1
        except Exception:
            time.sleep(0.5)
        time.sleep(random.uniform(0.02, 0.2))  # think time, like Locust's wait_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--peak", type=int, default=24, help="parallel clients at the top of each ramp")
    parser.add_argument("--ramp-s", type=float, default=240)
    parser.add_argument("--hold-s", type=float, default=60)
    parser.add_argument("--quiet-s", type=float, default=60)
    args = parser.parse_args()

    cycle_s = args.quiet_s + args.ramp_s + args.hold_s + args.ramp_s / 2 + args.quiet_s
    wanted = [1]
    counts = [0] * args.peak
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(i, wanted, stop, args.host, counts), daemon=True)
               for i in range(args.peak)]
    for th in threads:
        th.start()

    start = time.monotonic()
    total_s = cycle_s * args.cycles
    print(f"{args.cycles} cycles of {cycle_s / 60:.1f} min against {args.host}", flush=True)
    try:
        while (elapsed := time.monotonic() - start) < total_s:
            wanted[0] = load_profile(elapsed % cycle_s, args.ramp_s, args.hold_s, args.quiet_s, args.peak)
            if int(elapsed) % 15 == 0:
                print(f"t={elapsed:5.0f}s clients={wanted[0]:2d} requests={sum(counts)}", flush=True)
            time.sleep(1)
    finally:
        stop.set()
    print(f"done: {sum(counts)} requests", flush=True)


if __name__ == "__main__":
    main()
