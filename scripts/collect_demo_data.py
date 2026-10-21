"""Collect varied, repeatable load cycles for Infra Watchdog validation.

The ideal review requires evidence across many unseen load shapes and at least
10 slowdown events. This generator can run gradual ramps, sudden spikes, long
holds and bursty mixes. Use --profile all --cycles 12 for a research-quality
campaign, then split the resulting CSV by cycle/run before walk-forward tests.
"""
import argparse
import random
import threading
import time
import urllib.request


def wanted_clients(profile, t, cycle_s, peak):
    x = t / cycle_s
    if profile == "ramp":
        if x < .12: return 1
        if x < .55: return 1 + round((peak - 1) * (x - .12) / .43)
        if x < .72: return peak
        if x < .90: return max(1, round(peak * (1 - (x - .72) / .18)))
        return 1
    if profile == "spike":
        if x < .25: return 1
        if x < .55: return peak
        if x < .65: return max(1, peak // 3)
        if x < .82: return peak
        return 1
    if profile == "hold":
        if x < .15: return 1
        if x < .35: return 1 + round((peak - 1) * (x - .15) / .20)
        if x < .82: return peak
        return 1
    # mixed: several smaller surges with a rising baseline.
    base = max(1, round(1 + (peak * .35) * x))
    burst = peak if (.22 < x < .31 or .53 < x < .63 or .76 < x < .88) else base
    return min(peak, max(1, burst))


def worker(index, wanted, stop, host, counts):
    while not stop.is_set():
        if index >= wanted[0]:
            time.sleep(0.15)
            continue
        try:
            intensity = random.randint(1, 10)
            with urllib.request.urlopen(f"{host}/api/simulate?intensity={intensity}", timeout=30) as r:
                r.read()
            counts[index] += 1
        except Exception:
            time.sleep(0.35)
        time.sleep(random.uniform(0.02, 0.18))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8000")
    p.add_argument("--cycles", type=int, default=3,
                   help="Use 12 or more for the ideal-version evidence campaign")
    p.add_argument("--peak", type=int, default=24)
    p.add_argument("--cycle-s", type=float, default=480)
    p.add_argument("--profile", choices=["ramp", "spike", "hold", "mixed", "all"], default="ramp")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    profiles = ["ramp", "spike", "hold", "mixed"] if args.profile == "all" else [args.profile]
    wanted = [1]
    counts = [0] * args.peak
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(i, wanted, stop, args.host, counts), daemon=True)
               for i in range(args.peak)]
    for th in threads:
        th.start()

    start = time.monotonic()
    total_s = args.cycle_s * args.cycles
    print(f"campaign: {args.cycles} cycles x {args.cycle_s/60:.1f} min against {args.host}", flush=True)
    try:
        last_report = -1
        while (elapsed := time.monotonic() - start) < total_s:
            cycle = int(elapsed // args.cycle_s)
            cycle_t = elapsed % args.cycle_s
            profile = profiles[cycle % len(profiles)]
            wanted[0] = wanted_clients(profile, cycle_t, args.cycle_s, args.peak)
            sec = int(elapsed)
            if sec // 15 != last_report:
                last_report = sec // 15
                print(f"cycle={cycle+1:02d}/{args.cycles} profile={profile:<5} t={cycle_t:5.0f}s clients={wanted[0]:2d} requests={sum(counts)}", flush=True)
            time.sleep(1)
    finally:
        stop.set()
    print(f"done: {sum(counts)} requests", flush=True)


if __name__ == "__main__":
    main()
