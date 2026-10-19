"""
Does the watchdog actually scale earlier than simpler policies?

Replays a load-test log (metrics_log.csv) the way the guardrail Lambda sees
the server -- one look every --step-s seconds -- and records when each
policy would have asked for another instance:

  Model forecast   latency forecast H s ahead > threshold           (this project)
  Latency now      latency of the last 10 s > threshold              (reactive, same signal)
  CPU, 1 min       60 s average CPU > 70 %                           (an eager CPU rule)
  CPU, 3 min       60 s average CPU > 70 % for 3 minutes running     (an AWS target-tracking
                                                                       alarm's default shape)

"Latency now" is the honest baseline: if forecasting can't beat simply
measuring latency, the model isn't adding anything.

It then finds every slowdown -- a stretch where the latency users actually
got stayed above the threshold -- and reports, per policy, how many seconds
before (or after) the slowdown began it raised the alarm, which slowdowns it
missed, and how many alarms came with no slowdown behind them.

The model is scored on data it never saw: it is trained on the earlier part
of the log (or on all of --data when --test-data is a separate run) and
replayed on the rest. With a cyclic load test, hold out at least one whole
cycle (the default --test-fraction 0.35 does that for 3 cycles).

    pip install pandas scikit-learn joblib matplotlib
    python ml/compare_scaling.py --data metrics_log.csv
    python ml/compare_scaling.py --data run1.csv --test-data run2.csv

Writes scaling_comparison.png and scaling_comparison.json (see --out).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (FEATURES, LATENCY, add_model_args, add_targets, fit, forecast,  # noqa: E402
                   load_training_data, new_model, recent_latency_at, time_split, window_mean)

POLICIES = ["Model forecast", "Latency now", "CPU, 1 min", "CPU, 3 min"]
RES_S = 5.0  # time resolution for measuring actual latency and slowdowns


def replay(log: pd.DataFrame, test_start: float, model, args) -> pd.DataFrame:
    """
    One row per RES_S seconds of the test period. Policies only act on rows
    where the guardrail looks (every --step-s); `log` includes the time before
    the test period so the CPU rules have their history.
    """
    ts = log["timestamp"].to_numpy(dtype=float)
    rt = log[LATENCY].to_numpy(dtype=float)
    cpu = log["cpu_usage"].to_numpy(dtype=float)
    grid = np.arange(test_start, ts[-1] - args.window_s, RES_S)
    if len(grid) == 0:
        return pd.DataFrame()

    # What /metrics would return at each moment: readings as of the latest
    # request to arrive, plus latency of requests that finished in the last 10 s.
    latest = np.maximum(np.searchsorted(ts, grid, side="right") - 1, 0)
    feats = pd.DataFrame({f: log[f].to_numpy(dtype=float)[latest] for f in FEATURES})
    feats["recent_latency_ms"] = recent_latency_at(log, grid)
    pred = forecast(model, feats[FEATURES].values)

    cpu60 = lambda at: np.nan_to_num(window_mean(ts, cpu, at - 60, at), nan=0.0)  # noqa: E731
    over = lambda at: cpu60(at) > args.cpu_threshold  # noqa: E731
    looks = ((grid - test_start) % args.step_s) < RES_S / 2

    return pd.DataFrame({
        "t": grid,
        # Latency users got from requests arriving in the next W seconds.
        "actual_ms": window_mean(ts, rt, grid, grid + args.window_s),
        "recent_ms": feats["recent_latency_ms"].to_numpy(),
        "predicted_ms": pred,
        "cpu60": cpu60(grid),
        "looks": looks,
        "Model forecast": looks & (pred > args.threshold_ms),
        "Latency now": looks & (feats["recent_latency_ms"].to_numpy() > args.threshold_ms),
        "CPU, 1 min": looks & over(grid),
        "CPU, 3 min": looks & over(grid) & over(grid - 60) & over(grid - 120),
    })


def runs(mask: np.ndarray, t: np.ndarray, merge_gap_s: float):
    """[start, end] times of runs of True, merging runs separated by < merge_gap_s."""
    out = []
    for i in np.flatnonzero(mask):
        if out and t[i] - out[-1][1] < merge_gap_s:
            out[-1][1] = t[i]
        else:
            out.append([t[i], t[i]])
    return out


def score(steps: pd.DataFrame, args) -> dict:
    t = steps["t"].to_numpy()
    slow = (steps["actual_ms"] > args.threshold_ms).to_numpy()
    incidents = runs(slow, t, merge_gap_s=60)
    lookback = args.lookback_s
    result = {"incidents": [{"start": s, "end": e} for s, e in incidents], "policies": {}}

    gap = args.step_s * 1.5  # consecutive guardrail looks
    for name in POLICIES:
        alarm = steps[name].to_numpy()
        # Alarm episodes: consecutive looks that all said "scale out".
        episodes = runs(alarm, t, merge_gap_s=gap)
        # An episode warns about a slowdown if it is still going when the slowdown
        # starts (or ends at the look just before it) and began no more than
        # `lookback` earlier; anything else is a false alarm.
        linked = lambda a, b, s, e: b >= s - gap and a <= e  # noqa: E731
        leads = []
        for s, e in incidents:
            starts = [a for a, b in episodes if linked(a, b, s, e)]
            leads.append(float(min(s - min(starts), lookback)) if starts else None)
        false_alarms = sum(1 for a, b in episodes if not any(linked(a, b, s, e) for s, e in incidents))
        caught = [x for x in leads if x is not None]
        result["policies"][name] = {
            "lead_s": leads,
            "caught": len(caught),
            "missed": len(leads) - len(caught),
            "median_lead_s": float(np.median(caught)) if caught else None,
            "false_alarms": false_alarms,
            # Seconds of slow responses before the policy even asked for capacity
            # (a new instance then still needs a few minutes to boot).
            "slow_s_before_alarm": float(sum(max(0.0, -x) if x is not None else e - s + RES_S
                                             for x, (s, e) in zip(leads, incidents))),
        }
    return result


def print_summary(result: dict, args) -> None:
    n = len(result["incidents"])
    print(f"\n{n} slowdown(s) where actual latency stayed above {args.threshold_ms:g} ms\n")
    print(f"{'policy':<15} {'caught':>7} {'median lead':>12} {'slow before alarm':>18} {'false alarms':>13}")
    for name, p in result["policies"].items():
        lead = "none" if p["median_lead_s"] is None else f"{p['median_lead_s']:+.0f} s"
        print(f"{name:<15} {p['caught']:>3}/{n:<3} {lead:>12} {p['slow_s_before_alarm']:>16.0f} s {p['false_alarms']:>13}")
    print("\nlead: + means the alarm came that many seconds BEFORE the slowdown began, - means after.")


def plot(steps: pd.DataFrame, result: dict, args, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Beige theme from the dashboard; series colors validated for CVD separation.
    surface, ink, ink2, ink3, grid = "#faf4ea", "#2e1f14", "#5f4a37", "#8e7660", "#ebdfcb"
    actual_c, pred_c, cpu_c, crit = "#9c5a2a", "#1c6fa3", "#5f4a37", "#b4462d"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "text.color": ink, "axes.labelcolor": ink2,
        "axes.edgecolor": grid, "xtick.color": ink3, "ytick.color": ink3,
        "axes.facecolor": surface, "figure.facecolor": surface,
    })
    t0 = steps["t"].iloc[0]
    m = (steps["t"] - t0) / 60
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.6], "hspace": 0.12})

    for ax in (ax1, ax2, ax3):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=1)
        ax.set_axisbelow(True)
        for inc in result["incidents"]:
            ax.axvspan((inc["start"] - t0) / 60, (inc["end"] - t0 + RES_S) / 60, color=crit, alpha=0.08, linewidth=0)

    # 1) latency: actual vs the forecast made H seconds earlier for that moment
    pred_for_now = steps["predicted_ms"].shift(round(args.horizon_s / RES_S))
    ax1.plot(m, steps["actual_ms"], color=actual_c, linewidth=2, label="Actual latency")
    ax1.plot(m, pred_for_now, color=pred_c, linewidth=2, label=f"Forecast made {args.horizon_s:g} s earlier")
    ax1.axhline(args.threshold_ms, color=crit, linewidth=1, linestyle=(0, (4, 3)))
    ax1.text(m.iloc[-1], args.threshold_ms, f"threshold {args.threshold_ms:g} ms  ", color=crit,
             ha="right", va="bottom", fontsize=9)
    ax1.set_ylabel("ms")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", frameon=False, ncol=2)
    ax1.set_title("When would each policy have asked for another server?", loc="left",
                  fontsize=13, fontweight="bold", color=ink, pad=12)

    # 2) CPU, the signal the classic rules use
    ax2.plot(m, steps["cpu60"], color=cpu_c, linewidth=2)
    ax2.axhline(args.cpu_threshold, color=crit, linewidth=1, linestyle=(0, (4, 3)))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("CPU % (60 s avg)")

    # 3) alarm strip: one row per policy; color follows the signal it uses
    colors = {"Model forecast": pred_c, "Latency now": actual_c, "CPU, 1 min": cpu_c, "CPU, 3 min": cpu_c}
    for row, name in enumerate(POLICIES):
        on = steps[name].to_numpy()
        y = len(POLICIES) - 1 - row
        ax3.scatter(m[on], np.full(on.sum(), y), marker="|", s=140, linewidths=2.2, color=colors[name])
        p = result["policies"][name]
        lead = "missed" if p["median_lead_s"] is None else f"{p['median_lead_s']:+.0f} s"
        ax3.text(1.005, y, lead, transform=ax3.get_yaxis_transform(), va="center", fontsize=9, color=ink2)
    ax3.set_yticks(range(len(POLICIES)))
    ax3.set_yticklabels(POLICIES[::-1])
    ax3.set_ylim(-0.6, len(POLICIES) - 0.4)
    ax3.grid(False)
    ax3.tick_params(axis="y", length=0)
    ax3.set_xlabel("minutes into the replayed test period   (shaded = actual slowdown; right: median lead)")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_model_args(parser)
    parser.add_argument("--data", required=True, help="metrics_log.csv, or a folder of them")
    parser.add_argument("--test-data", help="separate run to replay; default: newest part of --data")
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--threshold-ms", type=float, default=500)
    parser.add_argument("--cpu-threshold", type=float, default=70)
    parser.add_argument("--step-s", type=float, default=30, help="guardrail check interval")
    parser.add_argument("--lookback-s", type=float, default=180,
                        help="longest lead credited: an alarm episode running into a slowdown counts from at most this long before it")
    parser.add_argument("--out", default="scaling_comparison", help="output path without extension")
    args = parser.parse_args()

    raw = load_training_data(args.data)
    data = add_targets(raw, args.horizon_s, args.window_s)
    if args.test_data:
        train, log = data, load_training_data(args.test_data)
        test_start = log["timestamp"].iloc[0]
    else:
        train, test = time_split(data, args.test_fraction, args.horizon_s, args.window_s)
        log, test_start = raw, test["timestamp"].iloc[0]

    model = fit(new_model(args), train)
    span_min = (log["timestamp"].iloc[-1] - test_start) / 60
    print(f"trained on {len(train)} rows, replaying {span_min:.1f} unseen minutes")

    steps = replay(log, test_start, model, args)
    if steps.empty or steps["looks"].sum() < 5:
        sys.exit("Test period too short to replay; collect a longer load test.")
    result = score(steps, args)
    print_summary(result, args)

    result["settings"] = {k: getattr(args, k) for k in
                          ("horizon_s", "window_s", "threshold_ms", "cpu_threshold", "step_s", "lookback_s")}
    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=2)
    plot(steps, result, args, args.out + ".png")
    print(f"\nwrote {args.out}.png and {args.out}.json")


if __name__ == "__main__":
    main()
