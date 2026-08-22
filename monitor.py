import json
import statistics
from datetime import datetime, timedelta

LOG_PATH = "query_log.jsonl"

# Initial thresholds, reasoned from this project's own measurements across
# Modules 1-3, not derived from statistically significant production volume -
# a real production deployment would tune these against actual traffic over
# time rather than treat these as final.
THRESHOLDS = {
    # eval.py's 11 real questions have cost $0.0003-0.0009 each (see query_log.jsonl
    # history); at even heavy manual-testing volume (~50 calls/hour) that's under
    # $0.05/hour. Set well above realistic single-session cost, but well below what
    # a genuine cost-runaway (e.g. an infinite retry loop) would produce.
    "cost_per_hour_usd": 1.00,
    # Any non-zero error rate is worth flagging for a project at this traffic
    # scale - there's no "acceptable" background failure rate yet.
    "error_rate_pct": 5.0,
    # Real latency measurements: ~2-5s per query once models are warm (Module 2/3
    # testing), ~25-45s during cold model-loading. Set above the cold-start case
    # so a single cold start doesn't false-positive, but catches genuine
    # degradation beyond that.
    "avg_latency_ms": 10000,
    # A real, objective degradation signal - retrieval returning nothing usable,
    # independent of whether the LLM's answer "sounds" confident (Finding 8
    # already showed LLM-judge confidence is not reliable on its own).
    "empty_retrieval_rate_pct": 10.0,
}


def load_recent_entries(window_minutes=60):
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    entries = []

    try:
        with open(LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time >= cutoff:
                    entries.append(entry)
    except FileNotFoundError:
        return []

    return entries


def compute_metrics(entries, window_minutes):
    if not entries:
        return None

    total_cost = sum(e.get("cost", 0) for e in entries)
    window_hours = window_minutes / 60
    cost_per_hour = total_cost / window_hours if window_hours > 0 else 0

    total_calls = len(entries)
    failed_calls = sum(1 for e in entries if not e.get("success", True))
    error_rate_pct = (failed_calls / total_calls) * 100

    latencies = [e["latency_ms"] for e in entries if "latency_ms" in e]
    avg_latency_ms = statistics.mean(latencies) if latencies else 0

    empty_retrievals = sum(1 for e in entries if e.get("retrieval_count", 1) == 0)
    empty_retrieval_rate_pct = (empty_retrievals / total_calls) * 100

    return {
        "total_calls": total_calls,
        "cost_per_hour_usd": cost_per_hour,
        "error_rate_pct": error_rate_pct,
        "avg_latency_ms": avg_latency_ms,
        "empty_retrieval_rate_pct": empty_retrieval_rate_pct,
    }


def check_alerts(metrics):
    alerts = []

    if metrics["cost_per_hour_usd"] > THRESHOLDS["cost_per_hour_usd"]:
        alerts.append(
            f"COST ALERT: ${metrics['cost_per_hour_usd']:.4f}/hour exceeds threshold "
            f"${THRESHOLDS['cost_per_hour_usd']:.2f}/hour"
        )

    if metrics["error_rate_pct"] > THRESHOLDS["error_rate_pct"]:
        alerts.append(
            f"ERROR RATE ALERT: {metrics['error_rate_pct']:.1f}% exceeds threshold "
            f"{THRESHOLDS['error_rate_pct']:.1f}%"
        )

    if metrics["avg_latency_ms"] > THRESHOLDS["avg_latency_ms"]:
        alerts.append(
            f"LATENCY ALERT: {metrics['avg_latency_ms']:.0f}ms avg exceeds threshold "
            f"{THRESHOLDS['avg_latency_ms']:.0f}ms"
        )

    if metrics["empty_retrieval_rate_pct"] > THRESHOLDS["empty_retrieval_rate_pct"]:
        alerts.append(
            f"EMPTY RETRIEVAL ALERT: {metrics['empty_retrieval_rate_pct']:.1f}% exceeds "
            f"threshold {THRESHOLDS['empty_retrieval_rate_pct']:.1f}%"
        )

    return alerts


def run_check(window_minutes=60):
    entries = load_recent_entries(window_minutes)

    if not entries:
        print(f"No log entries in the last {window_minutes} minutes. Nothing to check.")
        return

    metrics = compute_metrics(entries, window_minutes)

    print(f"--- Monitor check: last {window_minutes} minutes ({metrics['total_calls']} calls) ---")
    print(f"  Cost/hour:        ${metrics['cost_per_hour_usd']:.4f}")
    print(f"  Error rate:       {metrics['error_rate_pct']:.1f}%")
    print(f"  Avg latency:      {metrics['avg_latency_ms']:.0f}ms")
    print(f"  Empty retrieval:  {metrics['empty_retrieval_rate_pct']:.1f}%")

    alerts = check_alerts(metrics)

    if alerts:
        print("\n!!! ALERTS TRIGGERED !!!")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\nAll metrics within thresholds.")


if __name__ == "__main__":
    run_check()
