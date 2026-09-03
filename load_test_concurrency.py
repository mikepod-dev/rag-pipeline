"""
load_test_concurrency.py

Module 3 (Tier 4): a genuine concurrency stress test, extending Finding 12's
original 5-concurrent-request test (which only measured aggregate batch time)
to 20+ simultaneous requests, watching for queueing behavior and backpressure
specifically -- not just whether the batch eventually finishes.

Measures two separate things per request:
  - enqueue_latency: time from firing the POST /ask request to getting a
    task_id back. This is the real signal for API-level backpressure -- if
    this balloons under load, FastAPI/Uvicorn itself is struggling to accept
    requests, independent of whether the worker can keep up.
  - completion_latency: time from firing the request to the task reaching
    "complete" via polling /result. This reflects real end-to-end queueing
    at the worker -- compare against the known single-request warm baseline
    (~3.7s, measured in Module 1) to see the real concurrency multiplier.

Uses the 20 real questions from golden_set.py (cycling if --n exceeds 20)
rather than a single repeated placeholder query.

Usage:
    python load_test_concurrency.py --n 5
    python load_test_concurrency.py --n 20 --host http://localhost:10000
"""

import argparse
import json
import statistics
import threading
import time

import requests

QUESTIONS = [
    "How many breeds of dogs are there?",
    "What's the scientific name for the wolf?",
    "Why do cats sleep so much?",
    "What color can a wolf's fur be?",
    "How long has coffee been cultivated?",
    "What continent did coffee originate from?",
    "Do dogs have a better sense of smell than humans?",
    "What is the DSM-5 diagnosis related to caffeine?",
    "What's the relationship between dogs and wolves?",
    "How many teeth do dogs have?",
    "What does the parasite-mediated domestication hypothesis suggest?",
    "What's the difference between the 2023 and 2026 remote work policies?",
    "What is espresso?",
    "Why were dogs originally domesticated, according to the commensal pathway theory?",
    "What's the product code for the coffee maker, and how many units were made?",
    "Do cats have a social survival strategy like herd behavior?",
    "What happened during the French officer's coffee voyage in 1723?",
    "What percentage of caffeine users develop tolerance to its sleep effects?",
    "What is domestication syndrome?",
    "Can dogs communicate with humans?",
]

results_lock = threading.Lock()
results = []


def fire_and_poll(host, index, question, poll_interval, timeout):
    entry = {"index": index, "question": question}
    submit_ts = time.time()
    try:
        resp = requests.post(f"{host}/ask", json={"query": question}, timeout=30)
        enqueue_ts = time.time()
        entry["enqueue_latency"] = enqueue_ts - submit_ts
        data = resp.json()
        if "task_id" not in data:
            entry["error"] = f"no task_id in /ask response: {data}"
            with results_lock:
                results.append(entry)
            return
        task_id = data["task_id"]
        entry["task_id"] = task_id
    except Exception as e:
        entry["error"] = f"/ask request failed: {e}"
        with results_lock:
            results.append(entry)
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/result/{task_id}", timeout=10)
            body = r.json()
        except Exception:
            time.sleep(poll_interval)
            continue
        if body.get("status") == "complete":
            completion_ts = time.time()
            entry["completion_latency"] = completion_ts - submit_ts
            entry["status"] = "complete"
            break
        if body.get("status") == "failed":
            entry["status"] = "failed"
            entry["error"] = body.get("error")
            break
        time.sleep(poll_interval)
    else:
        entry["status"] = "timeout"

    with results_lock:
        results.append(entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of concurrent requests")
    parser.add_argument("--host", type=str, default="http://localhost:10000")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.n)]

    print(f"Firing {args.n} concurrent requests against {args.host} ...")
    batch_start = time.time()

    threads = []
    for i, q in enumerate(questions):
        t = threading.Thread(
            target=fire_and_poll, args=(args.host, i, q, args.poll_interval, args.timeout)
        )
        threads.append(t)

    # start all threads as close to simultaneously as possible
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    batch_elapsed = time.time() - batch_start

    ordered = sorted(results, key=lambda r: r["index"])
    completed = [r for r in ordered if r.get("status") == "complete"]
    failed = [r for r in ordered if r.get("status") in ("failed", "timeout") or "error" in r]

    print("\n=== Results ===")
    print(f"Total wall-clock for batch: {batch_elapsed:.2f}s")
    print(f"Completed: {len(completed)}/{args.n}")
    print(f"Failed/timeout/error: {len(failed)}/{args.n}")

    if completed:
        enqueue_latencies = [r["enqueue_latency"] for r in completed]
        completion_latencies = [r["completion_latency"] for r in completed]
        print("\nEnqueue latency (time to get task_id back) -- API-level backpressure signal:")
        print(
            f"  min={min(enqueue_latencies):.3f}s  mean={statistics.mean(enqueue_latencies):.3f}s"
            f"  max={max(enqueue_latencies):.3f}s"
        )
        print("\nCompletion latency (submit -> task done) -- end-to-end queueing signal:")
        print(
            f"  min={min(completion_latencies):.2f}s  mean={statistics.mean(completion_latencies):.2f}s"
            f"  median={statistics.median(completion_latencies):.2f}s"
            f"  max={max(completion_latencies):.2f}s"
        )

    if failed:
        print("\n=== Failures ===")
        for r in failed:
            print(f"  index={r['index']}  status={r.get('status')}  error={r.get('error')}")

    report = {
        "n": args.n,
        "batch_elapsed_seconds": batch_elapsed,
        "n_completed": len(completed),
        "n_failed": len(failed),
        "results": ordered,
    }
    with open("load_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nReport written: load_test_report.json")


if __name__ == "__main__":
    main()
