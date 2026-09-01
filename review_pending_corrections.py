"""
review_pending_corrections.py

The manual approval gate. Every "correction" ragas_gate.py produces lands in a
pending_review_v{n}.jsonl staging file, never directly in a golden dataset -- because
an early run of the gate produced a confirmed false correction (deleted true information
due to a retrieval-width limitation) with nothing in the automated pipeline catching it.
This script is the human check that catches what the automated pipeline can't.

For each pending record, shows the question, the original (flawed) answer, the judge's
extracted claims, the context actually used for judging, and the teacher's proposed
correction -- then asks for an explicit approve/reject/skip decision. Approved records
are appended to golden_dataset_v{n}.jsonl (same version number as the pending file being
reviewed). Rejected and skipped records are never silently discarded -- the full decision
record, including who/when and any reviewer note, is written to a review log so there's
an audit trail of what was rejected and why.

Usage:
    python review_pending_corrections.py --version 1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def print_record(record: dict, index: int, total: int) -> None:
    print("\n" + "=" * 70)
    print(f"Record {index + 1}/{total}" + (" [CANARY]" if record.get("is_canary") else ""))
    print("=" * 70)
    print(f"QUESTION: {record['question']}")
    print(f"\nORIGINAL (flawed) ANSWER:\n{record['original_answer']}")
    print(f"\nORIGINAL FAITHFULNESS SCORE: {record['original_faithfulness']:.3f}")
    print("\nCONTEXT USED FOR JUDGING:")
    for i, chunk in enumerate(record.get("context_used", [])):
        print(f"  [{i + 1}] {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
    print(f"\nTEACHER'S EXPLANATION: {record['explanation']}")
    print(f"\nPROPOSED CORRECTED ANSWER:\n{record['corrected_answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=int, required=True, help="pending_review_v{N}.jsonl to review"
    )
    args = parser.parse_args()

    pending_path = Path(f"pending_review_v{args.version}.jsonl")
    golden_path = Path(f"golden_dataset_v{args.version}.jsonl")
    review_log_path = Path(f"review_log_v{args.version}.jsonl")

    if not pending_path.exists():
        print(f"FATAL: {pending_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(pending_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    pending_only = [r for r in records if r.get("review_status") == "PENDING"]
    print(f"Loaded {len(records)} records, {len(pending_only)} still pending review.")

    if not pending_only:
        print("Nothing to review.")
        return

    approved = 0
    rejected = 0
    skipped = 0

    with (
        open(golden_path, "a", encoding="utf-8") as golden_f,
        open(review_log_path, "a", encoding="utf-8") as log_f,
    ):
        for i, record in enumerate(pending_only):
            print_record(record, i, len(pending_only))
            decision = input("\n[a]pprove / [r]eject / [s]kip / [q]uit: ").strip().lower()

            if decision == "q":
                print("Stopping review. Unreviewed records remain PENDING for next time.")
                break

            timestamp = datetime.now(timezone.utc).isoformat()
            log_entry = {"question": record["question"], "timestamp": timestamp}

            if decision == "a":
                note = input("Optional note (Enter to skip): ").strip()
                golden_record = dict(record)
                golden_record["review_status"] = "APPROVED"
                golden_record["reviewed_at"] = timestamp
                golden_record["reviewer_note"] = note or None
                golden_f.write(json.dumps(golden_record) + "\n")
                golden_f.flush()
                log_entry["decision"] = "APPROVED"
                log_entry["note"] = note or None
                approved += 1
            elif decision == "r":
                note = input("Reason for rejection: ").strip()
                log_entry["decision"] = "REJECTED"
                log_entry["note"] = note
                rejected += 1
            else:
                log_entry["decision"] = "SKIPPED"
                skipped += 1

            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

    print("\n" + "=" * 70)
    print(f"Approved: {approved}, Rejected: {rejected}, Skipped: {skipped}")
    print(f"Golden dataset: {golden_path.resolve()}")
    print(f"Review log: {review_log_path.resolve()}")


if __name__ == "__main__":
    main()
