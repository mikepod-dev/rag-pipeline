"""
finalize_module1_dataset.py

Closes out Module 1 (rewritten) by adding the two pieces the curriculum spec
requires that generate_split_brain_data.py doesn't yet produce:

1. Real tokenizer-based length audit (Section 3a) -- using the actual
   meta-llama/Llama-3.1-8B-Instruct tokenizer, NOT len(text.split()), which
   undercounts technical text by 30-40%.
2. Accept/quarantine dataset split (the math-gate's actual output contract) --
   accepted_dataset.jsonl vs quarantine_dataset.jsonl, not one undifferentiated
   JSONL with a status field.

Usage:
    python finalize_module1_dataset.py <source_jsonl> [--max-seq-length 512]

Requires HF_TOKEN in .env with access to the gated meta-llama/Llama-3.1-8B-Instruct
repo (visit the model page on huggingface.co, accept the license, generate a token
with read access, set HF_TOKEN=... in .env).
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

FAITHFULNESS_ACCEPT_THRESHOLD = 0.85
TOKENIZER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("FATAL: transformers not installed. Run: pip install transformers", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "FATAL: HF_TOKEN not set in .env. meta-llama/Llama-3.1-8B-Instruct is a gated "
            "model -- visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, "
            "accept the license, generate a read-access token at "
            "https://huggingface.co/settings/tokens, and add HF_TOKEN=... to .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, token=hf_token)
    except Exception as e:
        print(
            f"FATAL: could not load tokenizer for {TOKENIZER_MODEL}: {e}\n"
            "Common causes: license not yet accepted on the model page, token lacks "
            "read access, or token is invalid/expired.",
            file=sys.stderr,
        )
        sys.exit(1)

    return tokenizer


def audit_record(record: dict, tokenizer, max_seq_length: int) -> dict:
    """
    Real tokenizer-based length audit, per curriculum Section 3a.
    full_text mirrors the curriculum's own audit_record() function exactly:
    instruction + input + output concatenated, tokenized with special tokens.

    Design decision (logged here rather than silently applied, per the
    curriculum's explicit "decide in writing" requirement): records over
    max_seq_length are DISCARDED, not truncated. This project has no
    coherence-preserving split strategy for instruction-tuning triples, and
    truncating mid-answer risks producing a syntactically valid but
    semantically decapitated training example -- worse than dropping it.
    """
    candidate = record.get("candidate", {})
    full_text = (
        candidate.get("instruction", "") + candidate.get("input", "") + candidate.get("output", "")
    )
    token_count = len(tokenizer(full_text, add_special_tokens=True).input_ids)
    action = "KEEP" if token_count <= max_seq_length else "DISCARD"
    return {"token_count": token_count, "audit_action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_jsonl", help="Path to the generate_split_brain_data.py output file")
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    source_path = Path(args.source_jsonl)
    if not source_path.exists():
        print(f"FATAL: {source_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(line) for line in open(source_path, encoding="utf-8")]
    print(f"Loaded {len(records)} records from {source_path}")

    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded: {TOKENIZER_MODEL}")

    accepted = []
    quarantined = []
    token_counts = []
    discard_by_length = 0
    discard_by_faithfulness = 0
    discard_by_pipeline_failure = 0

    for record in records:
        status = record.get("status")

        if status != "SCORED":
            # Generation refusals, dead letters, API errors -- already failed
            # upstream of the math gate. Quarantine with the original reason intact.
            record["quarantine_reason"] = f"pipeline_failure: {status}"
            quarantined.append(record)
            discard_by_pipeline_failure += 1
            continue

        audit = audit_record(record, tokenizer, args.max_seq_length)
        record["token_count"] = audit["token_count"]
        record["audit_action"] = audit["audit_action"]
        token_counts.append(audit["token_count"])

        if audit["audit_action"] == "DISCARD":
            record["quarantine_reason"] = (
                f"token_length: {audit['token_count']} tokens exceeds max_seq_length={args.max_seq_length}"
            )
            quarantined.append(record)
            discard_by_length += 1
            continue

        if record.get("faithfulness", 0.0) < FAITHFULNESS_ACCEPT_THRESHOLD:
            record["quarantine_reason"] = (
                f"faithfulness: {record['faithfulness']:.3f} below {FAITHFULNESS_ACCEPT_THRESHOLD} threshold"
            )
            quarantined.append(record)
            discard_by_faithfulness += 1
            continue

        accepted.append(record)

    accepted_path = Path("accepted_dataset.jsonl")
    quarantine_path = Path("quarantine_dataset.jsonl")

    with open(accepted_path, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")

    with open(quarantine_path, "w", encoding="utf-8") as f:
        for r in quarantined:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print(f"Accepted:    {len(accepted)}")
    print(f"Quarantined: {len(quarantined)}")
    print(f"  - discarded for token length:  {discard_by_length}")
    print(f"  - discarded for low faithfulness: {discard_by_faithfulness}")
    print(f"  - discarded for upstream pipeline failure: {discard_by_pipeline_failure}")

    if token_counts:
        sorted_counts = sorted(token_counts)
        n = len(sorted_counts)
        p50 = sorted_counts[int(n * 0.50)]
        p90 = sorted_counts[min(int(n * 0.90), n - 1)]
        p99 = sorted_counts[min(int(n * 0.99), n - 1)]
        print("\nToken count distribution across all audited records:")
        print(f"  p50: {p50}  p90: {p90}  p99: {p99}  max: {max(sorted_counts)}")
        print(f"  mean: {statistics.mean(sorted_counts):.1f}")

    print(f"\nWritten: {accepted_path.resolve()}")
    print(f"Written: {quarantine_path.resolve()}")


if __name__ == "__main__":
    main()
