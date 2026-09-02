"""
compare_raft_vs_baseline.py

Compares Module 1 baseline (checkpoint-22, run2_masked) against the RAFT-trained
adapter (checkpoint-44, raft_run2_abstain5pct) on the same 76 held-out RAFT-formatted
prompts, using the same split-brain clerk methodology from Module 1 -- an independent
model (gpt-4o-mini) verifying discrete TRUE/FALSE claims against the real target,
with faithfulness computed by plain Python division, never an LLM-emitted score.

Deterministic checks happen first (did it abstain vs not, matching what was expected)
-- the clerk is only called when there's real generated content to verify against the
real target, keeping LLM judgment scoped to what actually needs it.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("FATAL: OPENROUTER_API_KEY not found in environment / .env", file=sys.stderr)
    sys.exit(1)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLERK_MODEL = "openai/gpt-4o-mini"
ABSTAIN_PHRASE = "does not contain this information"
MAX_RETRIES = 3
RATE_LIMIT_DELAY_SECONDS = 1.0


class Claim(BaseModel):
    claim_text: str
    verdict: Literal["TRUE", "FALSE"]


class EvaluationSchema(BaseModel):
    record_id: str
    claims: list[Claim] = Field(min_length=1)


def call_openrouter(messages: list[dict]) -> dict:
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": CLERK_MODEL, "messages": messages, "temperature": 0},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def clerk_verify(
    generated_text: str, real_answer: str, record_id: str
) -> tuple[EvaluationSchema | None, str | None]:
    system_prompt = (
        "You are a blind fact-checking clerk. You will be given a REFERENCE ANSWER "
        "(known correct) and a GENERATED ANSWER. Extract every atomic factual claim "
        "made in the GENERATED ANSWER and mark each TRUE if it is directly supported "
        "by the REFERENCE ANSWER, or FALSE if it is not supported or contradicts it. "
        "Respond ONLY with JSON: "
        '{"record_id": "...", "claims": [{"claim_text": "...", "verdict": "TRUE"}, ...]}. '
        "No markdown fences, no commentary."
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        user_content = (
            f"REFERENCE ANSWER:\n{real_answer}\n\nGENERATED ANSWER:\n{generated_text}\n\n"
            f"record_id to use: {record_id}"
        )
        if last_error:
            user_content += (
                f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {last_error}. Return ONLY valid JSON."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = call_openrouter(messages)
        raw = response["choices"][0]["message"]["content"]
        if raw is None:
            last_error = "null content"
            continue
        try:
            return EvaluationSchema.model_validate_json(raw.strip()), None
        except ValidationError as e:
            last_error = str(e.errors())
    return None, f"schema_validation_exhausted: {last_error}"


def score_generation(
    generated_text: str, real_answer: str, is_abstain: bool, record_id: str
) -> dict:
    generated_abstained = ABSTAIN_PHRASE.lower() in generated_text.lower()

    if is_abstain and generated_abstained:
        return {"status": "CORRECT_ABSTAIN", "faithfulness": 1.0}

    if not is_abstain and generated_abstained:
        return {"status": "FALSE_ABSTAIN", "faithfulness": 0.0}

    evaluation, error = clerk_verify(generated_text, real_answer, record_id)
    if evaluation is None:
        return {"status": "DEAD_LETTER", "faithfulness": None, "error": error}

    true_count = sum(1 for c in evaluation.claims if c.verdict == "TRUE")
    faithfulness = true_count / len(evaluation.claims)
    status = "HALLUCINATED_ON_ABSTAIN" if is_abstain else "SCORED"
    return {
        "status": status,
        "faithfulness": faithfulness,
        "claims": [c.model_dump() for c in evaluation.claims],
    }


def main():
    baseline_path = Path("baseline_generations.json")
    raft_path = Path("raft_generations.json")
    for p in (baseline_path, raft_path):
        if not p.exists():
            print(f"FATAL: {p} does not exist.", file=sys.stderr)
            sys.exit(1)

    with open(baseline_path, encoding="utf-8") as f:
        baseline_records = {r["chunk_id"]: r for r in json.load(f)}
    with open(raft_path, encoding="utf-8") as f:
        raft_records = {r["chunk_id"]: r for r in json.load(f)}

    common_ids = set(baseline_records) & set(raft_records)
    print(
        f"Baseline: {len(baseline_records)} records, RAFT: {len(raft_records)} records, common: {len(common_ids)}"
    )
    if len(common_ids) != len(baseline_records) or len(common_ids) != len(raft_records):
        print(
            "WARNING: record sets don't fully match -- comparison will only use the common subset."
        )

    results = []
    for i, chunk_id in enumerate(sorted(common_ids)):
        if (i + 1) % 20 == 0:
            print(f"--- progress: {i + 1}/{len(common_ids)} ---")

        b = baseline_records[chunk_id]
        r = raft_records[chunk_id]
        real_answer = b["real_answer"]
        is_abstain = b["is_abstain"]

        baseline_score = score_generation(
            b["baseline_generated"], real_answer, is_abstain, f"{chunk_id}-baseline"
        )
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        raft_score = score_generation(
            r["raft_generated"], real_answer, is_abstain, f"{chunk_id}-raft"
        )
        time.sleep(RATE_LIMIT_DELAY_SECONDS)

        results.append(
            {
                "chunk_id": chunk_id,
                "query": b["query"],
                "is_abstain": is_abstain,
                "real_answer": real_answer,
                "baseline_generated": b["baseline_generated"],
                "raft_generated": r["raft_generated"],
                "baseline_score": baseline_score,
                "raft_score": raft_score,
            }
        )

    with open("comparison_results.jsonl", "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec) + "\n")

    non_abstain = [r for r in results if not r["is_abstain"]]
    abstain = [r for r in results if r["is_abstain"]]

    print("\n" + "=" * 70)
    print(f"NON-ABSTAIN records: {len(non_abstain)}")
    baseline_scored = [
        r["baseline_score"]["faithfulness"]
        for r in non_abstain
        if r["baseline_score"]["faithfulness"] is not None
    ]
    raft_scored = [
        r["raft_score"]["faithfulness"]
        for r in non_abstain
        if r["raft_score"]["faithfulness"] is not None
    ]
    if baseline_scored:
        print(
            f"  Baseline mean faithfulness: {sum(baseline_scored)/len(baseline_scored):.3f} (n={len(baseline_scored)})"
        )
    if raft_scored:
        print(
            f"  RAFT mean faithfulness:     {sum(raft_scored)/len(raft_scored):.3f} (n={len(raft_scored)})"
        )

    baseline_false_abstain = sum(
        1 for r in non_abstain if r["baseline_score"]["status"] == "FALSE_ABSTAIN"
    )
    raft_false_abstain = sum(1 for r in non_abstain if r["raft_score"]["status"] == "FALSE_ABSTAIN")
    print(
        f"  Baseline false-abstain: {baseline_false_abstain}/{len(non_abstain)} ({100*baseline_false_abstain/len(non_abstain):.1f}%)"
    )
    print(
        f"  RAFT false-abstain:     {raft_false_abstain}/{len(non_abstain)} ({100*raft_false_abstain/len(non_abstain):.1f}%)"
    )

    raft_better = sum(
        1
        for r in non_abstain
        if r["raft_score"]["faithfulness"] is not None
        and r["baseline_score"]["faithfulness"] is not None
        and r["raft_score"]["faithfulness"] > r["baseline_score"]["faithfulness"]
    )
    baseline_better = sum(
        1
        for r in non_abstain
        if r["raft_score"]["faithfulness"] is not None
        and r["baseline_score"]["faithfulness"] is not None
        and r["baseline_score"]["faithfulness"] > r["raft_score"]["faithfulness"]
    )
    tied = sum(
        1
        for r in non_abstain
        if r["raft_score"]["faithfulness"] is not None
        and r["baseline_score"]["faithfulness"] is not None
        and r["baseline_score"]["faithfulness"] == r["raft_score"]["faithfulness"]
    )
    print(
        f"  Paired comparison: RAFT better on {raft_better}, baseline better on {baseline_better}, tied on {tied}"
    )

    print(f"\nABSTAIN records: {len(abstain)}")
    baseline_correct_abstain = sum(
        1 for r in abstain if r["baseline_score"]["status"] == "CORRECT_ABSTAIN"
    )
    raft_correct_abstain = sum(1 for r in abstain if r["raft_score"]["status"] == "CORRECT_ABSTAIN")
    print(f"  Baseline correctly abstained: {baseline_correct_abstain}/{len(abstain)}")
    print(f"  RAFT correctly abstained:     {raft_correct_abstain}/{len(abstain)}")

    print(f"\nWritten: {Path('comparison_results.jsonl').resolve()}")


if __name__ == "__main__":
    main()
