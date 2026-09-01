"""
ragas_gate.py

Module 3: Automated failure catcher for production RAG answers.

Evaluates real logged queries from query_log.jsonl (question, answer, cost, timestamp --
the only fields reliably present across 99.4% of real entries; retrieval_count/success/
latency_ms exist on only 4 of 657 real records and are not relied upon here).

No context is stored in the log at all -- every record's context is backfilled via a
fresh call to pipeline.hybrid_search_with_rerank(question), the same real retrieval
function pipeline.py's live chat path uses.

Faithfulness scoring follows this project's established, evidence-backed pattern from
Module 1/2: a cheap judge model (~google/gemini-flash-latest) extracts discrete TRUE/FALSE
claims, and faithfulness is computed by plain Python division -- never an LLM-emitted
float, per the real, measured non-determinism problems with continuous scores documented
in this project's own prior findings.

Records failing the gate (faithfulness < 0.7) are escalated to a teacher model
(~anthropic/claude-sonnet-latest) for a corrected answer + explanation. Corrections are
NOT written directly to a golden dataset -- they go to a versioned pending_review_v{n}.jsonl
staging file, requiring explicit human promotion via review_pending_corrections.py before
anything is trusted. This exists because an early run of this gate produced a confirmed
false "correction" that deleted true information (a real retrieval-width failure, documented
as its own finding) with nothing in the automated pipeline catching it.

A single circuit breaker can trip on EITHER condition (matching both framings in the
curriculum spec): batch failure rate exceeding --failure-rate-threshold (default 30%,
checked only after a minimum sample size to avoid tripping on early noise), or cumulative
cost exceeding --cost-threshold. Can be disabled via --disable-circuit-breaker specifically
to demonstrate what happens without it, on a small batch with a deliberately low mock
threshold -- never against the project's real remaining budget.

10 canary records (real questions from the sample, paired with a DIFFERENT record's real
answer -- a genuine mismatch, not fabricated content) are mixed into the batch to prove
the gate actually catches bad answers rather than passing everything.

Usage:
    python ragas_gate.py --limit 25 --cost-threshold 1.00
    python ragas_gate.py --limit 10 --disable-circuit-breaker --cost-threshold 0.02  # demo
"""

import argparse
import json
import os
import random
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
JUDGE_MODEL = (
    "~google/gemini-flash-latest"  # tilde prefix required for "latest"-alias slugs on OpenRouter
)
TEACHER_MODEL = "~anthropic/claude-sonnet-latest"  # always-current alias, avoids hardcoding a version that gets deprecated
FAITHFULNESS_THRESHOLD = 0.7
SEED = 3407
N_CANARIES = 10
MAX_RETRIES = 3
RATE_LIMIT_DELAY_SECONDS = 1.0


class Claim(BaseModel):
    claim_text: str
    verdict: Literal["TRUE", "FALSE"]


class EvaluationSchema(BaseModel):
    record_id: str
    claims: list[Claim] = Field(min_length=1)


def call_openrouter(model: str, messages: list[dict]) -> dict:
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "temperature": 0},
        timeout=90,
    )
    if resp.status_code != 200:
        # raise_for_status() alone swallows the actual error body -- print it first so a bad
        # model slug or malformed request is diagnosable instead of a bare HTTPError.
        print(
            f"  !! OpenRouter {resp.status_code} for model '{model}': {resp.text[:500]}",
            file=sys.stderr,
        )
    resp.raise_for_status()
    return resp.json()


def compute_cost(usage: dict, prices: dict) -> float:
    input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * prices["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * prices["output"]
    return input_cost + output_cost


# Prices as of Aug 2026 -- verify before scaling to a much larger batch.
JUDGE_PRICES = {"input": 0.075, "output": 0.30}
TEACHER_PRICES = {
    "input": 2.00,
    "output": 10.00,
}  # ~anthropic/claude-sonnet-latest, verified Aug 2026


# Scoping decision, disclosed rather than silently applied: this gate assumes every logged
# entry is a standard document-QA query that needs RAG context backfilled via
# hybrid_search_with_rerank(). A real, confirmed exception exists in the log: 223/657 (34%)
# of entries are a structurally different task -- a relationship-extraction template with
# its own source text embedded directly in the prompt, needing no retrieval at all. Naively
# backfilling context for these produced a confirmed real failure (a correct extraction
# "corrected" into an empty list, because the backfilled context was irrelevant to the
# self-contained task). Excluding this category explicitly rather than mishandling it.
NON_QA_MARKERS = ["Extract factual relationships", "Reply with ONLY a JSON"]


def is_qa_style(question: str) -> bool:
    return not any(marker in question for marker in NON_QA_MARKERS)


def load_log_sample(path: Path, limit: int, seed: int) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f]
    qa_entries = [e for e in all_entries if is_qa_style(e["question"])]
    excluded = len(all_entries) - len(qa_entries)
    if excluded:
        print(f"Excluded {excluded} non-QA-style (task-format) entries from the sampling pool.")
    random.seed(seed)
    return random.sample(qa_entries, min(limit, len(qa_entries)))


def build_canaries(sample: list[dict], n: int, seed: int) -> list[dict]:
    """
    Canary = a real question paired with a DIFFERENT record's real answer -- a genuine,
    verifiable mismatch, not fabricated content. Guarantees the gate has something it
    MUST catch.
    """
    random.seed(seed + 1)
    n = min(n, len(sample))
    picks = random.sample(sample, n)
    shuffled_answers = [r["answer"] for r in picks]
    random.shuffle(shuffled_answers)
    for i in range(len(picks)):
        if shuffled_answers[i] == picks[i]["answer"]:
            shuffled_answers[i], shuffled_answers[(i + 1) % len(picks)] = (
                shuffled_answers[(i + 1) % len(picks)],
                shuffled_answers[i],
            )
    canaries = []
    for record, mismatched_answer in zip(picks, shuffled_answers):
        canaries.append(
            {
                "question": record["question"],
                "answer": mismatched_answer,
                "timestamp": record.get("timestamp"),
                "cost": record.get("cost", 0.0),
                "is_canary": True,
            }
        )
    return canaries


def backfill_context(question: str, n_final: int = 5) -> list[str]:
    """
    n_final=5, not pipeline.py's own default of 2. A real, confirmed failure
    (documented as its own finding) showed the default width missed a genuinely
    relevant chunk, causing the judge to mark a true claim FALSE and the teacher
    to delete correct content. Widening reduces -- does not eliminate -- that risk;
    some retrieval width will always be able to miss something.
    """
    from pipeline import hybrid_search_with_rerank

    result = hybrid_search_with_rerank(question, n_final=n_final)
    return result["documents"][0]


def judge_faithfulness(question: str, answer: str, context_chunks: list[str], record_id: str):
    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a blind fact-checking clerk. Given RETRIEVED CONTEXT and a GENERATED "
        "ANSWER to a QUESTION, extract every atomic factual claim in the GENERATED ANSWER "
        "and mark each TRUE if directly supported by the RETRIEVED CONTEXT, or FALSE if "
        "not supported. Respond ONLY with JSON: "
        '{"record_id": "...", "claims": [{"claim_text": "...", "verdict": "TRUE"}, ...]}. '
        "No markdown fences, no commentary."
    )
    last_error = None
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(MAX_RETRIES):
        user_content = (
            f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{context_text}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\nrecord_id to use: {record_id}"
        )
        if last_error:
            user_content += (
                f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {last_error}. Return ONLY valid JSON."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = call_openrouter(JUDGE_MODEL, messages)
        usage = response.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        raw = response["choices"][0]["message"]["content"]
        if raw is None:
            last_error = "null content"
            continue
        try:
            evaluation = EvaluationSchema.model_validate_json(raw.strip())
            cost = compute_cost(total_usage, JUDGE_PRICES)
            return evaluation, cost, None
        except ValidationError as e:
            last_error = str(e.errors())
    cost = compute_cost(total_usage, JUDGE_PRICES)
    return None, cost, f"schema_validation_exhausted: {last_error}"


def escalate_to_teacher(question: str, context_chunks: list[str], bad_answer: str):
    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a teacher model correcting a flawed RAG answer. Given the question, the "
        "retrieved context, and a flawed answer, provide a corrected answer grounded ONLY "
        "in the retrieved context, plus a one-sentence explanation of what was wrong with "
        "the original. Respond ONLY with JSON: "
        '{"corrected_answer": "...", "explanation": "..."}. No markdown fences.'
    )
    user_content = f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{context_text}\n\nFLAWED ANSWER:\n{bad_answer}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = call_openrouter(TEACHER_MODEL, messages)
    usage = response.get("usage", {})
    cost = compute_cost(usage, TEACHER_PRICES)
    raw = response["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw.strip())
        return parsed.get("corrected_answer"), parsed.get("explanation"), cost, None
    except (json.JSONDecodeError, AttributeError) as e:
        return None, None, cost, f"teacher_parse_failure: {e}"


def next_pending_review_version() -> Path:
    n = 1
    while Path(f"pending_review_v{n}.jsonl").exists():
        n += 1
    return Path(f"pending_review_v{n}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--source", default="query_log.jsonl")
    parser.add_argument("--cost-threshold", type=float, default=1.00)
    parser.add_argument("--failure-rate-threshold", type=float, default=0.30)
    parser.add_argument("--min-sample-before-breaker", type=int, default=5)
    parser.add_argument("--disable-circuit-breaker", action="store_true")
    args = parser.parse_args()

    from pipeline import initialize

    print("Initializing pipeline (embeddings, vector DB connection)...")
    initialize()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"FATAL: {source_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    sample = load_log_sample(source_path, args.limit, SEED)
    canaries = build_canaries(sample, N_CANARIES, SEED)
    batch = sample + canaries
    random.seed(SEED + 2)
    random.shuffle(batch)

    print(f"Real records: {len(sample)}, canaries: {len(canaries)}, total batch: {len(batch)}")
    print(
        f"Circuit breaker: {'DISABLED (demo mode)' if args.disable_circuit_breaker else 'enabled'}"
    )
    print(
        f"  cost threshold: ${args.cost_threshold:.4f}, failure-rate threshold: {args.failure_rate_threshold:.0%}"
    )

    results = []
    total_gate_cost = 0.0
    total_escalation_cost = 0.0
    processed = 0
    failed = 0
    breaker_tripped = False
    breaker_reason = None

    pending_path = next_pending_review_version()
    run_log_path = Path("gate_run_results.jsonl")

    with (
        open(run_log_path, "w", encoding="utf-8") as run_log_f,
        open(pending_path, "w", encoding="utf-8") as pending_f,
    ):
        for i, record in enumerate(batch):
            record_id = f"record-{i}"
            question = record["question"]
            answer = record["answer"]
            is_canary = record.get("is_canary", False)

            print(f"\n=== {i + 1}/{len(batch)} {'[CANARY]' if is_canary else ''} ===")

            try:
                context_chunks = backfill_context(question)
            except Exception as e:
                print(f"  !! context backfill failed: {e}")
                continue

            evaluation, judge_cost, error = judge_faithfulness(
                question, answer, context_chunks, record_id
            )
            total_gate_cost += judge_cost
            processed += 1

            entry = {
                "record_id": record_id,
                "question": question,
                "answer": answer,
                "is_canary": is_canary,
                "judge_cost": judge_cost,
            }

            if evaluation is None:
                entry["status"] = "DEAD_LETTER"
                entry["error"] = error
                failed += 1
            else:
                true_count = sum(1 for c in evaluation.claims if c.verdict == "TRUE")
                faithfulness = true_count / len(evaluation.claims)
                entry["faithfulness"] = faithfulness
                entry["claims"] = [c.model_dump() for c in evaluation.claims]

                if faithfulness < FAITHFULNESS_THRESHOLD:
                    failed += 1
                    print(f"  GATE FAIL (faithfulness={faithfulness:.3f}) -- escalating to teacher")
                    corrected, explanation, teacher_cost, teacher_error = escalate_to_teacher(
                        question, context_chunks, answer
                    )
                    total_escalation_cost += teacher_cost
                    entry["teacher_cost"] = teacher_cost
                    if corrected is None:
                        entry["status"] = "ESCALATION_FAILED"
                        entry["teacher_error"] = teacher_error
                    else:
                        entry["status"] = "CORRECTED"
                        entry["corrected_answer"] = corrected
                        entry["explanation"] = explanation
                        pending_record = {
                            "question": question,
                            "original_answer": answer,
                            "corrected_answer": corrected,
                            "explanation": explanation,
                            "original_faithfulness": faithfulness,
                            "context_used": context_chunks,
                            "judge_model": JUDGE_MODEL,
                            "teacher_model": TEACHER_MODEL,
                            "correction_pass": 1,
                            "is_canary": is_canary,
                            "timestamp": record.get("timestamp"),
                            "review_status": "PENDING",
                        }
                        pending_f.write(json.dumps(pending_record) + "\n")
                        pending_f.flush()
                else:
                    entry["status"] = "PASSED"
                    print(f"  PASSED (faithfulness={faithfulness:.3f})")

            results.append(entry)
            run_log_f.write(json.dumps(entry) + "\n")
            run_log_f.flush()

            total_cost = total_gate_cost + total_escalation_cost
            if not args.disable_circuit_breaker:
                failure_rate = failed / processed if processed else 0.0
                if (
                    processed >= args.min_sample_before_breaker
                    and failure_rate > args.failure_rate_threshold
                ):
                    breaker_tripped = True
                    breaker_reason = f"failure_rate {failure_rate:.1%} exceeded {args.failure_rate_threshold:.0%} after {processed} records"
                    break
                if total_cost > args.cost_threshold:
                    breaker_tripped = True
                    breaker_reason = f"cost ${total_cost:.4f} exceeded ${args.cost_threshold:.4f} after {processed} records"
                    break
            else:
                if total_cost > args.cost_threshold:
                    print(
                        f"  !! cost ${total_cost:.4f} has exceeded threshold ${args.cost_threshold:.4f} -- circuit breaker is DISABLED, continuing anyway (demo mode)"
                    )

            time.sleep(RATE_LIMIT_DELAY_SECONDS)

    print("\n" + "=" * 70)
    if breaker_tripped:
        print(f"CIRCUIT BREAKER TRIPPED: {breaker_reason}")
    print(f"Processed: {processed}/{len(batch)}")
    print(f"Failed gate: {failed}")
    print(f"Gate (judge) cost: ${total_gate_cost:.4f}")
    print(f"Escalation (teacher) cost: ${total_escalation_cost:.4f}")
    print(f"Total cost: ${total_gate_cost + total_escalation_cost:.4f}")

    canary_results = [r for r in results if r.get("is_canary")]
    canary_caught = sum(
        1 for r in canary_results if r.get("status") in ("CORRECTED", "ESCALATION_FAILED")
    )
    print(f"\nCanary catch rate: {canary_caught}/{len(canary_results)}")

    print(f"\nRun log: {run_log_path.resolve()}")
    print(f"Pending review: {pending_path.resolve()}")
    print(
        "Nothing has been written to a golden dataset yet -- run review_pending_corrections.py to promote records."
    )


if __name__ == "__main__":
    main()
