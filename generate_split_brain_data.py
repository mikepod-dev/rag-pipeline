"""
generate_split_brain_data.py

Diagnostic run: 5 chunks through the split-brain generation -> clerk pipeline,
to establish a real cost-per-record baseline before committing the full 796-chunk run.

Stage 1 (Generator): deepseek/deepseek-v4-pro reads a chunk, writes a candidate
    {instruction, input, output} triplet.
Stage 2 (Clerk):      openai/gpt-4o-mini, blind to the generator's identity, extracts
    atomic claims from the candidate output and tags each TRUE/FALSE against the
    source chunk. It never sees which model wrote the candidate.

Faithfulness = verified_claims / total_claims, computed in plain Python -- no LLM
ever emits a float score. Zero extracted claims is treated as a CLERK FAILURE
(dead-lettered), never averaged into a score of any kind -- see the corrected-bug
note at the top of compute_faithfulness().

Requires: OPENROUTER_API_KEY in .env, and either a local chunks.jsonl or a running
Qdrant instance with the project's rag_docs_hybrid collection (see load_chunks()).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("FATAL: OPENROUTER_API_KEY not found in environment / .env", file=sys.stderr)
    sys.exit(1)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

GENERATOR_MODEL = "deepseek/deepseek-v4-pro"
CLERK_MODEL = "openai/gpt-4o-mini"
FINE_TUNE_TARGET = "llama-3.1-8b-instruct"  # never allowed as generator OR clerk

N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_PATH = Path(f"synthetic_run_{N_CHUNKS}chunks_{_run_timestamp}.jsonl")
RATE_LIMIT_DELAY_SECONDS = 2.0  # rolling delay at the END of each chunk iteration
MAX_CLERK_RETRIES = 3

# Pricing as of Aug 2026, verified against openrouter.ai/<model> pages.
# PRICES DRIFT -- re-check before running the full 796-chunk batch.
PRICING_PER_MILLION = {
    GENERATOR_MODEL: {"input": 0.5808, "output": 1.742},
    CLERK_MODEL: {"input": 0.15, "output": 0.60},
}

# Split-brain guardrail: fail loudly at import time if config is misconfigured,
# rather than discovering it 700 chunks into a run.
assert GENERATOR_MODEL != CLERK_MODEL, "Generator and clerk cannot be the same model."
assert (
    FINE_TUNE_TARGET not in GENERATOR_MODEL.lower()
), "Generator cannot be the fine-tuning target."
assert FINE_TUNE_TARGET not in CLERK_MODEL.lower(), "Clerk cannot be the fine-tuning target."


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Claim(BaseModel):
    claim_text: str
    verdict: Literal["TRUE", "FALSE"]


class EvaluationSchema(BaseModel):
    source_chunk_id: str
    clerk_model: str
    claims: list[Claim] = Field(min_length=1)

    @field_validator("clerk_model")
    @classmethod
    def block_fine_tune_target(cls, v: str) -> str:
        # Rust's regex engine (pydantic-core) doesn't support look-ahead,
        # so the hard-block is enforced here instead of via Field(pattern=...).
        if FINE_TUNE_TARGET in v.lower():
            raise ValueError(f"clerk_model cannot be the fine-tuning target ({FINE_TUNE_TARGET})")
        return v


class ZeroClaimsError(Exception):
    """Raised when the clerk extracts zero claims. This is a CLERK FAILURE,
    not a faithfulness score -- caller must dead-letter, never coerce to
    any float (not 0.0, not 1.0)."""

    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
        super().__init__(
            f"Zero claims extracted for chunk {chunk_id} -- clerk failure, not a score."
        )


class ClerkParseError(Exception):
    def __init__(self, chunk_id: str, errors, raw: str):
        self.chunk_id = chunk_id
        self.errors = errors
        self.raw = raw
        super().__init__(f"Clerk output failed schema validation for chunk {chunk_id}")


# --------------------------------------------------------------------------
# Chunk loading
# --------------------------------------------------------------------------


def load_chunks(n: int) -> list[dict]:
    """
    Load n chunks for the diagnostic run.

    ASSUMPTION FLAGGED: this project's real chunks live in Qdrant collection
    `rag_docs_hybrid` per the handoff doc, but I don't know the exact payload
    field name your ingestion pipeline used for chunk text (commonly "text" or
    "content" or "page_content"). Tries a local chunks.jsonl first (safest,
    zero risk of misreading Qdrant payload keys); falls back to a Qdrant scroll
    if that file doesn't exist. If the Qdrant fallback fires, PRINT the raw
    payload of the first point before trusting the extracted text -- verify the
    key name is right before this burns real API budget.
    """
    local_path = Path("chunks.jsonl")
    if local_path.exists():
        chunks = []
        with open(local_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                record = json.loads(line)
                chunks.append(
                    {
                        "chunk_id": record.get("chunk_id", f"local-{i}"),
                        "text": record["text"],
                    }
                )
        print(f"Loaded {len(chunks)} chunks from {local_path}")
        return chunks

    print(f"No {local_path} found -- falling back to Qdrant scroll.")
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print(
            "FATAL: qdrant-client not installed and no chunks.jsonl present.",
            file=sys.stderr,
        )
        sys.exit(1)

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    points, _ = client.scroll(collection_name="rag_docs_hybrid", limit=n, with_payload=True)

    if not points:
        print("FATAL: Qdrant scroll returned zero points.", file=sys.stderr)
        sys.exit(1)

    print("--- RAW PAYLOAD OF FIRST POINT (verify text field name before proceeding) ---")
    print(json.dumps(points[0].payload, indent=2)[:1000])
    print("--- END RAW PAYLOAD ---")

    text_key = None
    for candidate_key in ("text", "content", "page_content", "chunk_text"):
        if candidate_key in points[0].payload:
            text_key = candidate_key
            break
    if text_key is None:
        print(
            "FATAL: none of ['text','content','page_content','chunk_text'] found in payload. "
            "Inspect the printed payload above and hardcode the correct key.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using payload key '{text_key}' for chunk text.")
    return [{"chunk_id": str(p.id), "text": p.payload[text_key]} for p in points]


# --------------------------------------------------------------------------
# OpenRouter call + cost tracking
# --------------------------------------------------------------------------


def call_openrouter(
    model: str,
    messages: list[dict],
    response_format: dict | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0}
    if response_format:
        payload["response_format"] = response_format
    if extra:
        payload.update(extra)

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def compute_cost(model: str, usage: dict) -> float:
    prices = PRICING_PER_MILLION[model]
    input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * prices["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * prices["output"]
    return input_cost + output_cost


def reasoning_tokens_of(usage: dict) -> int:
    """Surfaces the hidden-reasoning-token count so we stop guessing about
    where completion tokens went on reasoning models like deepseek-v4-pro."""
    return usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)


# --------------------------------------------------------------------------
# Stage 1: Generator
# --------------------------------------------------------------------------


def generate_candidate(chunk_text: str, chunk_id: str) -> tuple[dict | None, dict, float]:
    """Returns (candidate_or_None, usage, cost). candidate is None on refusal/empty output."""
    system_prompt = (
        "You are generating a single instruction-tuning training example from a source "
        "document chunk. Output ONLY valid JSON with keys 'instruction', 'input', 'output'. "
        "The 'output' must be fully grounded in the provided chunk -- do not add outside "
        "knowledge. No markdown fences, no commentary, JSON only."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Source chunk:\n\n{chunk_text}"},
    ]
    response = call_openrouter(GENERATOR_MODEL, messages, extra={"reasoning": {"effort": "none"}})
    usage = response.get("usage", {})
    cost = compute_cost(GENERATOR_MODEL, usage)

    raw_content = response["choices"][0]["message"]["content"]
    if raw_content is None:
        # OpenRouter can return null content (e.g. certain content-filter or
        # finish_reason cases) instead of an empty string -- treat identically
        # to a refusal rather than crashing on .strip().
        print(
            f"  [GEN] chunk {chunk_id}: API returned null content -- treating as refusal/failure."
        )
        return None, usage, cost
    raw = raw_content.strip()
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError:
        print(
            f"  [GEN] chunk {chunk_id}: generator output was not valid JSON -- treating as refusal/failure."
        )
        return None, usage, cost

    output_text = candidate.get("output", "").strip()
    if (
        not output_text
        or "cannot" in output_text.lower()[:40]
        or "unable to" in output_text.lower()[:40]
    ):
        print(f"  [GEN] chunk {chunk_id}: empty or refusal-shaped output -- flagging, not scoring.")
        return None, usage, cost

    return candidate, usage, cost


# --------------------------------------------------------------------------
# Stage 2: Clerk (blind auditor)
# --------------------------------------------------------------------------


def clerk_audit(
    chunk_text: str, candidate_output: str, chunk_id: str
) -> tuple[EvaluationSchema | None, dict, float, str | None]:
    """
    Returns (EvaluationSchema_or_None, usage, cost, failure_reason).
    The clerk is NEVER told which model produced candidate_output.
    """
    system_prompt = (
        "You are a blind fact-checking clerk. You will be given a SOURCE TEXT and a "
        "GENERATED ANSWER. Extract every atomic factual claim made in the GENERATED ANSWER "
        "and mark each TRUE if it is directly supported by SOURCE TEXT, or FALSE if it is "
        "not supported (including any claim that adds information not present in SOURCE TEXT). "
        "Do not infer intent or grade style -- mechanical extraction and verification only. "
        "Respond ONLY with JSON matching this shape: "
        '{"source_chunk_id": "...", "clerk_model": "openai/gpt-4o-mini", '
        '"claims": [{"claim_text": "...", "verdict": "TRUE"}, ...]}. '
        "No markdown fences, no commentary."
    )

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    cost_total = 0.0
    last_error = None

    for attempt in range(MAX_CLERK_RETRIES):
        user_content = (
            f"SOURCE TEXT:\n{chunk_text}\n\nGENERATED ANSWER:\n{candidate_output}\n\n"
            f"source_chunk_id to use: {chunk_id}"
        )
        if last_error:
            user_content += (
                f"\n\nPREVIOUS ATTEMPT FAILED SCHEMA VALIDATION: {last_error}. "
                "Return ONLY valid JSON matching the schema, no markdown fences, no commentary."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = call_openrouter(CLERK_MODEL, messages, response_format={"type": "json_object"})
        usage = response.get("usage", {})
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
        cost_total += compute_cost(CLERK_MODEL, usage)

        raw_content = response["choices"][0]["message"]["content"]
        if raw_content is None:
            # Same null-content edge case as the generator -- treat as a schema-validation
            # failure so it goes through the existing retry path instead of crashing.
            last_error = "API returned null content for clerk response"
            print(
                f"  [CLERK] chunk {chunk_id}: attempt {attempt + 1}/{MAX_CLERK_RETRIES} got null content."
            )
            continue
        raw = raw_content.strip()
        try:
            evaluation = EvaluationSchema.model_validate_json(raw)
            return evaluation, usage_total, cost_total, None
        except ValidationError as e:
            last_error = e.errors()
            print(
                f"  [CLERK] chunk {chunk_id}: attempt {attempt + 1}/{MAX_CLERK_RETRIES} failed schema validation."
            )

    return None, usage_total, cost_total, f"schema_validation_exhausted: {last_error}"


# --------------------------------------------------------------------------
# Deterministic math gate
# --------------------------------------------------------------------------


def compute_faithfulness(evaluation: EvaluationSchema) -> float:
    """
    Plain Python division, no LLM involved.

    CORRECTED FROM SPEC: zero claims is NEVER coerced to a score (not 1.0, not
    0.0). It means the clerk failed to produce usable output -- that's a
    pipeline defect requiring investigation, per the curriculum's own stated
    rule. Raises ZeroClaimsError so the caller dead-letters the record instead
    of writing a fabricated score into accepted_dataset.jsonl.
    """
    claims = evaluation.claims
    if not claims:
        raise ZeroClaimsError(evaluation.source_chunk_id)
    true_count = sum(1 for c in claims if c.verdict == "TRUE")
    return true_count / len(claims)


# --------------------------------------------------------------------------
# Main diagnostic loop
# --------------------------------------------------------------------------


def main():
    chunks = load_chunks(N_CHUNKS)
    total_cost = 0.0
    records_written = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]
            print(f"\n=== Chunk {i + 1}/{len(chunks)} (id={chunk_id}) ===")
            if (i + 1) % 50 == 0:
                print(
                    f"--- progress: {i + 1}/{len(chunks)}, running cost so far ${total_cost:.4f} ---"
                )

            record = {
                "chunk_id": chunk_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generator_model": GENERATOR_MODEL,
                "clerk_model": CLERK_MODEL,
            }

            try:
                # --- Stage 1: Generation ---
                candidate, gen_usage, gen_cost = generate_candidate(chunk_text, chunk_id)
                total_cost += gen_cost
                gen_reasoning_tokens = reasoning_tokens_of(gen_usage)
                record["generation_reasoning_tokens"] = gen_reasoning_tokens
                print(
                    f"  [GEN] tokens in={gen_usage.get('prompt_tokens', 0)} "
                    f"out={gen_usage.get('completion_tokens', 0)} "
                    f"(reasoning={gen_reasoning_tokens}) cost=${gen_cost:.6f}"
                )

                if candidate is None:
                    record["status"] = "GENERATION_REFUSAL_OR_EMPTY"
                    record["generation_cost"] = gen_cost
                    record["faithfulness"] = None
                    print(
                        f"  -> Chunk {chunk_id}: generation refusal/empty. Logged, not scored, clerk skipped."
                    )
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    continue

                record["candidate"] = candidate

                # --- Stage 2: Clerk audit (blind) ---
                evaluation, clerk_usage, clerk_cost, failure_reason = clerk_audit(
                    chunk_text, candidate["output"], chunk_id
                )
                total_cost += clerk_cost
                print(
                    f"  [CLERK] tokens in={clerk_usage.get('prompt_tokens', 0)} "
                    f"out={clerk_usage.get('completion_tokens', 0)} cost=${clerk_cost:.6f}"
                )

                record["generation_cost"] = gen_cost
                record["clerk_cost"] = clerk_cost
                record["chunk_total_cost"] = gen_cost + clerk_cost

                if evaluation is None:
                    record["status"] = "DEAD_LETTER"
                    record["failure_reason"] = failure_reason
                    record["faithfulness"] = None
                    print(f"  -> Chunk {chunk_id}: DEAD-LETTERED ({failure_reason})")
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    continue

                # --- Stage 3: Deterministic math gate ---
                try:
                    faithfulness = compute_faithfulness(evaluation)
                    record["status"] = "SCORED"
                    record["claims"] = [c.model_dump() for c in evaluation.claims]
                    record["faithfulness"] = faithfulness
                    print(
                        f"  -> Chunk {chunk_id}: faithfulness={faithfulness:.3f} "
                        f"({sum(1 for c in evaluation.claims if c.verdict == 'TRUE')}/{len(evaluation.claims)} claims true)"
                    )
                except ZeroClaimsError:
                    record["status"] = "DEAD_LETTER"
                    record["failure_reason"] = "zero_claims_extracted"
                    record["faithfulness"] = None
                    print(
                        f"  -> Chunk {chunk_id}: DEAD-LETTERED (zero claims extracted -- clerk failure, not scored)"
                    )

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                records_written += 1

            except requests.HTTPError as e:
                print(f"  !! HTTP error on chunk {chunk_id}: {e}", file=sys.stderr)
                record["status"] = "API_ERROR"
                record["error"] = str(e)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"  !! Unexpected error on chunk {chunk_id}: {e}", file=sys.stderr)
                record["status"] = "UNEXPECTED_ERROR"
                record["error"] = str(e)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            # Rolling rate-limit delay at the END of each chunk iteration
            if i < len(chunks) - 1:
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

    print("\n" + "=" * 60)
    print(f"Diagnostic run complete. {records_written}/{len(chunks)} chunks fully scored.")
    print(f"Total cost for {len(chunks)} chunks: ${total_cost:.6f}")
    if len(chunks) > 0:
        per_chunk = total_cost / len(chunks)
        print(f"Cost per chunk: ${per_chunk:.6f}")
        print(f"Projected cost for full 796 chunks: ${per_chunk * 796:.4f}")
        print(
            f"Module 1 cap is $2.00 -- {'WITHIN' if per_chunk * 796 <= 2.0 else 'EXCEEDS'} budget at this rate."
        )
    print(f"Output written to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
