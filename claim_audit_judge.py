"""
claim_audit_judge.py

Implements the "Decompose, Judge, Score" evaluation pattern:

  1. PROBABILISTIC LAYER (the LLM): extracts individual factual claims from
     the generated answer and tags each one True/False for whether it's
     grounded in the retrieved context. This is the ONLY thing the model is
     trusted to do - a discrete, auditable, per-claim judgment call, not
     arithmetic and not a continuous 0.0-1.0 "vibe" score.

  2. DETERMINISTIC LAYER (plain Python): counts the True/False tags and
     computes the final score with ordinary division. No LLM ever touches
     this step - it's 100% reproducible, auditable, and immune to the
     threshold-boundary fragility found in Finding 17 (a continuous score
     landing ambiguously close to a pass/fail cutoff).
"""

import json
import os

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")

# Reusing the exact model string already proven working elsewhere in this
# project (ask_llm, custom_judge.py) via OpenRouter. A previous attempt to
# use 'anthropic/claude-3.5-haiku' directly returned a 404 from OpenRouter
# (Finding 15) - this string is not a guess.
JUDGE_MODEL = "~anthropic/claude-haiku-latest"


# --- Pydantic v2 schemas: define the SHAPE the model must tag data into. ---
# These are strict data-entry forms, not free-form reasoning. The model's
# only job is to fill them in correctly - it never sees or produces a score.


class ClaimAudit(BaseModel):
    """One individual factual claim extracted from the answer, and whether
    it's actually supported by the retrieved context. This is a discrete,
    auditable tag - not a probability, not a confidence level."""

    claim_text: str = Field(
        description="The individual factual claim extracted verbatim (or near-verbatim) from the answer."
    )
    is_grounded: bool = Field(
        description="True if this specific claim is explicitly stated or directly implied by the retrieved context. False if it is unsupported (a hallucination)."
    )


class EvaluationSchema(BaseModel):
    """The full structured output the judge model must produce. The model
    reasons in `internal_rationale` first (forcing it to actually think
    before tagging), then commits to a discrete claims_matrix. No score
    field exists here on purpose - scoring is not the model's job."""

    internal_rationale: str = Field(
        description="Step-by-step reasoning: what claims exist in the answer, and for each, whether the retrieved context actually supports it. Think this through before filling in claims_matrix."
    )
    claims_matrix: list[ClaimAudit] = Field(
        description="One entry per distinct factual claim in the answer. Empty list if the answer makes no checkable factual claims (e.g. a pure refusal)."
    )


JUDGE_SYSTEM_PROMPT = """You are a meticulous fact-auditing data entry clerk, not a scorer. Your only job is to:

1. Read the QUESTION, the RETRIEVED CONTEXT, and the ANSWER.
2. In `internal_rationale`, think step by step: what distinct factual claims does the answer actually make? For each one, does the retrieved context genuinely, directly support it?
3. Fill in `claims_matrix`: one entry per distinct factual claim, each tagged `is_grounded: true` or `is_grounded: false`.

Rules for tagging:
- A claim is grounded ONLY if the retrieved context explicitly states it or directly, unambiguously implies it. A claim that merely "sounds plausible" or is likely true in general is NOT grounded if the context doesn't actually say it.
- Break compound sentences into their real distinct claims. Don't merge multiple facts into one vague claim just to simplify your job.
- If the answer is a correct, well-calibrated refusal (it honestly states the context does not contain the requested information, and this is true) - it is making no factual claims about the world, so `claims_matrix` should be an empty list `[]`. Do not tag "the answer correctly refused" as a claim to be graded - that is not what claims_matrix is for.
- If the answer refuses on information that IS actually present in the retrieved context, that refusal is itself an incorrect claim - tag it as a single claim in claims_matrix with `is_grounded: false`.

Do not compute or mention any score. That is not your job - you only tag individual claims as grounded or not.

Respond with ONLY a JSON object matching this exact shape, no other text:
{
  "internal_rationale": "<your step-by-step reasoning>",
  "claims_matrix": [
    {"claim_text": "<claim 1>", "is_grounded": true},
    {"claim_text": "<claim 2>", "is_grounded": false}
  ]
}"""


def evaluate_response(question, context_chunks, generated_answer):
    """
    Runs the full Decompose -> Judge -> Score pipeline.

    Returns a dict:
      {
        "score": float,               # deterministic, Python-computed
        "total_claims": int,
        "grounded_claims": int,
        "claims_matrix": [...],       # the raw per-claim tags, for auditing
        "internal_rationale": str,
      }
    """
    context_block = "\n\n".join(f"[Context {i + 1}]\n{c}" for i, c in enumerate(context_chunks))

    user_prompt = f"""QUESTION:
{question}

RETRIEVED CONTEXT:
{context_block}

ANSWER:
{generated_answer}"""

    # --- PROBABILISTIC LAYER: the LLM tags individual claims. ---
    # This is the ONLY step where the model's output is trusted, and even
    # then only for discrete true/false tags on text it extracted itself -
    # never for a numeric score.
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    data = response.json()

    try:
        raw_content = data["choices"][0]["message"]["content"]
    except KeyError:
        raise RuntimeError(
            f"Judge API call failed or returned an unexpected shape: {json.dumps(data)[:300]}"
        )

    cleaned = raw_content.strip().strip("```json").strip("```").strip()

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge did not return valid JSON: {raw_content[:300]}") from e

    # Validate against the strict schema. If the model produced something
    # that doesn't match (wrong types, missing fields), this fails loudly
    # here rather than silently propagating a malformed claims_matrix into
    # the scoring math below.
    try:
        evaluation = EvaluationSchema.model_validate(parsed_json)
    except ValidationError as e:
        raise ValueError(f"Judge output failed schema validation: {e}") from e

    # --- DETERMINISTIC LAYER: plain Python arithmetic, no LLM involved. ---
    # This is the entire "scoring" step. It is 100% reproducible: given the
    # same claims_matrix, this always produces the same score, with no
    # ambiguity about where a threshold boundary sits.
    total_claims = len(evaluation.claims_matrix)
    grounded_claims = sum(1 for claim in evaluation.claims_matrix if claim.is_grounded)

    if total_claims == 0:
        # Explicit fallback guard: zero claims means the model made no
        # checkable factual assertions - typically a well-calibrated
        # refusal ("the context doesn't contain that information"). Rather
        # than divide by zero, or default to 0.0 (which would wrongly
        # penalize honesty, the exact bias found in Finding 15's
        # AnswerRelevancy metric), a well-calibrated refusal is awarded a
        # perfect score for data alignment.
        score = 1.0
    else:
        score = grounded_claims / total_claims

    return {
        "score": score,
        "total_claims": total_claims,
        "grounded_claims": grounded_claims,
        "claims_matrix": [claim.model_dump() for claim in evaluation.claims_matrix],
        "internal_rationale": evaluation.internal_rationale,
    }


if __name__ == "__main__":
    # Quick self-test: a genuinely correct, well-calibrated refusal.
    # Expect total_claims == 0 and score == 1.0.
    result = evaluate_response(
        question="What is the capital of France?",
        context_chunks=[
            "The history of coffee cultivation dates back centuries, originating in Ethiopia."
        ],
        generated_answer="I cannot answer this question using only the provided context, as it discusses coffee history rather than French geography.",
    )
    print(json.dumps(result, indent=2))
