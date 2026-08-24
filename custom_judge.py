import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")

# NOTE: 'anthropic/claude-3.5-haiku' was tested against OpenRouter earlier in
# this project's own RAGAS integration (Finding 15) and returned
# `404 - No endpoints found for anthropic/claude-3.5-haiku`. Defaulting to
# ask_llm's proven-working string instead. Change this one line if you want
# to re-verify the requested slug against OpenRouter's current model list.
JUDGE_MODEL = "~anthropic/claude-haiku-latest"

JUDGE_SYSTEM_PROMPT = """You are a strict, careful evaluator of RAG (retrieval-augmented generation) answers. You will be given a QUESTION, the RETRIEVED CONTEXT chunks that were available to the answering system, and the ANSWER it produced.

Score two things, each from 0.0 to 1.0:

1. FAITHFULNESS: Is every factual claim in the ANSWER actually supported by the RETRIEVED CONTEXT?
   - Internally decompose the answer into its individual factual claims before scoring. Do not judge the answer holistically as one block.
   - For each claim, check whether it is explicitly stated or directly, unambiguously implied by the retrieved context.
   - A specific number, name, date, or detail that does not appear anywhere in the context is an ungrounded claim, even if it sounds plausible or is likely true in general.
   - The faithfulness score should reflect the proportion of claims that are genuinely grounded, not a vibe-based overall impression.

2. CONTEXT_PRECISION: Of the retrieved context chunks, what proportion were actually relevant and useful for answering this specific question?
   - A chunk that is topically related but does not help answer the specific question counts against precision.
   - This is about the quality of what was retrieved, not about the answer itself.

CRITICAL CALIBRATION RULE - read carefully, this is the most common scoring mistake:
If the ANSWER correctly and honestly states that the retrieved context does NOT contain the specific information requested (a well-calibrated refusal or partial refusal), and this is TRUE - the context genuinely does not contain that information - then this is a CORRECT answer and must be scored FAITHFULNESS 1.00. Do not penalize honesty about missing information.
However, if the answer refuses or hedges on information that IS actually present in the retrieved context, that is a FAILURE to use the available context, and must be scored low. Check the actual context yourself before accepting the refusal at face value - a refusal is only correct if the information genuinely is not there. Do not give every refusal-shaped answer a free pass; verify each one against the real context.

Respond with ONLY a JSON object in exactly this shape, no other text:
{
  "faithfulness": <float 0.0-1.0>,
  "context_precision": <float 0.0-1.0>,
  "rationale": "<one or two sentences explaining both scores, mentioning specific claims or chunks>"
}"""


def evaluate_query(question, answer, contexts):
    context_block = "\n\n".join(f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts))

    user_prompt = f"""QUESTION:
{question}

RETRIEVED CONTEXT:
{context_block}

ANSWER:
{answer}"""

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
        return {
            "faithfulness": None,
            "context_precision": None,
            "rationale": f"API error - malformed response: {json.dumps(data)[:300]}",
        }

    cleaned = raw_content.strip().strip("```json").strip("```").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "faithfulness": None,
            "context_precision": None,
            "rationale": f"Failed to parse judge output as JSON: {raw_content[:300]}",
        }

    return {
        "faithfulness": result.get("faithfulness"),
        "context_precision": result.get("context_precision"),
        "rationale": result.get("rationale", ""),
    }


if __name__ == "__main__":
    test_result = evaluate_query(
        question="What is the capital of France?",
        answer="I cannot answer this question using only the provided context, as it discusses coffee history rather than French geography.",
        contexts=[
            "The history of coffee cultivation dates back centuries, originating in Ethiopia."
        ],
    )
    print(json.dumps(test_result, indent=2))
