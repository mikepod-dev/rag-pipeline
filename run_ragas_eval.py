import json
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

load_dotenv(override=True)

with open("ragas_dataset.json", "r") as f:
    raw_data = json.load(f)

samples = [
    SingleTurnSample(
        user_input=item["question"],
        response=item["answer"],
        retrieved_contexts=item["contexts"],
    )
    for item in raw_data
]
dataset = EvaluationDataset(samples=samples)

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="~anthropic/claude-haiku-latest",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print(result)

df = result.to_pandas()
df.to_json("ragas_results.json", orient="records", indent=2)
print("\nSaved per-question results to ragas_results.json")
