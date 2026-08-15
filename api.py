from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import ask_llm, hybrid_search_with_rerank, validate_query

app = FastAPI()


class Question(BaseModel):
    query: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "rag-pipeline-api"}


@app.post("/ask")
def ask(q: Question):
    is_valid, error = validate_query(q.query)
    if not is_valid:
        return {"error": error}

    results = hybrid_search_with_rerank(q.query)
    answer = ask_llm(q.query, results["documents"][0])
    return {"question": q.query, "answer": answer}
