from celery.result import AsyncResult
from fastapi import FastAPI
from pydantic import BaseModel

from celery_app import celery_app
from pipeline import answer_question_task, validate_query

app = FastAPI()


class Question(BaseModel):
    query: str
    tenant_id: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "rag-pipeline-api"}


@app.post("/ask")
def ask(q: Question):
    is_valid, error = validate_query(q.query)
    if not is_valid:
        return {"error": error}

    task = answer_question_task.delay(q.query, q.tenant_id)
    return {"task_id": task.id, "status": "processing"}


@app.get("/result/{task_id}")
def get_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == "PENDING":
        return {"status": "processing"}

    if task_result.state == "FAILURE":
        return {"status": "failed", "error": str(task_result.result)}

    if task_result.state == "SUCCESS":
        result = task_result.result
        return {
            "status": "complete",
            "question": result["question"],
            "answer": result["answer"],
            "tenant_id": result.get("tenant_id"),
        }

    return {"status": task_result.state.lower()}
