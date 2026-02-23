from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="IntelliRAG",
    description="Enterprise Healthcare Multi-Source RAG System",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "IntelliRAG is running"}