from fastapi import FastAPI
from api.routes import router
import os

app = FastAPI(
    title="IntelliRAG",
    description="Enterprise Healthcare Multi-Source RAG System",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "IntelliRAG is running"}


# Needed for Azure Web App
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
