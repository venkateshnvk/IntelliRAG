from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import os

app = FastAPI(
    title="IntelliRAG",
    description="Enterprise Healthcare Multi-Source RAG System",
    version="1.0.0"
)

# -----------------------------
# CORS Configuration
# -----------------------------

origins = [
    "https://happy-stone-0c01c011e.1.azurestaticapps.net",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Routes
# -----------------------------

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "IntelliRAG is running"}

# -----------------------------
# For local development only
# (Azure uses Gunicorn, not this block)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
