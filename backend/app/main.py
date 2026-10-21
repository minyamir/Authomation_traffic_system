
from fastapi import FastAPI

app = FastAPI(
    title="Bahir Dar AI Traffic Management",
    description=(
        "AI-powered vehicle detection, traffic congestion analysis, "
        "traffic signal recommendations, and speed monitoring."
    ),
    version="0.1.0",
)


@app.get("/")
async def home():
    return {
        "message": "Bahir Dar AI Traffic Management API is running",
        "status": "healthy",
    }


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Bahir Dar AI Traffic Management",
    }
