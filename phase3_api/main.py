"""Application FastAPI principale."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from phase3_api.dependencies import get_detector, get_pipeline

from phase3_api.schemas import HealthResponse
from phase3_api.endpoints import detect, rules, restore, analyze, graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Demarrage de l'API...")
    print("Chargement du detecteur YOLO...")
    get_detector()
    print("Chargement du pipeline RAG...")
    get_pipeline()
    print("API prete!")
    yield
    print("Arret de l'API")


app = FastAPI(
    title="Board Game AI API",
    description="API pour reconnaissance de jeux et restauration de regles",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={"vision": "ready", "rag": "ready"},
        supported_games=["chess", "checkers"],
    )


@app.get("/", tags=["System"])
async def root():
    return {"message": "Board Game AI API", "docs": "/docs", "health": "/health"}


app.include_router(detect.router, prefix="/api/v1", tags=["Vision"])
app.include_router(rules.router, prefix="/api/v1", tags=["RAG"])
app.include_router(restore.router, prefix="/api/v1", tags=["RAG"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Pipeline"])
app.include_router(graph.router, prefix="/api/v1", tags=["Graph"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase3_api.main:app", host="0.0.0.0", port=8000, reload=True)