from fastapi import FastAPI

from src.main.ioc.container import container
from src.presentation.controllers.rag_router import router as rag_router

app = FastAPI(title="Chat eBook AI", version="1.0.0")

# Initialize DI
container.init_resources()

app.include_router(rag_router, prefix="/api", tags=["RAG"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
