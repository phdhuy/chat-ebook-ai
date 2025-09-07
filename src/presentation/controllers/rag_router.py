import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.application.interactors.query_rag_interactor import QueryRAGInteractor
from src.application.interactors.upload_pdf_interactor import UploadPDFInteractor
from src.application.interfaces import ILLMService, IPDFProcessor, IVectorStore
from src.domain.entities.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from src.main.ioc.container import (
    get_llm_service,
    get_pdf_processor,
    get_vector_store,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    vector_store: IVectorStore = Depends(get_vector_store),
    llm_service: ILLMService = Depends(get_llm_service),
) -> QueryResponse:
    interactor = QueryRAGInteractor(vector_store, llm_service)
    try:
        result = await interactor.execute(
            request.query, request.history or "", request.conversation_id
        )
        return QueryResponse(
            answer=result["answer"], cited_excerpts=result["cited_excerpts"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    conversation_id: str,
    file: UploadFile = File(...),
    pdf_processor: IPDFProcessor = Depends(get_pdf_processor),
    vector_store: IVectorStore = Depends(get_vector_store),
    llm_service: ILLMService = Depends(get_llm_service),
) -> UploadResponse:
    try:
        # Read file content
        file_content = await file.read()

        # Execute upload interactor
        interactor = UploadPDFInteractor(pdf_processor, vector_store, llm_service)
        result = await interactor.execute(file_content, file.filename, conversation_id)

        return UploadResponse(
            message=result["message"],
            conversation_id=result["conversation_id"],
            chunks=result["chunks"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e
