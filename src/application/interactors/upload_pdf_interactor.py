import logging
from typing import Dict, Any, List
from uuid import uuid4

from src.application.interfaces import IVectorStore, ILLMService, IPDFProcessor

logger = logging.getLogger(__name__)


class UploadPDFInteractor:
    def __init__(
        self,
        pdf_processor: IPDFProcessor,
        vector_store: IVectorStore,
        llm_service: ILLMService,
    ):
        self.pdf_processor = pdf_processor
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def execute(
        self, file_content: bytes, filename: str, conversation_id: str
    ) -> Dict[str, Any]:
        try:
            # Process PDF and extract chunks
            chunks = await self.pdf_processor.process_pdf(file_content, filename)

            if not chunks:
                raise ValueError("No text found in PDF")

            # Generate embeddings and index
            chunk_count = await self.vector_store.index(chunks, conversation_id)

            # Generate summary using LLM
            full_text = " ".join([chunk["text"] for chunk in chunks])
            system_prompt = (
                "Summarize the following text in about 3 sentences, focusing on the main themes. "
                f"{full_text[:1000]}"
            )

            summary = await self.llm_service.generate(system_prompt)

            return {
                "message": "File indexed successfully",
                "conversation_id": conversation_id,
                "chunks": chunk_count,
                "summary": summary,
            }

        except Exception as e:
            logger.exception("PDF upload failed")
            raise