import logging
from typing import Dict, Any, List
import fitz
import nltk
from sentence_transformers import SentenceTransformer

from src.application.interfaces import IPDFProcessor

logger = logging.getLogger(__name__)


class PDFProcessorAdapter(IPDFProcessor):
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def _allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

    async def process_pdf(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        if not self._allowed_file(filename):
            raise ValueError("Invalid file type. Please upload a PDF")

        try:
            # Extract text from PDF
            text_by_page = []
            doc = fitz.open(stream=file_content, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc[page_num]
                extracted = page.get_text()
                if extracted.strip():
                    text_by_page.append({"page": page_num + 1, "text": extracted})

            doc.close()

            if not text_by_page:
                raise ValueError("No extractable text found in PDF")

            # Chunk the text
            chunk_size, overlap = 18, 1
            chunks = []

            for page_data in text_by_page:
                page_num = page_data["page"]
                page_text = " ".join(page_data["text"].split())
                sentences = nltk.sent_tokenize(page_text)

                for i in range(0, len(sentences), chunk_size - overlap):
                    end = min(i + chunk_size, len(sentences))
                    chunk_text = " ".join(sentences[i:end])
                    if chunk_text:
                        chunks.append({"text": chunk_text, "page": page_num})
                    if end == len(sentences):
                        break

            logger.info(f"Created {len(chunks)} chunks from PDF")

            # Generate embeddings
            if chunks:
                texts = [chunk["text"] for chunk in chunks]
                embeddings = self.embedder.encode(
                    texts, convert_to_tensor=False, normalize_embeddings=True
                )

                # Add embeddings to chunks
                for i, chunk in enumerate(chunks):
                    chunk["embedding"] = embeddings[i].tolist()

            return chunks

        except Exception as e:
            logger.exception("PDF processing failed")
            raise