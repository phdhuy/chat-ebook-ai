import logging

from src.application.interfaces import ILLMService, IVectorStore

logger = logging.getLogger(__name__)


class QueryRAGInteractor:
    def __init__(self, vector_store: IVectorStore, llm_service: ILLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def execute(self, query: str, history: str, conversation_id: str) -> dict:
        # HyDE: Generate hypothetical document
        hyde_prompt = f"Create a concise, factual, and detailed hypothetical document answering the question: '{query}'."
        hyde_response = await self.llm_service.generate(hyde_prompt)
        _ = hyde_response.strip()  # Placeholder for future use

        # Assume embedder is available; for now, mock embedding
        q_vec = [0.0] * 384  # Placeholder for embedding

        # Search vector store
        hits = await self.vector_store.search(q_vec, conversation_id, k=20)

        # Filter and process results
        matched = []
        for i, hit in enumerate(hits):
            if (
                hit["_score"] >= 2.5
                and hit["_source"]["conversation_id"] == conversation_id
            ):
                matched.append(
                    {
                        "id": i + 1,
                        "text": hit["_source"]["chunk"],
                        "page": hit["_source"].get("page", "Unknown"),
                        "score": hit["_score"],
                    }
                )

        matched.sort(key=lambda x: x["score"], reverse=True)
        matched = matched[:15]

        context = "\n".join([f"- [Page {m['page']}]: {m['text']}" for m in matched])

        # Generate answer
        prompt = f"Answer based on context: {context}\nQuestion: {query}"
        answer = await self.llm_service.generate(prompt)

        cited_excerpts = [
            {"id": m["id"], "text": m["text"], "page": m["page"], "score": m["score"]}
            for m in matched
        ]

        return {"answer": answer, "cited_excerpts": cited_excerpts}
