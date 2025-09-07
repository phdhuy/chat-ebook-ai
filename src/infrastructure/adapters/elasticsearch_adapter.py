from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch

from src.application.interfaces import IVectorStore


class ElasticsearchAdapter(IVectorStore):
    def __init__(self, es_client: AsyncElasticsearch, index: str):
        self.es = es_client
        self.index = index

    async def search(
        self, query_vector: List[float], conversation_id: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"chunk": {"query": "placeholder"}}},  # Simplified
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": query_vector,
                                "k": k,
                            }
                        },
                    ]
                }
            },
        }
        res = await self.es.search(index=self.index, body=body)
        hits = res.get("hits", {}).get("hits", [])
        return hits if isinstance(hits, list) else []

    async def index(self, chunks: List[Dict[str, Any]], conversation_id: str) -> int:
        # Simplified indexing
        actions = []
        for chunk in chunks:
            actions.append(
                {
                    "_index": self.index,
                    "_source": {
                        "chunk": chunk["text"],
                        "page": chunk["page"],
                        "conversation_id": conversation_id,
                        "embedding": chunk["embedding"],
                    },
                }
            )
        # Bulk index
        await self.es.bulk(actions)
        return len(actions)
