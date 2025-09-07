from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IVectorStore(ABC):
    @abstractmethod
    async def search(
        self, query_vector: List[float], conversation_id: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def index(self, chunks: List[Dict[str, Any]], conversation_id: str) -> int:
        pass


class ILLMService(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass


class IPDFProcessor(ABC):
    @abstractmethod
    async def process_pdf(
        self, file_content: bytes, filename: str
    ) -> List[Dict[str, Any]]:
        pass
