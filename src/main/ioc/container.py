import os

from dependency_injector import containers, providers
from elasticsearch import AsyncElasticsearch

from src.infrastructure.adapters.elasticsearch_adapter import ElasticsearchAdapter
from src.infrastructure.adapters.gemini_adapter import GeminiAdapter
from src.infrastructure.adapters.pdf_processor_adapter import PDFProcessorAdapter
from sentence_transformers import SentenceTransformer


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    es_client = providers.Singleton(
        AsyncElasticsearch,
        hosts=[os.getenv("ES_HOST", "localhost")],
        api_key=os.getenv("ES_API_KEY"),
        headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"},
    )

    vector_store = providers.Singleton(
        ElasticsearchAdapter,
        es_client=es_client,
        index=os.getenv("ES_INDEX", "rag_index"),
    )

    embedder = providers.Singleton(
        SentenceTransformer, "all-MiniLM-L6-v2"
    )

    pdf_processor = providers.Singleton(
        PDFProcessorAdapter, embedder=embedder
    )

    llm_service = providers.Singleton(
        GeminiAdapter, api_key=os.getenv("GEMINI_API_KEY")
    )


container = Container()


def get_vector_store():
    return container.vector_store()


def get_llm_service():
    return container.llm_service()


def get_pdf_processor():
    return container.pdf_processor()
