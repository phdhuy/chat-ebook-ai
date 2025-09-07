import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock

import pytest

from src.application.interactors.query_rag_interactor import QueryRAGInteractor
from src.domain.entities.schemas import QueryRequest


@pytest.mark.asyncio
async def test_query_rag_interactor():
    # Mock dependencies
    mock_vector_store = AsyncMock()
    mock_llm_service = AsyncMock()

    # Setup mocks
    mock_vector_store.search.return_value = [
        {
            "_score": 3.0,
            "_source": {"chunk": "test chunk", "page": 1, "conversation_id": "123"},
        }
    ]
    mock_llm_service.generate.side_effect = [
        "hypothetical document",  # For HyDE
        "Generated answer",  # For final answer
    ]

    # Create interactor
    interactor = QueryRAGInteractor(mock_vector_store, mock_llm_service)

    # Test request
    request = QueryRequest(query="test query", conversation_id="123")

    # Execute
    result = await interactor.execute(
        request.query, request.history, request.conversation_id
    )

    # Assertions
    assert "answer" in result
    assert "cited_excerpts" in result
    assert result["answer"] == "Generated answer"
    mock_vector_store.search.assert_called_once()
    assert mock_llm_service.generate.call_count == 2


@pytest.mark.asyncio
async def test_query_request_validation():
    import pytest
    from pydantic import ValidationError

    # Test valid request
    request = QueryRequest(query="test", conversation_id="123")
    assert request.query == "test"
    assert request.conversation_id == "123"

    # Test invalid request (missing required fields)
    with pytest.raises(ValidationError):
        QueryRequest(query="test")  # missing conversation_id

    with pytest.raises(ValidationError):
        QueryRequest(conversation_id="123")  # missing query
