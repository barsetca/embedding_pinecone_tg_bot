"""
Тесты для клиента Pinecone (pine.py).

Pinecone API и индекс замоканы, реальные запросы не выполняются.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def env_pinecone():
    """Переменные окружения для инициализации PineconeClient."""
    return {
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "PINECONE_REGION": "us-east-1",
    }


@pytest.fixture
def mock_pinecone_index():
    """Мок индекса Pinecone с методами upsert, query, delete, describe_index_stats."""
    index = MagicMock()
    index.upsert.return_value = {"upserted_count": 1}
    index.query.return_value = {"matches": []}
    index.delete.return_value = {}
    index.describe_index_stats.return_value = MagicMock(total_vector_count=0)
    return index


@pytest.fixture
def mock_pinecone(mock_pinecone_index, env_pinecone):
    """
    Мок класса Pinecone: при создании возвращает объект с list_indexes и Index.
    """
    with patch.dict(os.environ, env_pinecone, clear=False):
        with patch("pine.Pinecone") as MockPine:
            pc_instance = MagicMock()
            pc_instance.list_indexes.return_value.names.return_value = ["test-index"]
            pc_instance.Index.return_value = mock_pinecone_index
            MockPine.return_value = pc_instance
            yield MockPine


@pytest.fixture
def pine_client(mock_pinecone, env_pinecone):
    """Экземпляр PineconeClient с замоканным Pinecone."""
    with patch.dict(os.environ, env_pinecone, clear=False):
        with patch("pine.Pinecone") as MockPine:
            pc_instance = MagicMock()
            pc_instance.list_indexes.return_value.names.return_value = ["test-index"]
            mock_index = MagicMock()
            mock_index.upsert.return_value = {"upserted_count": 1}
            mock_index.query.return_value = {"matches": []}
            mock_index.delete.return_value = {}
            mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=0)
            pc_instance.Index.return_value = mock_index
            MockPine.return_value = pc_instance

            from pine import PineconeClient
            client = PineconeClient()
            client.index = mock_index
            yield client


class TestPineconeClientInit:
    """Тесты инициализации PineconeClient."""

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"PINECONE_INDEX_NAME": "idx"}, clear=False):
                from pine import PineconeClient
                with pytest.raises(ValueError, match="PINECONE_API_KEY"):
                    PineconeClient()

    def test_raises_without_index_name(self, env_pinecone):
        env = {k: v for k, v in env_pinecone.items() if k != "PINECONE_INDEX_NAME"}
        env["PINECONE_API_KEY"] = "key"
        with patch.dict(os.environ, env, clear=False):
            from pine import PineconeClient
            with pytest.raises(ValueError, match="PINECONE_INDEX_NAME"):
                PineconeClient()


class TestPineconeClientMethods:
    """Тесты методов PineconeClient с замоканным индексом."""

    def test_upsert_vectors(self, pine_client):
        vectors = [
            {"id": "1", "values": [0.1] * 1536, "metadata": {"text": "фраза 1"}},
        ]
        result = pine_client.upsert_vectors(vectors)
        pine_client.index.upsert.assert_called_once()
        call_kw = pine_client.index.upsert.call_args[1]
        assert len(call_kw["vectors"]) == 1
        assert call_kw["vectors"][0]["id"] == "1"
        assert result["upserted_count"] == 1

    def test_query(self, pine_client):
        vector = [0.1] * 1536
        pine_client.index.query.return_value = {
            "matches": [
                {"id": "1", "metadata": {"text": "найденный текст"}, "score": 0.9}
            ]
        }
        result = pine_client.query(vector, top_k=5, include_metadata=True)
        pine_client.index.query.assert_called_once_with(
            vector=vector, top_k=5, include_metadata=True
        )
        assert len(result["matches"]) == 1
        assert result["matches"][0]["metadata"]["text"] == "найденный текст"

    def test_delete_by_ids(self, pine_client):
        result = pine_client.delete_by_ids(["id1", "id2"])
        pine_client.index.delete.assert_called_once_with(ids=["id1", "id2"])
        assert result == {}

    def test_delete_all(self, pine_client):
        result = pine_client.delete_all()
        pine_client.index.delete.assert_called_once_with(delete_all=True)
        assert result == {}

    def test_describe_stats(self, pine_client):
        pine_client.index.describe_index_stats.return_value = MagicMock(
            total_vector_count=42
        )
        result = pine_client.describe_stats()
        pine_client.index.describe_index_stats.assert_called_once()
        assert getattr(result, "total_vector_count", None) == 42
