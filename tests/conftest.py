"""
Общие фикстуры для тестов.

Перед импортом модуля bot необходимо задать переменные окружения и замокать
PineconeClient, иначе импорт упадёт из-за отсутствия ключей и подключения к Pinecone.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session")
def env_vars():
    """Минимальный набор переменных окружения для импорта bot."""
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "TELEGRAM_BOT_TOKEN": "test-telegram-token",
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
    }


@pytest.fixture(scope="session")
def bot_module(env_vars):
    """
    Загружает модуль bot с подставленными env и замоканным PineconeClient.
    Использовать в тестах, которым нужны функции из bot (например, extract_memory_text).
    """
    with patch.dict(os.environ, env_vars, clear=False):
        with patch("pine.PineconeClient", MagicMock()):
            with patch("rag_agent.RAGAgent", MagicMock()):
                import bot as bot_mod
                return bot_mod


@pytest.fixture
def extract_memory_text(bot_module):
    """Функция извлечения текста для запоминания из сообщения."""
    return bot_module.extract_memory_text
