import json
import logging
import os
from typing import Iterable, List, Dict, Any

from dotenv import load_dotenv
from pinecone.pinecone import Pinecone
from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import CloudProvider, AwsRegion
from pinecone.db_data import Index
from pinecone.exceptions import NotFoundException


load_dotenv()
logger = logging.getLogger(__name__)


def _is_pinecone_namespace_not_found(exc: NotFoundException) -> bool:
    """
    True, если Pinecone вернул 404 из-за отсутствия namespace (часто у пустого индекса без upsert).
    """
    body = exc.body
    if body is None:
        return "Namespace not found" in str(exc)
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        if "Namespace not found" in body:
            return True
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("message") == "Namespace not found":
                return True
        except json.JSONDecodeError:
            pass
    return False

# Маппинг строки региона из .env в enum AwsRegion (при необходимости можно расширить)
_AWS_REGIONS = {
    "us-east-1": AwsRegion.US_EAST_1,
    "us-west-2": AwsRegion.US_WEST_2,
}


class PineconeClient:
    """
    Высокоуровневый клиент для работы с векторной базой Pinecone.

    Атрибуты:
        pc (Pinecone): объект клиентского подключения к Pinecone.
        index (Index): объект индекса Pinecone для операций с векторами.

    Параметры окружения (.env):
        PINECONE_API_KEY (str): API‑ключ Pinecone.
        PINECONE_INDEX_NAME (str): название индекса в Pinecone.
        PINECONE_REGION (str): регион для serverless‑индекса (по умолчанию "us-east-1").

    Методы:
        upsert_vectors(vectors): массовая запись или обновление векторов.
        query(vector, top_k, include_metadata): поиск ближайших векторов.
        delete_by_ids(ids): удаление векторов по списку ID.
        delete_all(): удаление всех векторов в индексе.
        describe_stats(): получение статистики индекса (количество векторов и др.).
    """

    def __init__(self, index_name: str | None = None, dimension: int = 1536) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        selected_index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        if not api_key:
            raise ValueError("Не указан PINECONE_API_KEY в .env")
        if not selected_index_name:
            raise ValueError("Не указан PINECONE_INDEX_NAME в .env")

        self.index_name = selected_index_name
        self.dimension = dimension

        try:
            self.pc = Pinecone(api_key=api_key)
            existing_names = set(self.pc.list_indexes().names())
            # Размерность по умолчанию выбрана для модели text-embedding-3-small (1536)
            if selected_index_name not in existing_names:
                region_enum = _AWS_REGIONS.get(region, AwsRegion.US_EAST_1)
                logger.info(
                    "Создание индекса Pinecone: name=%s, region=%s, dimension=%s",
                    selected_index_name,
                    region,
                    dimension,
                )
                self.pc.create_index(
                    name=selected_index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=CloudProvider.AWS, region=region_enum),
                )
            else:
                logger.debug("Используется существующий индекс: %s", selected_index_name)
            self.index = self.pc.Index(selected_index_name)
        except Exception as e:
            logger.exception("Ошибка подключения к Pinecone (индекс %s): %s", selected_index_name, e)
            raise

    def upsert_vectors(self, vectors: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Массовая запись или обновление векторов в индексе.

        Args:
            vectors: итерируемый объект словарей вида:
                {
                    "id": str,               # уникальный идентификатор вектора
                    "values": List[float],   # список координат вектора
                    "metadata": dict         # произвольные метаданные (например, текст)
                }

        Returns:
            dict: ответ от Pinecone с информацией об операции upsert.
        """
        try:
            vectors_list = list(vectors)
            result = self.index.upsert(vectors=vectors_list)
            logger.debug("Upsert: записано векторов: %d", len(vectors_list))
            return result
        except Exception as e:
            logger.exception("Ошибка upsert в Pinecone: %s", e)
            raise

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Поиск ближайших векторов в индексе.

        Args:
            vector: вектор запроса (список float).
            top_k: максимальное количество ближайших векторов.
            include_metadata: включать ли метаданные в ответ.

        Returns:
            dict: ответ от Pinecone с найденными совпадениями (matches).
        """
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata,
            )
            matches_count = len(result.get("matches", []) or [])
            logger.debug("Query: найдено совпадений: %d", matches_count)
            return result
        except Exception as e:
            logger.exception("Ошибка query в Pinecone: %s", e)
            raise

    def delete_by_ids(self, ids: Iterable[str]) -> Dict[str, Any]:
        """
        Удалить векторы по списку идентификаторов.

        Args:
            ids: итерируемый объект строковых ID.

        Returns:
            dict: ответ от Pinecone об операции удаления.
        """
        try:
            ids_list = list(ids)
            result = self.index.delete(ids=ids_list)
            logger.debug("Delete: удалено id: %d", len(ids_list))
            return result
        except Exception as e:
            logger.exception("Ошибка delete_by_ids в Pinecone: %s", e)
            raise

    def delete_all(self) -> Dict[str, Any]:
        """
        Удалить все векторы из индекса.

        У пустого индекса Pinecone serverless иногда отвечает 404 «Namespace not found» —
        это считаем успехом (удалять нечего).

        Returns:
            dict: ответ от Pinecone об операции удаления (deleteAll=True).
        """
        try:
            result = self.index.delete(delete_all=True)
            logger.warning("Delete all: все векторы в индексе удалены")
            return result
        except NotFoundException as e:
            if _is_pinecone_namespace_not_found(e):
                logger.info(
                    "delete_all: индекс «%s» уже пуст (namespace не создан), пропуск",
                    self.index_name,
                )
                return {}
            logger.exception("Ошибка delete_all в Pinecone: %s", e)
            raise
        except Exception as e:
            logger.exception("Ошибка delete_all в Pinecone: %s", e)
            raise

    def describe_stats(self) -> Dict[str, Any]:
        """
        Получить статистику индекса, включая количество векторов.

        Returns:
            dict: структура с полями namespace, total_vector_count и др.
        """
        try:
            result = self.index.describe_index_stats()
            return result
        except Exception as e:
            logger.exception("Ошибка describe_index_stats в Pinecone: %s", e)
            raise

    def integration_check(self) -> bool:
        """
        Проверка интеграции с Pinecone: тестовый запрос к API индекса.

        Выполняется describe_index_stats и query с нулевым вектором нужной размерности
        (тот же тип вызова, что при семантическом поиске в авто-режиме).

        Returns:
            True если оба вызова успешны, иначе False (ошибка логируется).
        """
        try:
            stats = self.describe_stats()
            total = getattr(stats, "total_vector_count", None)
            if total is None and isinstance(stats, dict):
                total = stats.get("total_vector_count", 0)
            dummy_vector = [0.0] * self.dimension
            self.query(dummy_vector, top_k=1, include_metadata=True)
            logger.info(
                "Pinecone: проверка индекса «%s» OK (describe_index_stats + query), векторов в индексе: %s",
                self.index_name,
                total,
            )
            return True
        except Exception:
            logger.exception(
                "Pinecone: проверка индекса «%s» не пройдена (describe_index_stats / query)",
                self.index_name,
            )
            return False

