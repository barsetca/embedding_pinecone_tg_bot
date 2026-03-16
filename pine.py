import logging
import os
from typing import Iterable, List, Dict, Any

from dotenv import load_dotenv
from pinecone.pinecone import Pinecone
from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import CloudProvider, AwsRegion
from pinecone.db_data import Index


load_dotenv()
logger = logging.getLogger(__name__)

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

    def __init__(self) -> None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        if not api_key:
            raise ValueError("Не указан PINECONE_API_KEY в .env")
        if not index_name:
            raise ValueError("Не указан PINECONE_INDEX_NAME в .env")

        try:
            self.pc = Pinecone(api_key=api_key)
            existing_names = set(self.pc.list_indexes().names())
            # Размерность выбрана для модели text-embedding-3-small (1536)
            if index_name not in existing_names:
                region_enum = _AWS_REGIONS.get(region, AwsRegion.US_EAST_1)
                logger.info("Создание индекса Pinecone: name=%s, region=%s", index_name, region)
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=CloudProvider.AWS, region=region_enum),
                )
            else:
                logger.debug("Используется существующий индекс: %s", index_name)
            self.index = self.pc.Index(index_name)
        except Exception as e:
            logger.exception("Ошибка подключения к Pinecone (индекс %s): %s", index_name, e)
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

        Returns:
            dict: ответ от Pinecone об операции удаления (deleteAll=True).
        """
        try:
            result = self.index.delete(delete_all=True)
            logger.warning("Delete all: все векторы в индексе удалены")
            return result
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

