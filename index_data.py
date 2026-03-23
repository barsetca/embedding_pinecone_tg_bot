"""
Отдельный скрипт индексации (рекомендуемый способ из практик RAG):
- индекс авто-консультанта из data/data.txt;
- опционально RAG-индекс из data/data.txt.
"""
import logging

from bot import initialize_index_from_file
from rag_agent import RAGAgent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    initialize_index_from_file()
    rag = RAGAgent()
    rag.initialize_knowledge_base()


if __name__ == "__main__":
    main()
