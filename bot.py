import logging
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from pine import PineconeClient


load_dotenv()

# Уровень логирования из окружения (по умолчанию INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Не логировать HTTP-запросы httpx/httpcore — в URL попадает TELEGRAM_BOT_TOKEN
for _logger_name in ("httpx", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "data.txt")


if not OPENAI_API_KEY:
    raise ValueError("Не указан OPENAI_API_KEY в .env")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не указан TELEGRAM_BOT_TOKEN в .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pine_client = PineconeClient()
logger.info("Клиенты OpenAI и Pinecone инициализированы")


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Получить embedding для переданного текста с помощью OpenAI.

    Args:
        text: исходный текст.
        model: имя модели embedding (по умолчанию EMBEDDING_MODEL).

    Returns:
        List[float]: список координат вектора embedding.
    """
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.exception("Ошибка при получении embedding для текста длиной %d символов: %s", len(text), e)
        raise


def initialize_index_from_file() -> None:
    """
    Однократная инициализация индекса Pinecone из файла data/data.txt.

    Логика:
        - Если в индексе уже есть векторы (total_vector_count > 0), ничего не делаем.
        - Иначе читаем файл data.txt, для каждой непустой строки:
            * создаём ID равный порядковому номеру строки (начиная с 1),
            * получаем embedding с помощью get_embedding,
            * сохраняем в Pinecone с метаданными, содержащими исходную фразу.
    """
    try:
        stats = pine_client.describe_stats()
        total = getattr(stats, "total_vector_count", 0)
        if total > 0:
            logger.info("Индекс уже содержит %s векторов, пропуск загрузки из data.txt", total)
            return

        if not os.path.exists(DATA_FILE_PATH):
            logger.warning("Файл данных не найден: %s", DATA_FILE_PATH)
            return

        with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info("Загрузка %d фраз из %s в Pinecone", len(lines), DATA_FILE_PATH)
        vectors: List[Dict[str, Any]] = []
        for idx, phrase in enumerate(lines, start=1):
            try:
                embedding = get_embedding(phrase)
                vector = {
                    "id": str(idx),
                    "values": embedding,
                    "metadata": {"text": phrase},
                }
                vectors.append(vector)
            except Exception as e:
                logger.exception("Не удалось получить embedding для фразы #%d, пропуск: %s", idx, e)
                continue

        if vectors:
            pine_client.upsert_vectors(vectors)
            logger.info("В индекс записано %d векторов из data.txt", len(vectors))
    except Exception as e:
        logger.exception("Ошибка инициализации индекса из файла: %s", e)
        raise


def extract_memory_text(message: str) -> str:
    """
    Определить, является ли сообщение просьбой запомнить информацию,
    и вернуть текст для сохранения в память (или пустую строку, если нет).
    """
    lowered = message.lower().strip()
    memory_triggers = ["важно", "запомни", "запомнить", "запиши", "это важно"]

    for trigger in memory_triggers:
        if lowered.startswith(trigger):
            # Убираем ключевое слово и возможные двоеточия/пробелы
            without_trigger = lowered[len(trigger) :].lstrip(" :,-")
            # Если после триггера ничего нет, сохраняем всё сообщение как есть
            return without_trigger if without_trigger else message

    return ""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /start.
    """
    user_id = update.effective_user.id if update.effective_user else None
    logger.info("Команда /start от user_id=%s", user_id)
    await update.message.reply_text(
        "Привет! Я умный бот-помощник. Пиши вопросы, а также можешь помечать важную информацию словами "
        "\"Важно\" или \"Запомни\", чтобы я сохранял её в память."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Основной обработчик текстовых сообщений.

    Логика:
        1. При необходимости сохраняем важную информацию в память (Pinecone).
        2. Делаем embedding запроса пользователя.
        3. Ищем топ-10 близких воспоминаний в Pinecone.
        4. Отправляем запрос и воспоминания в LLM (gpt-4o-mini) для генерации ответа.
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    user_id = update.effective_user.id if update.effective_user else None
    logger.debug("Сообщение от user_id=%s: %s", user_id, user_text[:100])

    try:
        # 1. Сохраняем важную информацию, если есть триггер
        memory_text = extract_memory_text(user_text)
        if memory_text:
            try:
                embedding = get_embedding(memory_text)
                stats = pine_client.describe_stats()
                current_count = getattr(stats, "total_vector_count", 0)
                memory_id = f"user_memory_{current_count + 1}"
                pine_client.upsert_vectors(
                    [
                        {
                            "id": memory_id,
                            "values": embedding,
                            "metadata": {"text": memory_text},
                        }
                    ]
                )
                logger.info("Сохранено воспоминание id=%s для user_id=%s", memory_id, user_id)
            except Exception as e:
                logger.exception("Не удалось сохранить воспоминание: %s", e)
                await update.message.reply_text(
                    "Не удалось сохранить в память. Попробуйте позже или переформулируйте."
                )
                return

        # 2. Embedding запроса
        query_embedding = get_embedding(user_text)

        # 3. Поиск ближайших воспоминаний
        search_response = pine_client.query(query_embedding, top_k=10, include_metadata=True)
        matches = search_response.get("matches", []) or []

        memories_texts: List[str] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            text = metadata.get("text")
            if text:
                memories_texts.append(text)

        logger.debug("Найдено воспоминаний: %d", len(memories_texts))

        context_block = ""
        if memories_texts:
            joined_memories = "\n- ".join(memories_texts)
            context_block = (
                "Вот воспоминания и связанные факты, которые ты знаешь:\n"
                f"- {joined_memories}\n\n"
                "Используй их, если они помогают ответить на вопрос пользователя."
            )

        # 4. Запрос к LLM
        messages = []
        system_prompt = (
            "Ты телеграм-бот-помощник, который использует сохранённые воспоминания из векторной базы. "
            "Отвечай по-русски, понятно и кратко. Если фактов не хватает, можно отвечать общими словами."
        )
        if context_block:
            system_prompt += "\n\n" + context_block

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        answer = completion.choices[0].message.content.strip()

        await update.message.reply_text(answer)
        logger.debug("Ответ отправлен user_id=%s", user_id)
    except Exception as e:
        logger.exception("Ошибка при обработке сообщения от user_id=%s: %s", user_id, e)
        try:
            await update.message.reply_text(
                "Произошла ошибка при формировании ответа. Попробуйте позже или переформулируйте запрос."
            )
        except Exception as send_err:
            logger.exception("Не удалось отправить сообщение об ошибке пользователю: %s", send_err)


def main() -> None:
    """
    Точка входа в приложение.

    Действия:
        - Однократная инициализация индекса из файла data.txt (при первом запуске и пустом индексе).
        - Запуск Telegram‑бота в режиме long polling.
    """
    logger.info("Запуск бота, уровень логирования: %s", LOG_LEVEL)
    try:
        initialize_index_from_file()
    except Exception as e:
        logger.exception("Критическая ошибка при инициализации индекса: %s", e)
        sys.exit(1)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен (long polling)")
    try:
        app.run_polling()
    except Exception as e:
        logger.exception("Ошибка при работе бота: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

