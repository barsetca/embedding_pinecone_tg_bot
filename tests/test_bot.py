"""
Тесты для логики бота (bot.py).

Используются моки для OpenAI, Pinecone и Telegram Update/Context,
чтобы не обращаться к реальным API.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import bot_module  # noqa: F401 — фикстура для загрузки bot


class TestExtractMemoryText:
    """Тесты для extract_memory_text."""

    def test_empty_string(self, extract_memory_text):
        assert extract_memory_text("") == ""

    def test_no_trigger(self, extract_memory_text):
        assert extract_memory_text("Просто вопрос про погоду") == ""
        assert extract_memory_text("Расскажи про Volkswagen") == ""

    def test_trigger_важно(self, extract_memory_text):
        assert extract_memory_text("Важно: пароль от почты 123") == "пароль от почты 123"
        assert extract_memory_text("Важно пароль 123") == "пароль 123"
        # Текст после триггера возвращается в нижнем регистре (обработка через lowered)
        assert extract_memory_text("ВАЖНО контакт Иванов") == "контакт иванов"

    def test_trigger_запомни(self, extract_memory_text):
        assert extract_memory_text("Запомни: день рождения 15 марта") == "день рождения 15 марта"
        assert extract_memory_text("Запомни адрес офиса") == "адрес офиса"

    def test_trigger_запомнить(self, extract_memory_text):
        # В списке триггеров раньше идёт «запомни», поэтому срезается 8 символов
        assert extract_memory_text("Запомнить номер телефона 8-999-123-45-67") == "ть номер телефона 8-999-123-45-67"

    def test_trigger_запиши(self, extract_memory_text):
        assert extract_memory_text("Запиши: встреча в понедельник") == "встреча в понедельник"

    def test_trigger_это_важно(self, extract_memory_text):
        # После триггера lstrip убирает « - »
        assert extract_memory_text("Это важно - не забыть документы") == "не забыть документы"

    def test_trigger_only_no_content(self, extract_memory_text):
        """Если после триггера пусто, возвращается исходное сообщение."""
        assert extract_memory_text("Важно") == "Важно"
        assert extract_memory_text("Запомни ") == "Запомни "

    def test_trigger_with_punctuation(self, extract_memory_text):
        assert extract_memory_text("Важно:   текст") == "текст"
        assert extract_memory_text("Запомни, что купить") == "что купить"


@pytest.mark.asyncio
class TestStartHandler:
    """Тесты для обработчика /start."""

    async def test_start_replies_with_welcome(self, bot_module):
        update = MagicMock()
        update.effective_user = MagicMock(id=12345)
        update.message = AsyncMock()
        context = MagicMock()

        await bot_module.start(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Привет" in call_args
        assert "Запомни" in call_args or "Важно" in call_args


@pytest.mark.asyncio
class TestHandleMessage:
    """Тесты для обработчика сообщений (с моками OpenAI и Pinecone)."""

    @pytest.fixture
    def mock_openai_response(self):
        """Ответ OpenAI chat completion."""
        msg = MagicMock()
        msg.content = "Тестовый ответ бота"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.fixture
    def mock_embedding(self):
        return [0.1] * 1536

    async def test_handle_message_sends_reply(
        self, bot_module, mock_openai_response, mock_embedding
    ):
        """При успешном прохождении цепочки пользователь получает ответ."""
        with patch.object(bot_module, "openai_client") as mock_openai:
            with patch.object(bot_module, "pine_client") as mock_pine:
                mock_openai.embeddings.create.return_value = MagicMock(
                    data=[MagicMock(embedding=mock_embedding)]
                )
                mock_openai.chat.completions.create.return_value = mock_openai_response
                mock_pine.describe_stats.return_value = MagicMock(total_vector_count=0)
                mock_pine.query.return_value = {"matches": []}

                update = MagicMock()
                update.effective_user = MagicMock(id=999)
                update.message = MagicMock(text="Что ты умеешь?")
                update.message.reply_text = AsyncMock()
                context = MagicMock()

                await bot_module.handle_message(update, context)

                update.message.reply_text.assert_called()
                # Вызов с ответом от LLM (не сообщение об ошибке)
                call_args = update.message.reply_text.call_args[0][0]
                assert "Тестовый ответ бота" in call_args

    async def test_handle_message_empty_text_does_nothing(self, bot_module):
        """Пустое сообщение не обрабатывается."""
        update = MagicMock()
        update.effective_user = MagicMock(id=999)
        update.message = MagicMock(text="")
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot_module.handle_message(update, context)
        update.message.reply_text.assert_not_called()

    async def test_handle_message_replies_on_error(self, bot_module):
        """При исключении пользователю отправляется сообщение об ошибке."""
        with patch.object(bot_module, "get_embedding", side_effect=RuntimeError("API error")):
            update = MagicMock()
            update.effective_user = MagicMock(id=999)
            update.message = MagicMock(text="Вопрос")
            update.message.reply_text = AsyncMock()
            context = MagicMock()

            await bot_module.handle_message(update, context)

            update.message.reply_text.assert_called_once()
            call_args = update.message.reply_text.call_args[0][0]
            assert "ошибка" in call_args.lower() or "попробуйте" in call_args.lower()

    async def test_clear_indexes_denied_without_allowlist(self, bot_module):
        """Без CLEAR_INDEXES_ALLOWED_USER_IDS кнопка очистки отклоняется."""
        bot_module.USER_MODE.clear()
        bot_module.CLEAR_INDEXES_ALLOWED_USER_IDS.clear()

        update = MagicMock()
        update.effective_user = MagicMock(id=999)
        update.message = MagicMock(text=bot_module.BTN_CLEAR)
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot_module.handle_message(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "недоступна" in call_args.lower() or "CLEAR_INDEXES" in call_args

    async def test_clear_indexes_confirm_and_execute(self, bot_module):
        """После «ДА» вызывается очистка обоих индексов."""
        bot_module.USER_MODE.clear()
        bot_module.CLEAR_INDEXES_ALLOWED_USER_IDS.clear()
        bot_module.CLEAR_INDEXES_ALLOWED_USER_IDS.add(999)

        with patch.object(bot_module, "clear_all_pinecone_indexes", MagicMock()) as mock_clear:
            update1 = MagicMock()
            update1.effective_user = MagicMock(id=999)
            update1.message = MagicMock(text=bot_module.BTN_CLEAR)
            update1.message.reply_text = AsyncMock()
            await bot_module.handle_message(update1, MagicMock())

            update2 = MagicMock()
            update2.effective_user = MagicMock(id=999)
            update2.message = MagicMock(text="ДА")
            update2.message.reply_text = AsyncMock()
            await bot_module.handle_message(update2, MagicMock())

            mock_clear.assert_called_once()
            update2.message.reply_text.assert_called()
            final_msg = update2.message.reply_text.call_args[0][0]
            assert "очищен" in final_msg.lower()
