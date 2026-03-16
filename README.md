# Телеграм‑бот с памятью на Pinecone

Умный телеграм‑бот‑помощник с векторной памятью: запоминает важные сообщения в Pinecone, перед каждым ответом подтягивает топ‑10 релевантных воспоминаний и формирует ответ через OpenAI.

---

## Требования

- **Python** 3.10+ (рекомендуется 3.10 или 3.11)
- Аккаунты и ключи:
  - [Telegram Bot Token](https://core.telegram.org/bots#creating-a-new-bot) (BotFather)
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Pinecone](https://www.pinecone.io/) — API Key и индекс (serverless, dimension 1536)

---

## Быстрый старт

```bash
# Клонирование (подставьте свой GitHub-username вместо YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/embedding_pinecone_tg_bot.git
cd embedding_pinecone_tg_bot

# Виртуальное окружение (рекомендуется)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Зависимости
pip install -r requirements.txt

# Конфигурация: скопировать пример и заполнить ключи
cp .env.example .env
# Отредактировать .env — указать TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Запуск бота
python bot.py
```

После запуска бот работает в режиме long polling. В Telegram: команда `/start`, затем обычные сообщения или фразы вида «Запомни …» / «Важно …».

---

## Структура проекта

| Файл / папка   | Назначение |
|----------------|------------|
| `bot.py`       | Точка входа, логика бота: embedding запросов, поиск в Pinecone, вызов OpenAI, сохранение «важных» сообщений |
| `pine.py`      | Клиент Pinecone: подключение, upsert, query, delete, describe_stats |
| `data/data.txt`| Начальные факты (по умолчанию 100 фраз про Volkswagen и Mercedes-Benz), загружаются при первом запуске в пустой индекс |
| `.env`         | Секреты и настройки (не коммитить) |
| `.env.example` | Шаблон переменных окружения |
| `requirements.txt` | Зависимости Python |

---

## Переменные окружения

Скопируйте `.env.example` в `.env` и задайте значения.

| Переменная | Обязательно | Описание |
|------------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Да | Токен бота от [@BotFather](https://t.me/BotFather) |
| `OPENAI_API_KEY`     | Да | Ключ API OpenAI |
| `PINECONE_API_KEY`   | Да | API Key из Pinecone Console |
| `PINECONE_INDEX_NAME`| Да | Имя индекса (например `telegram-bot-memories`) |
| `PINECONE_REGION`    | Нет | Регион serverless (по умолчанию `us-east-1`) |
| `EMBEDDING_MODEL`    | Нет | Модель эмбеддингов (по умолчанию `text-embedding-3-small`) |
| `CHAT_MODEL`         | Нет | Модель чата (по умолчанию `gpt-4o-mini`) |

---

## Как это устроено

1. **Инициализация при первом запуске**  
   Если индекс пустой, читается `data/data.txt`: каждая строка — отдельная фраза. Для каждой фразы считается embedding (OpenAI), в Pinecone сохраняется вектор с `id` = порядковый номер и `metadata.text` = фраза.

2. **Ответ на сообщение**  
   Текст пользователя → embedding → поиск в Pinecone (top‑10) → тексты из `metadata.text` попадают в system‑prompt → запрос + контекст уходят в `gpt-4o-mini` → ответ отправляется в чат.

3. **Запоминание**  
   Если сообщение начинается с триггера («важно», «запомни», «запомнить», «запиши», «это важно»), из текста выделяется содержание, делается embedding и запись в Pinecone с `id` вида `user_memory_<N>`.

Подробнее про формат данных и триггеры — в комментариях в `bot.py` и `pine.py`.

---

## Устранение неполадок

- **`ImportError: cannot import name 'Pinecone'`**  
  В проекте используются импорты из подмодулей (`pinecone.pinecone`, `pinecone.db_control`, `pinecone.db_data`). Убедитесь, что установлен пакет `pinecone` (не `pinecone-client`):  
  `pip install "pinecone>=5.0.0"`

- **`FileNotFoundError: __version__`**  
  Повреждённая установка Pinecone. Попробуйте:  
  `pip install --force-reinstall pinecone`

- **Ошибки Pinecone при создании индекса**  
  Проверьте имя индекса (только lowercase, цифры, дефисы), регион и квоты в [Pinecone Console](https://app.pinecone.io/). Размерность индекса должна быть **1536** для `text-embedding-3-small`.

- **Бот не отвечает в Telegram**  
  Проверьте токен и что бот не запущен в другом процессе. Убедитесь, что в `.env` нет лишних пробелов вокруг `=` и значений.

---

## Лицензия

Проект распространяется под лицензией MIT — см. файл [LICENSE](LICENSE).

---

## Участие в разработке

Предложения и правки приветствуются. Краткие правила — в [CONTRIBUTING.md](CONTRIBUTING.md).
