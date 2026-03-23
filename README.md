# Телеграм‑бот с памятью на Pinecone

Умный телеграм‑бот‑помощник с двумя режимами векторного поиска:
- авто-консультант по автомобилям (`PINECONE_INDEX_NAME`);
- RAG-поиск по пользовательской базе знаний (`PINECONE_RAG_INDEX_NAME`).

Также поддерживаются tools-функции для котиков GIF (`https://cataas.com/cat/gif`) и добавление знаний из URL-страниц.

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
Скриншоты работы бота размещены в директории screenshots/

После запуска бот работает в режиме long polling. В Telegram: команда `/start`, затем обычные сообщения или фразы вида «Запомни …» / «Важно …».

Кнопки бота:
- `Авто-консультант` — поиск по автомобильной памяти (`PINECONE_INDEX_NAME`);
- `Добавить в базу знаний` — отправьте URL, и страница будет добавлена в RAG-базу;
- `Поиск по базе знаний` — вопросы по добавленной базе знаний (`PINECONE_RAG_INDEX_NAME`);
- `Очистить индексы` — удаление **всех** векторов в обоих индексах Pinecone (только для `user_id` из `CLEAR_INDEXES_ALLOWED_USER_IDS` в `.env`); после нажатия нужно подтвердить сообщением «ДА» или отменить «Нет».

Запрос котиков без кнопки:
- если пользователь пишет фразы вроде `покажи котика`, `пришли картинку котика`, бот отправляет случайную GIF с API `https://cataas.com/cat/gif`.

---

## Структура проекта

| Файл / папка   | Назначение |
|----------------|------------|
| `bot.py`       | Точка входа, логика Telegram-кнопок и маршрутизация запросов по режимам (авто/база знаний/котики) |
| `rag_agent.py` | Класс `RAGAgent`: парсинг URL, чанкинг, embedding, запись/поиск в RAG-индексе, tools (`@tool`) |
| `pine.py`      | Клиент Pinecone: подключение, upsert, query, delete, describe_stats |
| `data/data.txt`| Начальные факты для авто-консультанта |
| `index_data.py` | Отдельный процесс индексации (запуск вручную) |
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
| `PINECONE_RAG_INDEX_NAME`| Нет | Имя индекса RAG-агента (по умолчанию `telegram-bot-rag`) |
| `PINECONE_REGION`    | Нет | Регион serverless (по умолчанию `us-east-1`) |
| `RAG_INIT_FROM_DATA_FILE` | Нет | Автоинициализация RAG из `data/data.txt` при пустом индексе (`true`/`false`, по умолчанию `false`) |
| `RUN_INDEXING_ON_START` | Нет | Выполнять индексацию при старте `bot.py` (`true`/`false`, по умолчанию `false`) |
| `PINECONE_STARTUP_CHECK` | Нет | Перед long polling выполнять тестовые запросы к обоим индексам Pinecone (`true` по умолчанию) |
| `PINECONE_STRICT_STARTUP` | Нет | Если `true` и проверка не прошла — завершить процесс с кодом 1 (по умолчанию `false`, бот всё равно стартует с предупреждением в логе) |
| `LOG_LEVEL`          | Нет | Уровень логирования: `DEBUG`, `INFO` (по умолчанию), `WARNING`, `ERROR` |
| `LOG_PINECONE_TOP_K` | Нет | Сколько top-результатов Pinecone выводить в логах поиска (по умолчанию `3`) |
| `CLEAR_INDEXES_ALLOWED_USER_IDS` | Нет | Через запятую: Telegram `user_id`, которым разрешена кнопка «Очистить индексы». Если пусто — очистка недоступна |
| `EMBEDDING_MODEL`    | Нет | Модель эмбеддингов (по умолчанию `text-embedding-3-small`) |
| `CHAT_MODEL`         | Нет | Модель чата (по умолчанию `gpt-4o-mini`) |

---



## Как это устроено

1. **Режим `Авто-консультант`**  
   Использует индекс `PINECONE_INDEX_NAME`: факты из `data/data.txt` + пользовательские «Важно/Запомни». Перед ответом подтягивается top‑10 воспоминаний.

   При ненулевом значении LOG_PINECONE_TOP_K в лог выводится информация вида:

   ```
   2026-03-23 20:21:42 [INFO] __main__: Auto Pinecone top1 score=0.508605897 text=Аудиосистемы в Mercedes-Benz могут быть разработаны совместно с премиальными брендами.
   2026-03-23 20:21:42 [INFO] __main__: Auto Pinecone top2 score=0.463970631 text=Mercedes-Benz уделяет внимание шумоизоляции и комфортному уровню звука в салоне.
   2026-03-23 20:21:42 [INFO] __main__: Auto Pinecone top3 score=0.424352854 text=Volkswagen уделяет внимание уровню шумоизоляции в своих моделях.
   ```

2. **Режим `Добавить в базу знаний`**  
   Пользователь отправляет URL. `RAGAgent` парсит страницу, превращает текст в чанки, делает embedding и сохраняет в `PINECONE_RAG_INDEX_NAME`.

3. **Режим `Поиск по базе знаний`**  
   Вопрос пользователя идет в embedding, далее поиск top‑10 в `PINECONE_RAG_INDEX_NAME`, после чего LLM формирует ответ по найденному контексту.

4. **Индексация как отдельный процесс (рекомендуется)**  
   Для заполнения индексов запускайте:

   ```bash
   python index_data.py
   ```

   По умолчанию `bot.py` не индексирует данные на старте (`RUN_INDEXING_ON_START=false`).

5. **Tools-функции**  
   В `rag_agent.py` есть функции с декоратором `@tool`, включая получение случайного GIF котика с `https://cataas.com/cat/gif`.

6. **Проверка интеграции с Pinecone (тестовые запросы)**  
   Файл `data/tutor.txt` (если был) **не** проверял Pinecone — только локальный чанкинг. Доступность Pinecone проверяется так:
   - при **`python bot.py`**: после опциональной индексации вызываются **`PineconeClient.integration_check()`** для авто-индекса (`describe_index_stats` + тестовый `query`) и **`RAGAgent.health_check()`** для RAG-индекса (embedding + `similarity_search`). Успех виден в логах (`Pinecone: проверка индекса … OK`, `Pinecone RAG: проверка индекса … OK`). Отключить: `PINECONE_STARTUP_CHECK=false`. Жёстко падать при ошибке: `PINECONE_STRICT_STARTUP=true`.
   - при **`python rag_agent.py`**: опционально `validate_on_tutor_file()` (если есть файл) + `health_check()` для RAG.

7. **Защита от prompt injection**  
   Контекст оборачивается в `<context>...</context>`, а system prompt явно требует трактовать контекст как данные и игнорировать инструкции внутри него.

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

## Тестирование

Используются [pytest](https://pytest.org/), [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) и [pytest-cov](https://pytest-cov.readthedocs.io/). API (OpenAI, Pinecone, Telegram) в тестах замоканы.

```bash
# Все тесты
pytest

# С выводом покрытия по основным модулям
pytest --cov=bot --cov=pine --cov=rag_agent --cov-report=term-missing

# Только тесты бота
pytest tests/test_bot.py -v
```

- **tests/test_bot.py** — тесты для `extract_memory_text`, обработчиков `/start` и сообщений (с моками).
- **tests/test_pine.py** — тесты для `PineconeClient`: инициализация без ключей, upsert, query, delete, describe_stats.
- **tests/test_rag_agent.py** — тесты для `RAGAgent`: tools, чанкинг, ingest URL, поиск.

---

## Лицензия

Проект распространяется под лицензией MIT — см. файл [LICENSE](LICENSE).

---

## Участие в разработке

Предложения и правки приветствуются. Краткие правила — в [CONTRIBUTING.md](CONTRIBUTING.md).
