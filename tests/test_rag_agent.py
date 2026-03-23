import os
from unittest.mock import MagicMock, patch


def _build_agent():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai",
            "PINECONE_API_KEY": "test-pine",
            "PINECONE_RAG_INDEX_NAME": "test-rag-index",
        },
        clear=False,
    ):
        with patch("rag_agent.PineconeClient") as pc_cls:
            with patch("rag_agent.OpenAI") as openai_cls:
                with patch("rag_agent.ChatOpenAI") as chat_cls:
                    with patch("rag_agent.OpenAIEmbeddings") as emb_cls:
                        with patch("rag_agent.Pinecone") as pinecone_cls:
                            with patch("rag_agent.PineconeVectorStore") as store_cls:
                                with patch("rag_agent.create_agent") as create_agent_mock:
                                    pc = MagicMock()
                                    pc.describe_stats.return_value = MagicMock(total_vector_count=0)
                                    pc_cls.return_value = pc
                                    openai = MagicMock()
                                    openai_cls.return_value = openai
                                    chat_cls.return_value = MagicMock()
                                    emb_cls.return_value = MagicMock()
                                    pinecone_cls.return_value = MagicMock(Index=MagicMock(return_value=MagicMock()))
                                    vector_store = MagicMock()
                                    store_cls.return_value = vector_store
                                    create_agent_mock.return_value = MagicMock(
                                        invoke=MagicMock(
                                            return_value={
                                                "messages": [MagicMock(content="agent answer")]
                                            }
                                        )
                                    )

                                    from rag_agent import RAGAgent

                                    agent = RAGAgent()
                                    return agent, pc, openai


def test_cat_tool_url():
    from rag_agent import get_random_cat_gif_url

    url = get_random_cat_gif_url.invoke({})
    assert url.startswith("https://cataas.com/cat/gif")


def test_chunk_text():
    agent, _, _ = _build_agent()
    text = "a" * 3000
    chunks = agent.chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(isinstance(c, str) and c for c in chunks)


def test_ingest_url_adds_chunks():
    agent, _, _ = _build_agent()
    with patch("rag_agent.requests.get") as rq:
        response = MagicMock()
        response.text = "<html><body><h1>Title</h1><p>Hello world from page</p></body></html>"
        response.raise_for_status.return_value = None
        rq.return_value = response
        with patch.object(agent, "add_text_chunks", return_value=2) as add_chunks:
            added = agent.ingest_url("https://example.com")
            assert added == 2
            add_chunks.assert_called_once()


def test_search_reads_matches_text():
    agent, _, _ = _build_agent()
    doc1 = MagicMock(page_content="chunk 1", metadata={"source": "s1"})
    doc2 = MagicMock(page_content="chunk 2", metadata={"source": "s2"})
    agent.vector_store.similarity_search.return_value = [doc1, doc2]
    found = agent.search("test query", top_k=2)
    assert found == ["chunk 1", "chunk 2"]


def test_answer_with_context_uses_agent():
    agent, _, _ = _build_agent()
    answer = agent.answer_with_context("test q", top_k=2)
    assert answer == "agent answer"


def test_initialize_knowledge_base_uses_data_txt_only():
    with patch.dict(os.environ, {"RAG_INIT_FROM_DATA_FILE": "true"}, clear=False):
        agent, _, _ = _build_agent()
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", MagicMock()) as _:
                with patch.object(agent, "add_text_chunks", return_value=5) as add_chunks:
                    # эмулируем строки data.txt
                    with patch("rag_agent.open", create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_file.__enter__.return_value = ["line1\n", "line2\n"]
                        mock_open.return_value = mock_file
                        agent.initialize_knowledge_base()
                        # Проверяем, что используется источник data/data.txt
                        add_chunks.assert_called_once()
                        assert add_chunks.call_args[0][0] == "data/data.txt"

