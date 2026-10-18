from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docuchat.loader import load_pdf
from docuchat.qa import NOT_FOUND, DocuChat, extract_citations, format_context
from docuchat.store import index_documents

DOCS = [
    Document(page_content="Leave is 24 days.", metadata={"source": "hr.pdf", "page": 2}),
    Document(page_content="Office opens at 9am.", metadata={"source": "hr.pdf", "page": 1}),
]


def test_format_context_numbers_chunks():
    context = format_context(DOCS)
    assert context.startswith("[1] hr.pdf, page 2\nLeave is 24 days.")
    assert "[2] hr.pdf, page 1" in context


def test_extract_citations_ignores_bad_numbers():
    cited = extract_citations("You get 24 days [1]. See also [1][7].", DOCS)
    assert list(cited) == [1]
    assert cited[1].metadata["page"] == 2


def test_ask_returns_cited_sources(store, sample_pdf):
    index_documents(store, load_pdf(sample_pdf))
    llm = FakeListChatModel(responses=["Employees get 24 days of paid leave [1]."])
    bot = DocuChat(store=store, llm=llm, k=2)

    answer = bot.ask("How much leave do employees get?")

    assert answer.found
    assert len(answer.retrieved) == 2
    assert list(answer.cited) == [1]


def test_not_found_answer(store, sample_pdf):
    index_documents(store, load_pdf(sample_pdf))
    bot = DocuChat(store=store, llm=FakeListChatModel(responses=[NOT_FOUND]))

    answer = bot.ask("What is the CEO's salary?")

    assert not answer.found
    assert answer.cited == {}


def test_empty_index_skips_llm(store):
    bot = DocuChat(store=store, llm=FakeListChatModel(responses=["should not be called"]))
    assert bot.ask("anything").text == NOT_FOUND
