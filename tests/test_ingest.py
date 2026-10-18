from langchain_core.documents import Document

from docuchat.loader import clean_text, load_pdf
from docuchat.store import index_documents, list_sources, remove_source, split_documents


def test_clean_text_joins_hyphenated_words():
    assert clean_text("retrie-\nval   augmented\n\n\n\ngeneration") == "retrieval augmented\n\ngeneration"


def test_load_pdf_keeps_page_numbers(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert pages[0].metadata["source"] == "handbook.pdf"
    assert "24 days" in pages[1].page_content


def test_split_keeps_metadata():
    long_page = Document(page_content="word " * 1000, metadata={"source": "a.pdf", "page": 7})
    chunks = split_documents([long_page])
    assert len(chunks) > 1
    assert all(c.metadata["page"] == 7 for c in chunks)


def test_reindexing_does_not_duplicate(store, sample_pdf):
    pages = load_pdf(sample_pdf)
    index_documents(store, pages)
    index_documents(store, pages)
    assert list_sources(store) == {"handbook.pdf": 2}


def test_remove_source(store, sample_pdf):
    index_documents(store, load_pdf(sample_pdf))
    remove_source(store, "handbook.pdf")
    assert list_sources(store) == {}
