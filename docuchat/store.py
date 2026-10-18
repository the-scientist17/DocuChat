from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docuchat import config


def get_vectorstore(embeddings: Embeddings | None = None, persist_dir=None) -> Chroma:
    """Open (or create) the Chroma collection on disk."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        persist_directory=str(persist_dir or config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def split_documents(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks. Each chunk keeps its page's metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(pages)


def chunk_id(chunk: Document) -> str:
    meta = chunk.metadata
    return f"{meta['source']}:p{meta['page']}:{meta['start_index']}"


def index_documents(store: Chroma, pages: list[Document]) -> int:
    """Chunk, embed and save pages. Re-indexing a file replaces its old chunks."""
    if not pages:
        return 0

    for source in {p.metadata["source"] for p in pages}:
        remove_source(store, source)

    chunks = split_documents(pages)
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return len(chunks)


def remove_source(store: Chroma, source: str) -> None:
    store.delete(where={"source": source})


def list_sources(store: Chroma) -> dict[str, int]:
    """Return {file name: number of chunks} for everything in the index."""
    counts: dict[str, int] = {}
    for meta in store.get(include=["metadatas"])["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return dict(sorted(counts.items()))
