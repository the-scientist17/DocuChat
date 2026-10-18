import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Fix the usual PDF extraction mess: broken hyphens and extra whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[Document]:
    """Read a PDF into one Document per page, keeping the file name and page number."""
    path = Path(path)
    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue  # scanned or blank page
        pages.append(
            Document(page_content=text, metadata={"source": path.name, "page": page_number})
        )

    return pages


def load_folder(folder: str | Path) -> list[Document]:
    """Load every PDF in a folder."""
    docs = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        docs.extend(load_pdf(pdf))
    return docs
