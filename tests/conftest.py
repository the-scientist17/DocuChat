import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from docuchat.store import get_vectorstore

PAGES = [
    "The office is open Monday to Friday from 9am to 6pm.",
    "Employees get 24 days of paid leave per year. Unused leave carries over.",
]


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "handbook.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for text in PAGES:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return path


@pytest.fixture
def store(tmp_path):
    # fake embeddings so tests run offline without an API key
    return get_vectorstore(DeterministicFakeEmbedding(size=64), persist_dir=tmp_path / "db")
