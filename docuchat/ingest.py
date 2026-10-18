"""Index PDFs from the command line.

    python -m docuchat.ingest                # everything in data/
    python -m docuchat.ingest path/to/file.pdf
"""

import argparse
from pathlib import Path

from docuchat import config
from docuchat.loader import load_pdf
from docuchat.store import get_vectorstore, index_documents, list_sources


def find_pdfs(paths: list[str]) -> list[Path]:
    pdfs = []
    for p in map(Path, paths):
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdfs.append(p)
    return pdfs


def main():
    parser = argparse.ArgumentParser(description="Add PDFs to the DocuChat index")
    parser.add_argument("paths", nargs="*", default=[str(config.DATA_DIR)])
    args = parser.parse_args()

    pdfs = find_pdfs(args.paths)
    if not pdfs:
        print("No PDFs found. Put some in data/ or pass a path.")
        return

    store = get_vectorstore()
    for pdf in pdfs:
        pages = load_pdf(pdf)
        n_chunks = index_documents(store, pages)
        print(f"{pdf.name}: {len(pages)} pages -> {n_chunks} chunks")

    print("\nIndex now contains:")
    for source, count in list_sources(store).items():
        print(f"  {source} ({count} chunks)")


if __name__ == "__main__":
    main()
