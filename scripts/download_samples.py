"""Download the sample PDFs used by the evaluation into data/."""

import urllib.request
from pathlib import Path

SAMPLES = {
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401",
    "nist_ai_rmf.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in SAMPLES.items():
        target = DATA_DIR / name
        if target.exists():
            print(f"{name} already downloaded")
            continue
        print(f"Downloading {name}...")
        request = urllib.request.Request(url, headers={"User-Agent": "docuchat"})
        with urllib.request.urlopen(request) as resp:
            target.write_bytes(resp.read())
    print("Done. Now run: python -m docuchat.ingest")


if __name__ == "__main__":
    main()
