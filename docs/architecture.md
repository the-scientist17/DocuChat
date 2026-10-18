# Architecture and design notes

This explains how a question travels through DocuChat and why things are built the way they are.

## Indexing path

```
PDF ──pypdf──> pages ──splitter──> chunks ──OpenAI──> vectors ──> ChromaDB (chroma_db/)
```

**One Document per page (`loader.py`).** Each page becomes a LangChain `Document` with
`{"source": "file.pdf", "page": 3}` in its metadata. Splitting happens *after* this, and the
splitter copies metadata to every chunk, so each chunk always knows which page it came from.
That's what makes page citations possible. The trade-off: a sentence that crosses a page break
ends up in two chunks. For citation purposes that's acceptable.

**Cleaning.** PDF text extraction often splits words at line ends (`retrie-\nval`) and leaves
runs of spaces. `clean_text` fixes the common cases. It doesn't try to remove headers/footers.

**Chunking (`store.py`).** `RecursiveCharacterTextSplitter` tries to split on paragraphs first,
then lines, then sentences, then words, so chunks break at natural points. Settings:

- `CHUNK_SIZE = 1000` characters (~200 words). Small enough that one chunk is about one idea,
  large enough to hold a full answer. Too small loses context; too large dilutes the embedding
  and wastes prompt space.
- `CHUNK_OVERLAP = 150` so a fact sitting on a chunk boundary is fully contained in at least one chunk.

**IDs.** Each chunk gets an ID like `rag_paper.pdf:p3:1200` (file, page, character offset).
Before indexing a file, its old chunks are deleted, so re-running ingest never creates duplicates.

**Embeddings.** `text-embedding-3-small` maps each chunk to a 1536-dimension vector. Texts with
similar meaning end up close together, even when they share no words ("time off" vs "paid leave").

**Vector store.** ChromaDB runs in-process and persists to `chroma_db/`, so there's no server to
manage. The collection uses cosine distance. Internally Chroma uses an HNSW index for fast
approximate nearest-neighbour search.

## Question path

```
question ──embed──> top-k chunks ──> numbered prompt ──> LLM ──> answer + [n] ──> citations
```

**Retrieval.** `similarity_search(question, k=4)` returns the 4 closest chunks. `k` is a
trade-off: more chunks raise the chance the answer is included but add noise and cost.
The Streamlit sidebar lets you change it, and `eval/run_eval.py --k` lets you measure it.

**Prompt (`qa.py`).** The chunks are numbered and labelled with their source:

```
[1] rag_paper.pdf, page 3
<chunk text>

[2] nist_ai_rmf.pdf, page 12
<chunk text>
```

The system prompt tells the model to use only these passages, cite `[n]` after each sentence,
and reply with a fixed sentence when the answer isn't there. Two reasons for this design:

1. **Grounding.** Restricting the model to the passages, plus a clear "way out", is the main
   defence against made-up answers.
2. **Reliable citations.** Asking the model to cite short numbers is more reliable than asking it
   to copy file names and page numbers. The code then maps `[n]` back to the real metadata
   (`extract_citations`), so a page number shown to the user always comes from the index, never
   from the model. Out-of-range numbers are ignored.

**Answer object.** `DocuChat.ask` returns an `Answer` with the text, all retrieved chunks and
just the cited chunks. The UI shows the cited ones; the evaluation uses both.

## Evaluation

See `eval/run_eval.py`. For each of the 20 questions it records:

| Check | How |
|---|---|
| Retrieval hit | Any retrieved chunk has the expected file and page |
| Citation correct | Any *cited* chunk has the expected file and page |
| Answer correct | An LLM judge compares the answer to the reference answer. For the unanswerable questions, the answer must be the refusal sentence |

Keeping these separate tells you what to fix. Low retrieval hit rate means chunking or
embeddings; good retrieval with wrong answers means the prompt or model.

LLM-as-judge isn't perfect, so the per-question table in `eval/results.md` includes every
question for a quick manual check.

## Testing

`tests/` runs without network access: `DeterministicFakeEmbedding` stands in for OpenAI
embeddings and `FakeListChatModel` returns canned answers. This covers loading, chunking,
de-duplication, citation mapping, the refusal path and the eval scoring logic.

## Possible improvements

- **Hybrid search:** combine vector search with BM25 keyword search for exact terms.
- **Re-ranking:** retrieve 20 chunks, re-rank with a cross-encoder, keep the best 4.
- **Conversation memory:** rewrite follow-up questions into standalone ones before retrieval.
- **Streaming:** stream tokens to the UI instead of waiting for the full answer.
- **OCR** for scanned PDFs.
