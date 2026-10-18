"""Run the 20-question test set and report how often DocuChat gets it right.

    python -m eval.run_eval
    python -m eval.run_eval --k 6

Make sure the sample PDFs are indexed first (python -m docuchat.ingest).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from docuchat import config
from docuchat.qa import Answer, DocuChat

EVAL_DIR = Path(__file__).parent

JUDGE_PROMPT = """You are grading a document Q&A system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Is the system answer correct? It is correct if it contains the key facts of the
reference answer and does not contradict it. Extra detail is fine.
Reply with one word: CORRECT or INCORRECT."""


def on_expected_page(docs: list[Document], item: dict) -> bool:
    return any(
        d.metadata["source"] == item["source"] and d.metadata["page"] in item["pages"]
        for d in docs
    )


def judge(llm, item: dict, answer: Answer) -> bool:
    """Unanswerable questions should get a refusal; the rest are graded by an LLM."""
    if not item["pages"]:
        return not answer.found
    if not answer.found:
        return False

    reply = llm.invoke(JUDGE_PROMPT.format(
        question=item["question"], reference=item["answer"], answer=answer.text
    )).content
    return reply.strip().upper().startswith("CORRECT")


def evaluate(bot: DocuChat, judge_llm, questions: list[dict]) -> list[dict]:
    rows = []
    for item in questions:
        answer = bot.ask(item["question"])
        answerable = bool(item["pages"])
        row = {
            "id": item["id"],
            "question": item["question"],
            "answer": answer.text,
            "answerable": answerable,
            "retrieval_hit": on_expected_page(answer.retrieved, item) if answerable else None,
            "citation_ok": on_expected_page(list(answer.cited.values()), item) if answerable else None,
            "correct": judge(judge_llm, item, answer),
        }
        rows.append(row)
        print(f"[{row['id']:>2}] {'PASS' if row['correct'] else 'FAIL'}  {item['question']}")
    return rows


def rate(rows: list[dict], key: str) -> tuple[int, int]:
    values = [r[key] for r in rows if r[key] is not None]
    return sum(values), len(values)


def mark(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def write_report(rows: list[dict], k: int) -> str:
    lines = [
        "# DocuChat evaluation results",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M} | chat model: `{config.CHAT_MODEL}` | "
        f"embeddings: `{config.EMBEDDING_MODEL}` | top-k: {k}",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for label, key in [
        ("Answer accuracy (all questions)", "correct"),
        (f"Retrieval hit rate @ {k}", "retrieval_hit"),
        ("Citation accuracy", "citation_ok"),
    ]:
        hits, total = rate(rows, key)
        lines.append(f"| {label} | {hits}/{total} ({hits / total:.0%}) |")

    lines += ["", "| # | Question | Retrieved right page | Cited right page | Correct |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['question']} | {mark(r['retrieval_hit'])} | "
            f"{mark(r['citation_ok'])} | {mark(r['correct'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--questions", default=EVAL_DIR / "questions.json")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    bot = DocuChat(k=args.k)
    rows = evaluate(bot, ChatOpenAI(model=config.CHAT_MODEL), questions)

    report = write_report(rows, args.k)
    (EVAL_DIR / "results.md").write_text(report)
    (EVAL_DIR / "results.json").write_text(json.dumps(rows, indent=2))

    print("\n" + report.split("\n\n| #")[0])
    print("\nFull report saved to eval/results.md")


if __name__ == "__main__":
    main()
