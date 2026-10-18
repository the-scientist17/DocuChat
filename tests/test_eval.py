import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docuchat.qa import NOT_FOUND, Answer
from eval.run_eval import judge, on_expected_page, write_report

ITEM = {"id": 1, "question": "How long is leave?", "answer": "24 days", "source": "hr.pdf", "pages": [2]}
UNANSWERABLE = {"id": 2, "question": "CEO salary?", "answer": "", "source": None, "pages": []}


def test_on_expected_page():
    right = Document(page_content="", metadata={"source": "hr.pdf", "page": 2})
    wrong = Document(page_content="", metadata={"source": "hr.pdf", "page": 5})
    assert on_expected_page([wrong, right], ITEM)
    assert not on_expected_page([wrong], ITEM)


def test_judge_uses_llm_verdict():
    answer = Answer(text="It is 24 days [1].")
    assert judge(FakeListChatModel(responses=["CORRECT"]), ITEM, answer)
    assert not judge(FakeListChatModel(responses=["INCORRECT"]), ITEM, answer)


def test_unanswerable_needs_refusal():
    llm = FakeListChatModel(responses=["unused"])
    assert judge(llm, UNANSWERABLE, Answer(text=NOT_FOUND))
    assert not judge(llm, UNANSWERABLE, Answer(text="The CEO earns $1M."))


def test_question_set_is_well_formed():
    questions = json.loads((Path(__file__).parent.parent / "eval" / "questions.json").read_text())
    assert len(questions) == 20
    assert len({q["id"] for q in questions}) == 20
    for q in questions:
        assert q["question"] and q["answer"]
        if q["pages"]:
            assert q["source"] in {"rag_paper.pdf", "nist_ai_rmf.pdf"}


def test_report_has_scores():
    rows = [
        {"id": 1, "question": "q1", "answerable": True, "retrieval_hit": True, "citation_ok": False, "correct": True},
        {"id": 2, "question": "q2", "answerable": False, "retrieval_hit": None, "citation_ok": None, "correct": True},
    ]
    report = write_report(rows, k=4)
    assert "| Answer accuracy (all questions) | 2/2 (100%) |" in report
    assert "| Citation accuracy | 0/1 (0%) |" in report
