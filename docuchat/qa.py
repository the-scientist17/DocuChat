import re
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from docuchat import config
from docuchat.store import get_vectorstore

NOT_FOUND = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions about the user's documents.

Rules:
- Use ONLY the numbered passages below. Do not use outside knowledge.
- Cite the passage numbers you used after each sentence, like [1] or [2][3].
- If the passages don't contain the answer, reply exactly: "{NOT_FOUND}"
- Keep the answer short and factual.

Passages:
{{context}}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


@dataclass
class Answer:
    text: str
    retrieved: list[Document] = field(default_factory=list)
    cited: dict[int, Document] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return NOT_FOUND.lower() not in self.text.lower()


def format_context(docs: list[Document]) -> str:
    """Number each chunk so the model can cite it as [n]."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        header = f"[{i}] {doc.metadata['source']}, page {doc.metadata['page']}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def extract_citations(text: str, docs: list[Document]) -> dict[int, Document]:
    """Map the [n] markers in the answer back to the chunks they point to."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return {n: docs[n - 1] for n in numbers if 1 <= n <= len(docs)}


class DocuChat:
    def __init__(self, store: Chroma | None = None, llm: BaseChatModel | None = None, k: int = config.TOP_K):
        self.store = store or get_vectorstore()
        self.llm = llm or ChatOpenAI(model=config.CHAT_MODEL)
        self.k = k
        self.chain = prompt | self.llm | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        return self.store.similarity_search(question, k=self.k)

    def ask(self, question: str) -> Answer:
        """Retrieve the top chunks, ask the LLM, and attach the cited sources."""
        docs = self.retrieve(question)
        if not docs:
            return Answer(text=NOT_FOUND)

        text = self.chain.invoke({"context": format_context(docs), "question": question})
        return Answer(text=text.strip(), retrieved=docs, cited=extract_citations(text, docs))
