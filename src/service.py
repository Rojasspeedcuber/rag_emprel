from dataclasses import dataclass, field

from src.ingest import Corpus
from src.llm import answer_question
from src.retriever import retrieve


@dataclass(frozen=True)
class AnswerResult:
    content: str
    sources: list[dict] = field(default_factory=list)


def answer(corpus: Corpus, question: str, history: list[dict] | None = None) -> AnswerResult:
    """Orquestra recuperação e geração. Sem evidência suficiente, recusa antes do LLM."""
    retrieved, fallback = retrieve(corpus, question)
    if fallback is not None:
        return AnswerResult(content=fallback)
    content = answer_question(question, retrieved, history=history)
    sources = [
        {
            "source": item.chunk.source,
            "page": item.chunk.page,
            "text": item.chunk.text,
        }
        for item in retrieved
    ]
    return AnswerResult(content=content, sources=sources)
