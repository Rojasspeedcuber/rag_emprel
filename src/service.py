from dataclasses import dataclass, field

from src.config import SIMILARITY_THRESHOLD
from src.ingest import Corpus
from src.llm import answer_question
from src.retriever import EMPTY_CORPUS_MESSAGE, FALLBACK_MESSAGE, retrieve


@dataclass(frozen=True)
class AnswerResult:
    content: str
    sources: list[dict] = field(default_factory=list)


def answer(corpus: Corpus, question: str, history: list[dict] | None = None) -> AnswerResult:
    """Orquestra recuperação e geração. Pergunta fora do tema é recusada antes do LLM."""
    retrieved = retrieve(corpus, question)
    if not retrieved:
        return AnswerResult(content=EMPTY_CORPUS_MESSAGE)
    # Trechos lexicais (código ou sigla exata) têm similaridade 1.0 e sempre passam.
    if max(item.similarity for item in retrieved) < SIMILARITY_THRESHOLD:
        return AnswerResult(content=FALLBACK_MESSAGE)
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
