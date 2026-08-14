"""
Document Q&A Pipeline — retrieve, ground, then generate.

The knowledge base (loading, chunking, vector store) is already built
in knowledge_base.py. This module is the response layer: a small
orchestration loop over two tools (retriever + generator) with a
fail-closed policy when evidence is missing or out of scope.

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence, TypedDict

# Make `python src/pipeline.py` work from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
LLM = Callable[[str], list[dict[str, str]]]


def get_llm() -> LLM:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> list[dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ──────────────────────────────────────────────
# Pipeline policy
# ──────────────────────────────────────────────
# The product is a marketing-agency assistant. `data/` also contains
# unrelated Acme Corp files; those are treated as retrieval noise.
IN_SCOPE_FILENAMES = frozenset({"services.txt", "pricing.txt", "faq.txt"})
TOP_K = 3
CANDIDATE_K = 8
# flan-t5-base encodes at most 512 tokens. Budget the context so the
# question at the end of the prompt is not truncated away.
MAX_CONTEXT_CHARS = 1400
FALLBACK_ANSWER = "I don't have enough information to answer that."
EMPTY_QUESTION_MESSAGE = "Please ask a question about our services, pricing, or process."
QUIT_COMMANDS = frozenset({"quit", "exit", "q"})


class QAResult(TypedDict):
    answer: str
    sources: list[str]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm: LLM, question: str) -> QAResult:
    """Retrieve relevant chunks and generate an answer.

    Orchestration:
      1. Validate the question (empty input never hits the LLM).
      2. Retrieve candidate chunks, keep in-scope agency docs, take top 3.
      3. If nothing usable was retrieved, fail closed with the fallback.
      4. Format PROMPT_TEMPLATE, call the LLM, return answer + sources.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return {"answer": EMPTY_QUESTION_MESSAGE, "sources": []}

    documents = _retrieve_in_scope(vector_store, cleaned)
    if not documents:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = _build_context(documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=cleaned)
    answer = _generate(llm, prompt)

    return {
        "answer": answer or FALLBACK_ANSWER,
        "sources": [doc.page_content for doc in documents],
    }


def _source_filename(document) -> str:
    source = ""
    metadata = getattr(document, "metadata", None) or {}
    if isinstance(metadata, dict):
        source = str(metadata.get("source", ""))
    return Path(source).name


def _is_in_scope(document) -> bool:
    """Keep agency docs; drop distractor files. Unknown sources stay in."""
    name = _source_filename(document)
    if not name:
        return True
    return name in IN_SCOPE_FILENAMES


def _retrieve_in_scope(vector_store, question: str) -> list:
    """Search, drop out-of-scope files, return the top-k agency chunks."""
    candidates = vector_store.similarity_search(question, k=CANDIDATE_K)
    in_scope = [doc for doc in candidates if _is_in_scope(doc)]
    return in_scope[:TOP_K]


def _build_context(documents: Sequence) -> str:
    """Join chunk text, truncated so the question still fits in 512 tokens."""
    context = "\n\n".join(
        doc.page_content.strip() for doc in documents if getattr(doc, "page_content", "")
    )
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    return context[: MAX_CONTEXT_CHARS - 3].rstrip() + "..."


def _generate(llm: LLM, prompt: str) -> str:
    result = llm(prompt)
    if not result:
        return ""
    first = result[0]
    if isinstance(first, dict):
        return str(first.get("generated_text", "")).strip()
    return str(first).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Q&A chatbot for a marketing agency.",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the documents directory (default: ./data next to src/).",
    )
    return parser.parse_args(argv)


def resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return (_REPO_ROOT / "data").resolve()


def validate_data_dir(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt documents found in {data_dir}")


def format_result(result: QAResult) -> str:
    lines = ["", "📄 Sources:"]
    sources = result.get("sources") or []
    if not sources:
        lines.append("  (none)")
    else:
        for i, source in enumerate(sources, start=1):
            preview = " ".join(source.split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            lines.append(f"  {i}. {preview}")
    lines.append("")
    lines.append(f"💬 Answer: {result.get('answer', '').strip()}")
    lines.append("")
    return "\n".join(lines)


def run_query(vector_store, llm: LLM, question: str) -> QAResult:
    result = ask_question(vector_store, llm, question)
    print(format_result(result))
    return result


def run_interactive_loop(vector_store, llm: LLM) -> None:
    print("Ask about services, pricing, or process. Type 'quit' to exit.\n")
    while True:
        try:
            raw = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if raw.strip().lower() in QUIT_COMMANDS:
            print("Goodbye.")
            return
        if not raw.strip():
            print(f"\n💬 {EMPTY_QUESTION_MESSAGE}\n")
            continue

        run_query(vector_store, llm, raw)


def main(argv: list[str] | None = None) -> int:
    """Interactive Q&A loop (or a single --query)."""
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    try:
        validate_data_dir(data_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    vector_store = build_knowledge_base(str(data_dir))
    llm = get_llm()

    if args.query is not None:
        run_query(vector_store, llm, args.query)
        return 0

    run_interactive_loop(vector_store, llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
