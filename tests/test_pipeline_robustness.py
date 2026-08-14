"""
Bonus tests: input validation, grounding, mixed-corpus filtering, CLI.

These do not load the local LLM. Run with: pytest tests/ -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.pipeline import (
    EMPTY_QUESTION_MESSAGE,
    FALLBACK_ANSWER,
    IN_SCOPE_FILENAMES,
    ask_question,
    format_result,
    parse_args,
    resolve_data_dir,
    validate_data_dir,
)


def _doc(text: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(page_content=text, metadata={"source": source})


def _store_with(docs: list) -> MagicMock:
    store = MagicMock()
    store.similarity_search.return_value = docs
    return store


class TestInputValidation:
    def test_empty_question_does_not_call_retriever_or_llm(self):
        store = MagicMock()
        llm = MagicMock()
        result = ask_question(store, llm, "   ")
        store.similarity_search.assert_not_called()
        llm.assert_not_called()
        assert result["answer"] == EMPTY_QUESTION_MESSAGE
        assert result["sources"] == []

    def test_none_like_empty_string(self):
        store = MagicMock()
        llm = MagicMock()
        result = ask_question(store, llm, "")
        assert result["answer"] == EMPTY_QUESTION_MESSAGE
        llm.assert_not_called()


class TestGrounding:
    def test_no_documents_returns_fallback(self):
        store = _store_with([])
        llm = MagicMock()
        result = ask_question(store, llm, "What is the meaning of life?")
        llm.assert_not_called()
        assert result["answer"] == FALLBACK_ANSWER
        assert result["sources"] == []

    def test_out_of_scope_only_docs_are_dropped(self):
        store = _store_with(
            [
                _doc("AcmeCloud Pro is $12 per user.", "data/product_faq.txt"),
                _doc("Employees receive 20 days of PTO.", "data/company_handbook.txt"),
            ]
        )
        llm = MagicMock()
        result = ask_question(store, llm, "How much PTO do employees get?")
        llm.assert_not_called()
        assert result["answer"] == FALLBACK_ANSWER
        assert result["sources"] == []

    def test_agency_docs_are_kept_and_passed_to_llm(self):
        growth = _doc("GROWTH PACKAGE — $5,500/month.", "data/pricing.txt")
        store = _store_with(
            [
                _doc("AcmeCloud Free plan includes 5GB.", "data/product_faq.txt"),
                growth,
                _doc("SEO includes keyword research.", "data/services.txt"),
            ]
        )
        llm = MagicMock(return_value=[{"generated_text": "The Growth package costs $5,500 per month."}])
        result = ask_question(store, llm, "How much does Growth cost?")
        llm.assert_called_once()
        assert "5,500" in result["answer"]
        assert growth.page_content in result["sources"]
        assert all(
            "AcmeCloud" not in source for source in result["sources"]
        ), "Distractor corpus should not be cited"

    def test_returns_at_most_three_sources(self):
        docs = [
            _doc(f"chunk {i}", f"data/{name}")
            for i, name in enumerate(
                ["pricing.txt", "services.txt", "faq.txt", "pricing.txt", "services.txt"]
            )
        ]
        store = _store_with(docs)
        llm = MagicMock(return_value=[{"generated_text": "ok"}])
        result = ask_question(store, llm, "What do you offer?")
        assert len(result["sources"]) == 3

    def test_empty_generation_falls_back(self):
        store = _store_with([_doc("Starter is $2,500/month.", "data/pricing.txt")])
        llm = MagicMock(return_value=[{"generated_text": "  "}])
        result = ask_question(store, llm, "How much is Starter?")
        assert result["answer"] == FALLBACK_ANSWER


class TestPromptWiring:
    def test_prompt_includes_context_and_question(self):
        store = _store_with([_doc("PPC management fee is 15%.", "data/services.txt")])
        llm = MagicMock(return_value=[{"generated_text": "15% of ad spend"}])
        ask_question(store, llm, "What is the PPC fee?")
        prompt = llm.call_args[0][0]
        assert "PPC management fee is 15%." in prompt
        assert "What is the PPC fee?" in prompt
        assert "Context:" in prompt


class TestCliHelpers:
    def test_parse_query_flag(self):
        args = parse_args(["--query", "How much is Growth?"])
        assert args.query == "How much is Growth?"

    def test_parse_short_query_flag(self):
        args = parse_args(["-q", "Do you offer SEO?"])
        assert args.query == "Do you offer SEO?"

    def test_parse_data_dir(self):
        args = parse_args(["--data-dir", "/tmp/docs"])
        assert args.data_dir == "/tmp/docs"

    def test_validate_missing_directory(self, tmp_path):
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_data_dir(missing)

    def test_validate_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .txt"):
            validate_data_dir(tmp_path)

    def test_validate_accepts_txt_directory(self, tmp_path):
        (tmp_path / "faq.txt").write_text("hello")
        validate_data_dir(tmp_path)

    def test_resolve_data_dir_default_is_repo_data(self):
        path = resolve_data_dir(None)
        assert path.name == "data"
        assert path.is_dir()

    def test_format_result_includes_sources_and_answer(self):
        rendered = format_result(
            {
                "answer": "The Growth package costs $5,500 per month.",
                "sources": ["GROWTH PACKAGE — $5,500/month. Best for scaling businesses."],
            }
        )
        assert "📄 Sources:" in rendered
        assert "GROWTH PACKAGE" in rendered
        assert "💬 Answer:" in rendered
        assert "$5,500" in rendered

    def test_in_scope_filenames_match_assignment_docs(self):
        assert IN_SCOPE_FILENAMES == frozenset({"services.txt", "pricing.txt", "faq.txt"})
