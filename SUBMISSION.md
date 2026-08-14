# Submission notes

This file is the design record for the take-home: what I built, why, and what I
deliberately left out.

The assignment is a local RAG chatbot. I treated it as a **small production
workflow**, not a notebook: retrieve evidence, decide whether that evidence is
usable, then generate — and refuse when it is not.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
# CPU wheel is enough for this assignment and much smaller than the CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

pytest tests/ -v

# Interactive
python -m src.pipeline

# Single question
python -m src.pipeline --query "How much does the Growth package cost?"
python -m src.pipeline -q "Can I cancel early?" --data-dir data
```

First run downloads two models (~1.2GB), then they are cached.

## Architecture

The knowledge base (`src/knowledge_base.py`) is used as given. The response
layer in `src/pipeline.py` is a two-tool loop with a deterministic policy:

```
question
    │
    ├─ empty / whitespace ──────────────────────────────► "please ask a question"
    │
    ▼
retriever (similarity_search, k=8)
    │
    ├─ drop files outside {services, pricing, faq}.txt
    ├─ keep top 3 in-scope chunks
    │
    ├─ no usable evidence ──────────────────────────────► fail closed
    │
    ▼
prompt (provided template, context truncated to the 512-token window)
    │
    ▼
generator (flan-t5-base)
    │
    └─ empty model output ──────────────────────────────► fail closed
```

That is the whole agent. Retrieval and generation are explicit tools;
orchestration is code, not another LLM call. With a 250M-parameter CPU model,
an LLM router or self-critique loop would add latency and failure modes without
reliable gains.

## Design decisions

**Fail closed.** If the user asks nothing, retrieval returns nothing, or the
only hits are out-of-scope, we do not call the LLM. A weak local model will
happily invent an answer from an empty prompt; skipping generation is the
cheapest groundedness check.

**Mixed corpus.** `data/` contains the three agency files from the brief *and*
two unrelated Acme Corp documents (`product_faq.txt`, `company_handbook.txt`).
`DirectoryLoader` indexes all of them. A naive `k=3` search can cite employee
PTO or AcmeCloud pricing in a marketing-agency chatbot. I retrieve extra
candidates, keep only the in-scope filenames, then take top 3. Off-topic
questions (Acme product, handbook, weather) return the fallback instead of a
confident wrong answer.

I did not modify `knowledge_base.py`. Filtering happens in the response layer,
which is the right place for product policy.

**Protect the question from truncation.** `get_llm()` encodes with
`max_length=512`. Three 500-character chunks plus the template can push the
question off the end of the encoder window. Context is capped so the question
and the "Answer:" cue still fit.

**Use the provided prompt.** I did not rewrite the template. Instruction
following on flan-t5-base is limited; the high-leverage work is retrieval
scope and refusal, not prompt poetry.

**CLI is a thin adapter.** `--query` is for scripts and reviewers; the
interactive loop handles `quit` / `exit` / `q`, empty input, EOF, and
Ctrl+C. Missing or empty `--data-dir` fails with a clear error before model
load.

**Typed, testable helpers.** `ask_question` still returns `{answer, sources}`
exactly as specified. CLI parsing, data-dir checks, and the retrieve/ground
policy are unit-tested with mocks so they do not depend on model download.

## What I did not build

These are common take-home extras. I left them out on purpose:

| Idea | Why not here |
| --- | --- |
| Chat memory | The encoder is 512 tokens. History would evict the retrieved docs, which are the only source of truth. |
| LLM-as-router / multi-agent | flan-t5-base is a poor classifier; a filename allowlist is the correct router for this corpus. |
| LangChain `RetrievalQA` | The assignment is to wire retrieve → prompt → generate. An opaque chain hides the control flow the tests (and a reviewer) need to see. |
| Web UI / API | Out of scope for a CLI assignment; would steal time from retrieval quality. |
| Fine-tuning | Wrong tool. The failures here are retrieval and grounding, not model capacity. |

Knowing where not to add moving parts is part of orchestrating a workflow.

## If this shipped

The same skeleton scales. I would change the tools, not the shape:

1. **Namespaced indexes** — agency vs. internal docs as separate collections, not a post-filter.
2. **Hybrid retrieval + rerank** — BM25 plus embeddings, then a cross-encoder on the top 20. Better than raising `k`.
3. **Eval harness** — a golden set of (question, must-cite, must-not-cite, refusal) cases in CI, plus faithfulness checks when a larger model is available.
4. **Tracing** — log query, chunk ids, scores, and refusal reason. You cannot improve a RAG system you cannot see.
5. **Real agent loop** — only once there are real tools (CRM lookup, package quote, ticket). A router would choose `retrieve_docs` vs `create_lead` vs `refuse`; each tool would still fail closed.

That last step is how I think about agentic systems at Gesture's shape of problem: personalization and ops workflows with tools, not an unconstrained chat model.

## Tests

```bash
pytest tests/ -v
```

- `tests/test_pipeline.py` — original structure, retrieval, and generation checks (loads models).
- `tests/test_pipeline_robustness.py` — empty input, distractor-corpus filter, refusal, CLI flags, missing data dir (no model load).
