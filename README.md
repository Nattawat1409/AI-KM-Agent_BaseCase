# AI-KM-Agent_BaseCase

Dedicate for AI personalize R&amp;D to measure by same benchmark between "Base Case" and "AI personalize"

## What's here

A working KM (Buildee) hybrid dense/sparse retrieval + generation baseline, ported from `cbm-system` (`dev` branch), against the **live** Pinecone project. Directory layout mirrors `cbm-system`'s `code/` tree exactly, so this repo and `Personalize-AI` (which copies the same structure) compare like-for-like. See `HANDOFF_NEW_ARCHITECTURE.md` for the locked configuration the comparison system must match.

**Files with real content, at the same relative path as cbm-system:**

- `code/config/pinecone_core.py` — Pinecone client, Gemini embedding client, reranker. Copied verbatim from cbm-system with one deviation: `EMBEDDING_MODEL` now reads from the `EMBEDDING_MODEL` env var (falling back to the same hardcoded default cbm-system uses) instead of ignoring it.
- `code/config/vector.py` — `DenseSearcher`, `SparseSearcher`, `HybridRRFSearcher` (RRF fusion, k=60), `EnsembleSearcher`, `resolve_index_names` — copied verbatim.
- `code/nodes/general/utils.py` — `detect_language()`, trimmed from cbm-system's file (langdetect, seeded for determinism, falls back to `"en"`).
- `code/nodes/general/prompts.py` — **new here**, not a port. cbm-system's file is a Langfuse registry with no local prompt text and a persona-switching resolver (`profile_for()`-style: Greenie/GIBI/Buildee/CiMie depending on namespace). This repo has no Langfuse credentials and deliberately does not port persona-switching — see "Known deviations" below.
- `code/tools/general/build.py` — KM's Pinecone-key mapping and the relevance floor (`RAG_RELEVANCE_FLOOR = 0.15`, ported from cbm-system), trimmed from the real file (which also assembles the full agent tool list and RAG-gate tool wrapper this repo skips).
- `code/km_search.py` — retrieval-only CLI (no LLM call). No direct cbm-system equivalent.
- `code/service/graph_llm.py` — retrieval + relevance floor + generation, as a plain async function. No direct cbm-system equivalent (that file there is a ~2000-line LangGraph workflow).
- `code/km_chat.py` — generation-capable CLI, thin wrapper around `service/graph_llm.py`.

**Everything else** (`controller/`, `edges/`, `model/`, `nodes/{da,operation_text2sql,plus,safety,standard_nodes,text2sql}/`, `prisma/`, `repository/`, `tools/{da,operation_text2sql_tools,plus,safety,standard_tools,text2sql}/`, `workflow/`) is an empty skeleton — same folders as cbm-system, `__init__.py`/`.gitkeep` placeholders only, no logic. They're unrelated to KM (safety agent, text2sql, data-analysis, etc.) but kept for structural parity.

Deliberately **not** mirrored: `Dockerfile`, `docker-compose.yml`, `.github/workflows/`, `uv.lock`, `LICENSE`. Copying these verbatim would reference the full FastAPI app (`main:app`) that doesn't exist here — safer to skip than to ship something broken.

## Live Pinecone facts (verified against the live project)

| Index | Type | Metric | Dim |
|---|---|---|---|
| `cimie-km-th-dense` | dense | cosine | 3072 |
| `cimie-km-th-sparse` | sparse | dotproduct | — |
| `cimie-km-en-dense` | dense | cosine | 3072 |
| `cimie-km-en-sparse` | sparse | dotproduct | — |

27,024 vectors per index, split across 16 namespaces (largest: `promotion-profession` 11,274, `manufactur-profession` 9,998; smallest: `cimie` 7). Sparse search is uniform-weight CRC32 token matching, not BM25 — it works (self-retrieval passes, see `test/sparse_index_check.py`), but ranks weaker than real BM25 would.

## Setup

1. `.env` in this repo already has real credentials — **do not edit it**. Its key names carry trailing spaces in the raw text (`PINECONE `, `EMBEDDING_MODEL `, `GOOGLE_APPLICATION_CREDENTIALS_BASE64 `) and the Pinecone key is named `PINECONE`, not `PINECONE_SL`. The code adapts to this (`tools/general/build.py::load_env()` / `resolve_km_key()`) — you should never need to touch `.env`.
2. Dependencies: this repo has no virtualenv of its own. Run everything through the reference repo's venv:
   ```
   PYTHONPATH=code /Users/nattawat1409/Desktop/Cemie/cbm-system/.venv/bin/python3 <script>
   ```
3. Retrieval only (no LLM call):
   ```
   PYTHONPATH=code .../python3 code/km_search.py "<query>" "<namespace>"
   ```
4. Full Q&A (retrieve -> floor check -> generate):
   ```
   PYTHONPATH=code .../python3 code/km_chat.py "<query>" "<namespace>"
   ```
   Default namespace for both is `manufactur-profession` — cbm-system itself has no hardcoded default (the frontend supplies `name_space` per request), but its own UAT demo (`test/foundation_uat_demo.py`) pairs `library="cimie-km"` with `name_space="manufactur-profession"` as the plain-agent case, and it's the second-largest real namespace (9,998 vectors), so it's a reasonable stand-in for ad hoc queries. `general-knowledge` (195 vectors, 0.7% of the corpus) was considered and rejected as a default for the same reason.
5. A comma-separated namespace list also works and fans out across namespaces via `EnsembleSearcher`, e.g. `code/km_search.py "<query>" "manufactur-profession,green-industrial-knowledge"`.

## Architecture (current)

```mermaid
flowchart TD
    Q[User query] --> LD[detect_language\nnodes/general/utils.py]
    LD -->|th/en| RI[resolve_index_names\ncimie-km + language]
    RI --> IDX{{cimie-km-th-* or\ncimie-km-en-*}}
    IDX --> DS[DenseSearcher\ntop_k=20, cosine]
    IDX --> SS[SparseSearcher\ntop_k=20, CRC32 tokens]
    DS --> RRF[RRF fusion, k=60\nHybridRRFSearcher]
    SS --> RRF
    RRF --> RR[Rerank\nbge-reranker-v2-m3, top 8]
    RR --> FLOOR{top score\n>= 0.15?}
    FLOOR -->|no| REFUSE["not covered" message\n(th/en, no LLM call)]
    FLOOR -->|yes| CTX[Assemble context\nDocument ID headers]
    CTX --> PROMPT[SYSTEM_PROMPT\nnodes/general/prompts.py\ncontext-only, cite by ID]
    PROMPT --> LLM[ChatGoogleGenerativeAI\ngemini-2.5-flash, GOOGLE_API_KEY]
    LLM --> ANS[answer + sources + language + indexes_used]
```

## Known deviations from cbm-system (read before comparing against Personalize-AI)

1. **No language split for KM** was the *original* bug, now fixed — `KM_LIBRARY = "cimie-km"` is not in `resolve_index_names()`'s `ignore_suffix`, so Thai/English queries correctly hit `cimie-km-{th,en}-*`.
2. **`EMBEDDING_MODEL` now read from env** (`config/pinecone_core.py`) instead of a silently-ignored hardcoded constant.
3. **Relevance floor** (`RAG_RELEVANCE_FLOOR = 0.15`) ported from cbm-system, applied to the top reranked score in `service/graph_llm.py` rather than via a LangChain-tool wrapper (no tool-calling agent here).
4. **Generation is direct Google AI Studio** (`ChatGoogleGenerativeAI` + `GOOGLE_API_KEY`), not Vertex AI (cbm-system's `create_model_vertex`, ADC auth from `GOOGLE_APPLICATION_CREDENTIALS_BASE64`). Different auth path, quota, and latency profile from production.
5. **Single fixed system prompt, no persona-switching.** cbm-system picks a persona (CiMie/Buildee/Greenie/GIBI) from `(library, name_space)`. This repo intentionally uses one hand-written prompt for every namespace so persona never becomes a hidden variable alongside the memory layer under test. The real CiMie prompt text lives in Langfuse (remote), which this repo has no credentials for — the prompt in `nodes/general/prompts.py` is **not** that text and must not be mistaken for it.
6. **No chat history / query rewriting / LangGraph** — `service/graph_llm.py` is one straight-line async function, not cbm-system's multi-node workflow.

The locked configuration and the confound checklist for the comparison live in `HANDOFF_NEW_ARCHITECTURE.md`.

## Benchmark

Compares this baseline against the memory-augmented system on four metrics: win-rate, cross-session recall, wrong-memory rate, and Recall@8 (guardrail). The conversation is `test/eval/evalset.json` — 22 scripted Thai turns (17 questions, 3 reminders, 2 profile statements) over two sessions. It is frozen: do not regenerate it (`build_conversation_evalset.py` is kept only as provenance).

Run the baseline:
```
uv run python test/eval/run_benchmark.py
```
Score it (picks the newest `baseline_*.jsonl`):
```
uv run python test/eval/score_benchmark.py
```
Compare against the new system (adds win-rate; refuses if the locked config differs):
```
uv run python test/eval/score_benchmark.py --baseline <baseline>.jsonl --new <new>.jsonl
```
`test/eval/benchmark_adapter.py` is the only system-specific file; the runner and scorer are identical for both systems.
