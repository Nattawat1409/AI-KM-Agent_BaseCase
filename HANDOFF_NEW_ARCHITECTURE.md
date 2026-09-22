# HANDOFF — build the "new architecture" repo so it can be benchmarked against the baseline

You are building a second repo: the baseline RAG **plus a memory / personalization layer**. A human will compare it against the baseline repo (`AI-KM-Agent_BaseCase`) with an LLM judge. That comparison is only valid if **the memory layer is the only difference between the two systems.**

> **The one rule:** everything in §1 must be identical to the baseline. If you are tempted to "improve" retrieval, the prompt, the model, or the index while you are in here — don't. A retrieval improvement would be credited to memory and invalidate the result. This has already happened once in testing: widening the reranker pool from 8 to 50 raised Recall@8 from 68.2% to 84.1% (+15.9pp, measured on an earlier retrieval-only eval set) with no memory involved at all.

Baseline snapshot: 2026-09-21. **The baseline's work is not committed to git** (its only commit is "Initial commit"), so a git SHA identifies nothing. The **sha256 hashes in §2 are the source of truth** for "same code".

---

## 1. Locked configuration (must be identical)

| Item | Value |
|---|---|
| Python | 3.12 |
| Pinecone project | the one behind `PINECONE` in the shared `.env` (see §3) |
| Indexes | `cimie-km-th-dense`, `cimie-km-th-sparse` (Thai) · `cimie-km-en-dense`, `cimie-km-en-sparse` (English) |
| Index routing | `detect_language(question)` → `"th"` ⇒ `-th-*` indexes; **anything else ⇒ `-en-*`** |
| Namespace | `manufactur-profession` (single namespace per query) |
| Embedding model | `models/gemini-embedding-001` (3072-dim, cosine), read from env `EMBEDDING_MODEL` |
| Dense retrieval | query embedding → `top_k=20`, no metadata filter |
| Sparse retrieval | CRC32-hashed tokens (whitespace words + every 3-char shingle), all weights `1.0`, `top_k=20`. **Not BM25.** dotproduct index |
| Fusion | Reciprocal Rank Fusion, `k=60`, keep top `top_n=8` |
| Reranker | Pinecone-hosted `bge-reranker-v2-m3`, final `rerank_top_n=8` |
| **Rerank pool depth** | **8** (fusion keeps 8, reranker sees exactly those 8). See §4-A — this is the most dangerous number in the repo |
| Relevance floor | `0.15` on the **top reranked score**. Below it: return the fixed "not covered" message, **no LLM call** |
| Generation model | `gemini-2.5-flash` via `ChatGoogleGenerativeAI` with `GOOGLE_API_KEY` (direct Google AI Studio, **not** Vertex). `temperature` and `max_output_tokens` are **not set** — leave them unset in both repos |
| System prompt | `code/nodes/general/prompts.py::SYSTEM_PROMPT`, used verbatim, one prompt for every namespace (no persona switching) |
| Context assembly | `"\n\n".join(doc["content"] for doc in reranked_docs)` — each `content` already starts with a `[Document ID: N] \| Source: … \| Page: …` header |
| Message format | `[("system", SYSTEM_PROMPT.format(context=context)), ("human", question)]` |
| Not-covered messages | th: `ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้ในฐานความรู้` · en: `The knowledge base does not contain information relevant to this question.` |
| Library versions (`uv.lock`) | pinecone 7.3.0 · langchain-pinecone 0.2.13 · langchain-google-genai 4.4.0 · langchain-core 1.6.3 · langdetect 1.0.9 |

---

## 2. Copy these files byte-for-byte — do not rewrite them

Copy from the baseline repo into the **same relative paths**. Do not reformat, "clean up", or re-implement.

```
code/config/vector.py            HybridRRFSearcher, sparse encoder, RRF, index-name resolution
code/config/pinecone_core.py     embedding client, reranker, Pinecone client
code/nodes/general/prompts.py    SYSTEM_PROMPT, NOT_COVERED_MESSAGE
code/nodes/general/utils.py      detect_language (langdetect, seed=0, fallback "en")
code/tools/general/build.py      load_env, resolve_km_key, RAG_RELEVANCE_FLOOR, KM_LIBRARY
code/service/graph_llm.py        the baseline pipeline — your starting point, see §5
test/eval/evalset.json           the frozen 22-turn conversation (see §8)
test/eval/run_benchmark.py       the runner — the SAME file drives both systems
```

Verify after copying (run in your repo root; every line must print `OK`):

```bash
check() { [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$2" ] && echo "OK   $1" || echo "DIFF $1"; }
check code/config/vector.py           ec4158ba3cef9de2a71e62e9c12363f6dd2a82d1131141071f7e6d5815991428
check code/config/pinecone_core.py    1ce1521f81014b77157fde33af24a8948663860c9996a260c9d49bf1dd1259e2
check code/nodes/general/prompts.py   122080c99aa1b16cc5ba63c89aa1998049d3a66e3727f1ec2a9ecb7f1da54d19
check code/nodes/general/utils.py     f39ccf64511941a6726e7623e273337c4058428a23847a0eb28571b5e6b4241c
check code/tools/general/build.py     4f0ca6f993ef5a3e5868a5dd2d26804d3b623b52910e370b2419c789ca95ac14
check test/eval/evalset.json          4b14cbdc55bd85e215944231912bbc4f22c87118d8c448807fb07330f9e9e614
check test/eval/run_benchmark.py      003abfc7653e2cb3260c7fdc0a54f52416be836f0485a64e2c2bc33ee38e2454
```

`code/service/graph_llm.py` (`e4a68d74…bd2`) is **expected to differ** in your repo — that is where memory gets added. **You write one new file, `test/eval/benchmark_adapter.py`**, exposing the contract in §6; the baseline's own copy is a working template to read (its docstring is the authoritative contract) but do not copy it verbatim — it calls the stateless pipeline.

`code/` is the package root: imports are `from config.vector import …`, not `from code.config…`.

## 3. Environment

Use the **same** `.env` values as the baseline (copy the file; never print or commit it; never edit the original). Variables the pipeline reads:

| Var | Used for |
|---|---|
| `PINECONE` | Pinecone key. **Named `PINECONE`, not `PINECONE_SL`** — `resolve_km_key()` tries `PINECONE_CIMIE`, `PINECONE`, `PINECONE_SL` in that order |
| `GOOGLE_API_KEY` | embeddings **and** generation |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` |
| `GOOGLE_APPLICATION_CREDENTIALS_BASE64` | present in `.env`, **unused** by this pipeline |

The real `.env` is written as `KEY = "value"` (spaces, quotes). `load_env()` in `build.py` already normalises that — call it before anything else, as the baseline does.

Run with `uv run python …` from the repo root. Add `.env` to `.gitignore`.

---

## 4. Traps already found — do not rediscover them

**A. The reranker is starved (pool depth = 8).** `_rrf_fusion()` truncates to `top_n=8` and `search()` passes exactly those 8 documents to the cross-encoder, so it can reorder them but never rescue a chunk ranked 9th. Fusing 50 → reranking to 8 gives +15.9pp Recall@8. **Do not fix this in only one repo.** If the baseline changes it, this document will be re-issued; until then keep it at 8 and record it in your results config (§7).

**B. Reranker key.** `PineconeRerank` takes no key argument and reads only the env var `PINECONE_API_KEY`. `resolve_km_key()` mirrors the found key into it — so call `resolve_km_key()` before building any searcher, or reranking fails with an auth error even though dense/sparse work.

**C. `search()` destroys chunk identity.** `_format_document()` renumbers `id` to 1..N and drops the Pinecone vector ID. Anything that needs real chunk IDs (Recall/MRR) must use `HybridRRFSearcher.search_raw()`, which keeps the original metadata.

**D. Post-rerank text is not the raw text.** `_format_document()` prepends the chunk's `contextual_text` metadata before the body, so string equality between raw and reranked text fails. Match on an interior slice instead.

**E. `langdetect` guesses wrong on very short queries** (`"Mill gearbox eff.?"` → Welsh). Harmless today only because every non-`th` result routes to the English index. Do not "fix" this by adding a third route.

**F. Rerank scores are on a different scale from cosine.** The floor `0.15` applies to the reranker score. Off-topic queries score ≈ 0.01; on-topic ≈ 0.7–0.99.

---

## 5. What you MAY change — the memory layer only

Start from the baseline's `service/graph_llm.py::answer_query()`. Memory plugs in at exactly two places:

```
question ──► [read memory] ──► retrieval (UNCHANGED) ──► floor (UNCHANGED)
                  │                                            │
                  └────────► prompt assembly ◄─────────────────┘
                                   │
                                   ▼
                            LLM call (same model)
                                   │
                                   ▼
                     answer ──► [write-back memory]
```

- **Read:** load user profile + relevant topic/episodic notes, inject into the prompt.
- **Write-back:** after answering, decide "worth remembering?" and persist. Keep it off the critical path if you like, but it must complete before the next session begins.

**Retrieval must not see memory.** Do not use memory to rewrite the query, add metadata filters, or re-rank documents. That changes retrieval and confounds the benchmark. (If you want to test memory-influenced retrieval, that is a *separate arm* — ask the human.)

**Memory must not change the floor decision.** If the floor triggers, the answer is the fixed not-covered message, exactly as in the baseline.

### Required: a kill switch

Provide `MEMORY_ENABLED=false` (env or argument). With it off, your pipeline must reproduce the baseline exactly — same retrieval, same prompt bytes, same answers modulo LLM sampling. **Run the benchmark with memory off and confirm retrieval is identical to the baseline's (§9) before benchmarking anything.** If it does not match, your repo has drifted and nothing else can be trusted.

### A warning about the "target architecture" diagram

The diagram the human designed for this project also contains: BM25 instead of the CRC32 sparse encoder, metadata filter, taxonomy prune, cross-encoder rerank 20→5, a freshness multiplier, and a rebuilt KM index with `doc_uid / content_hash / freshness_score`. **None of that may be in this benchmark build.** It is retrieval, not memory. Build only: `user_profile.md`, `topics_index.json` + LLM router, `topic .md` files, episodic notes, the write-back gate, and the nightly summarisation. Ask the human before adding anything to retrieval.

---

## 6. Interface the runner calls — `test/eval/benchmark_adapter.py`

`run_benchmark.py` imports your adapter and calls it; you never edit the runner. Provide, in `test/eval/benchmark_adapter.py`:

```python
SYSTEM_LABEL = "new-arch"          # names your result file
MEMORY_ENABLED = True              # False when the kill switch is off
NAMESPACE = "manufactur-profession"
GENERATION_MODEL = "gemini-2.5-flash"

async def answer(question: str, user_id: str, session_id: str) -> dict: ...   # sync also fine
async def reset_memory(user_id: str) -> None: ...                # called once, before turn 1
async def end_session(user_id: str, session_id: str) -> None: ...  # called at each session boundary
```

`answer()` must return every one of these keys:

```python
{
  "answer": str,
  "sources": [{"id": int, "source": str, "score": float}],
  "language": "th" | "en",
  "floor_triggered": bool,
  "top_score": float,
  "memory_used": [str],              # ids/paths of memory items injected; [] when none/off
  "latency_ms": int,                 # time of the answer path only
  "tokens": {"prompt": int, "completion": int} | None,   # None is acceptable
  "retrieved_chunk_ids": [str],      # REQUIRED — see below
}
```

**`retrieved_chunk_ids`** = the fusion-stage top-8 Pinecone vector IDs, in rank order, from `HybridRRFSearcher.search_raw(<the raw user question>)`. Recall@8 is computed from it. The baseline adapter shows exactly how. It must come from the **raw question** — memory may not alter it.

`end_session` exists so nightly / episodic summarisation can be triggered on demand instead of waiting for a real night. `reset_memory` must leave no state behind.

**Isolation is a tested property.** Memory for one `user_id` must never appear in an answer for another.

## 7. Results file — produced by the runner, never hand-written

```bash
uv run python test/eval/run_benchmark.py            # writes test/eval/results/<SYSTEM_LABEL>_<timestamp>.jsonl
```

Line 1 is a config fingerprint the scorer compares against the baseline's. **It refuses to compare two systems if any locked field differs** (retrieval code hashes, prompt, embedding/generation model, `top_k`, fusion pool, `rerank_top_n`, `rrf_k`, floor, evalset hash). The runner computes all of it live from your actual files — do not fake or hard-code any of it.

Every later line is one turn: `turn_id`, `session_id`, `user_id`, `type` (`profile` / `question` / `reminder`), `question`, plus everything `answer()` returned.

## 8. The benchmark conversation — `test/eval/evalset.json`

One Thai user (`u1`), **22 scripted turns across two sessions**. Replayed in exact order by the runner; do not shuffle, edit, or extend it.

| Turns | What |
|---|---|
| 2 × `profile` (session 1) | the user states who they are ("new engineer, explain simply") and how they like answers ("bullet points + a short summary underneath") |
| 17 × `question` (9 in session 1, 8 in session 2) | Thai questions written from 17 real Pinecone chunks; each carries `gold_chunk_id` for Recall@8 |
| 3 × `reminder` (end of session 2) | "did we discuss X — and what did I say about myself?" Only a system with memory can answer them |

The four metrics the human will report: **win-rate** (blind pairwise LLM judge, A/B swapped), **cross-session recall** (the 3 reminders), **wrong-memory rate** (false claims about the user or the past), and **Recall@8** as a guardrail that memory did not damage retrieval.

**Do not write your own scenarios** — a test authored by whoever built the system under test flatters it. Session 1 turns are meant to teach your memory: write-back must complete before session 2 begins (`end_session` is your hook).

## 9. Acceptance checklist — do not report "done" until every box is true

- [ ] All §2 hash checks print `OK`
- [ ] `resolve_km_key()` is called before any searcher is built; reranking works
- [ ] **Kill switch parity:** with `MEMORY_ENABLED=false`, run the benchmark and confirm `retrieved_chunk_ids` are **identical to the baseline's on all 17 questions** (`score_benchmark.py` prints "retrieval identical on 17/17"). The baseline's Recall@8 on this set is **41.2% chunk-level (7/17)**, 58.8% document-level — not the 68.2% from the older eval set. If your numbers differ, your repo has drifted
- [ ] Retrieval, floor and prompt code paths are untouched; memory only enters via prompt assembly
- [ ] Floor-triggered questions still return the fixed message with no LLM call
- [ ] `test/eval/benchmark_adapter.py` implements §6 including `retrieved_chunk_ids`, `reset_memory`, `end_session`
- [ ] A test you wrote proves user isolation (one user's fact never surfaces for another)
- [ ] Results come from `run_benchmark.py` (config line present, hashes computed live)
- [ ] `.env` is not committed, not printed, not edited
- [ ] You did not change any retrieval parameter, index, namespace, model or prompt

## 10. Report back to the human

State plainly: what you built, which acceptance boxes are **not** checked and why, and every place you were unsure whether something counted as "memory" or "retrieval". A broken box reported honestly is worth more than a claim that everything passed.
