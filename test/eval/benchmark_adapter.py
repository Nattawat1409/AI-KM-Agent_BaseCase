"""Baseline adapter — the ONLY file that differs between the two systems.

`run_benchmark.py` replays the frozen conversation through whichever adapter it
is pointed at. The new (memory) repo ships its own `benchmark_adapter.py`
exposing the same names, so both systems are driven by byte-identical runner
code and produce byte-compatible result files.

Contract
--------
Required
  answer(question, user_id, session_id) -> dict      (sync or async)
    keys: answer, sources, language, floor_triggered, top_score,
          memory_used, latency_ms, tokens, retrieved_chunk_ids
Optional (the runner calls them if present)
  reset_memory(user_id)            called once before the first turn
  end_session(user_id, session_id) called when the script leaves a session, so
                                   nightly / episodic summarisation can be
                                   triggered on demand
Module constants
  SYSTEM_LABEL, MEMORY_ENABLED, NAMESPACE, GENERATION_MODEL

`retrieved_chunk_ids` is the fusion-stage top-8 Pinecone vector IDs, in rank
order, from HybridRRFSearcher.search_raw(). It is the only place chunk identity
survives (search() renumbers ids 1..N), and Recall@8 is computed from it. It
must come from the RAW user question — memory must not alter retrieval.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from tools.general.build import load_env  # noqa: E402

load_env()

from nodes.general.utils import detect_language  # noqa: E402
from service.graph_llm import LLM_MODEL, answer_query, build_km_searcher  # noqa: E402

SYSTEM_LABEL = "baseline"
MEMORY_ENABLED = False
NAMESPACE = "manufactur-profession"
GENERATION_MODEL = LLM_MODEL


async def answer(question: str, user_id: str, session_id: str) -> dict:
    # The baseline is stateless: user_id / session_id are accepted and ignored.
    # That is exactly the behaviour under test.
    language = detect_language(question)

    # Chunk identity (not timed — this is measurement, not the product path).
    raw = await build_km_searcher(NAMESPACE, language).search_raw(question)

    t0 = time.perf_counter()
    result = await answer_query(question, NAMESPACE)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "language": result["language"],
        "floor_triggered": result["floor_triggered"],
        "top_score": result["top_score"],
        "memory_used": [],
        "latency_ms": latency_ms,
        # answer_query() does not surface usage metadata; null, not a guess.
        "tokens": None,
        "retrieved_chunk_ids": [r["id"] for r in raw],
    }
