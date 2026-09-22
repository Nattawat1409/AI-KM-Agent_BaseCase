"""Baseline KM Q&A orchestration: detect language -> retrieve -> floor check
-> assemble context -> call LLM -> return {answer, sources, language,
indexes_used}.

Deviation from cbm-system: the real service/graph_llm.py is a ~2000-line
LangGraph workflow (km_cimie_workflow, reached via controller/llm.py) with
separate history, query-rewrite, RAG, and generation nodes, plus tool-calling
agent scaffolding this repo doesn't have (no chat history, no query rewriting,
no multi-turn state). This is a single async function doing the same
retrieve-then-generate shape without a graph runtime — note this if
node-level behavior (e.g. rewrite quality, history handling) is ever part of
the comparison; this baseline has none of it.

Deviation from cbm-system's generation path: cbm-system generates via Vertex
AI (tools/standard_tools/core.py::create_model_vertex — ADC auth from
GOOGLE_APPLICATION_CREDENTIALS_BASE64, a separate auth path from the
GOOGLE_API_KEY used for embeddings). This uses ChatGoogleGenerativeAI's direct
Google AI Studio path (GOOGLE_API_KEY) for both embeddings and generation
instead, per BASELINE_PLAN.md — fewer moving parts for a benchmark baseline,
but a real difference in auth path / quota / latency characteristics from the
production system if that's ever benchmarked too.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from config.vector import THAI_LANGS, HybridRRFSearcher, resolve_index_names
from nodes.general.prompts import NOT_COVERED_MESSAGE, SYSTEM_PROMPT
from nodes.general.utils import detect_language
from tools.general.build import KM_LIBRARY, RAG_RELEVANCE_FLOOR, resolve_km_key

# Model choice per BASELINE_PLAN.md Phase 2. Not read from .env: there is no
# LLM_MODEL var in this repo's .env (only EMBEDDING_MODEL), and pinning the
# model in code keeps it identical across benchmark runs regardless of env
# drift.
LLM_MODEL = "gemini-2.5-flash"


def _lang_bucket(language: str) -> str:
    """th/en bucket for NOT_COVERED_MESSAGE, using the exact same test
    resolve_index_names() uses (THAI_LANGS) so the refusal language and the
    index actually queried never disagree."""
    return "th" if (language or "").strip().lower() in THAI_LANGS else "en"


def build_km_searcher(name_space: str, language: str = "Thai") -> HybridRRFSearcher:
    dense_idx, sparse_idx = resolve_index_names(KM_LIBRARY, language)
    api_key = resolve_km_key()
    return HybridRRFSearcher(
        name_space=name_space,
        index_name_dense=dense_idx,
        index_name_sparse=sparse_idx,
        api_key_dense=api_key,
        api_key_sparse=api_key,
        rerank=True,
    )


async def answer_query(
    query: str,
    name_space: str = "manufactur-profession",
    language: str | None = None,
) -> dict:
    """Run one KM question end to end.

    Returns a dict with:
      answer         - the model's answer, or NOT_COVERED_MESSAGE if the
                        relevance floor wasn't met (no LLM call in that case)
      sources        - [{id, source, score}, ...] for the documents actually
                        used (empty when the floor triggered)
      language       - "th" or "en", the bucket actually used for index
                        selection and the refusal message
      indexes_used   - [dense_index_name, sparse_index_name]
      top_score      - the top document's post-rerank score
      floor_triggered- bool
    """
    if language is None:
        language = detect_language(query)
    lang_code = _lang_bucket(language)

    dense_idx, sparse_idx = resolve_index_names(KM_LIBRARY, language)
    indexes_used = [dense_idx, sparse_idx]

    searcher = build_km_searcher(name_space, language)
    docs = await searcher.search(query)

    # Floor applies to the *reranked* score (HybridRRFSearcher's "score" field
    # is the reranker's relevance_score once rerank=True — see
    # config/vector.py::_rerank_documents / _format_document), matching
    # cbm-system's RAG_RELEVANCE_FLOOR semantics exactly (see
    # tools/general/build.py).
    top_score = max((d.get("score", 0.0) for d in docs), default=0.0)

    if top_score < RAG_RELEVANCE_FLOOR:
        return {
            "answer": NOT_COVERED_MESSAGE[lang_code],
            "sources": [],
            "language": lang_code,
            "indexes_used": indexes_used,
            "top_score": top_score,
            "floor_triggered": True,
        }

    context = "\n\n".join(d["content"] for d in docs)
    prompt = SYSTEM_PROMPT.format(context=context)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    response = await llm.ainvoke([("system", prompt), ("human", query)])

    sources = [
        {"id": d["id"], "source": d["source"], "score": d["score"]} for d in docs
    ]
    return {
        "answer": response.content,
        "sources": sources,
        "language": lang_code,
        "indexes_used": indexes_used,
        "top_score": top_score,
        "floor_triggered": False,
    }
