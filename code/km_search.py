"""KM (Buildee) retrieval entrypoint — base case for the AI-personalize benchmark.

Ported from cbm-system (dev branch), now laid out at the same relative paths
as the source repo:
  - config/pinecone_core.py, config/vector.py — copied verbatim (one
    deviation: EMBEDDING_MODEL now reads from env, see pinecone_core.py).
  - nodes/general/utils.py — detect_language(), trimmed from the real file.
  - tools/general/build.py — KM's Pinecone-key mapping, trimmed from the real
    file (which also assembles the full agent tool list this repo skips).

This script is the one piece with no direct cbm-system equivalent: the real
app drives retrieval through controller/llm.py -> service/graph_llm.py's
LangGraph workflow. This is a plain CLI stand-in for that, kept flat at
code/ root since it doesn't belong to any one mirrored layer. (The
generation-capable counterpart is code/km_chat.py -> service/graph_llm.py;
this file stays retrieval-only.)

KM_LIBRARY is "cimie-km" (see tools/general/build.py for why), which is NOT
in config/vector.py's `ignore_suffix`, so it DOES get the -th/-en split:
Thai queries hit cimie-km-th-{dense,sparse}, English queries hit
cimie-km-en-{dense,sparse}. This corrects the module's original caveat
(carried over before the real index names were confirmed against the live
Pinecone project — see BASELINE_PLAN.md).

Default namespace is "manufactur-profession". cbm-system itself has no
hardcoded default — the frontend supplies name_space per request — but its
own UAT demo (test/foundation_uat_demo.py) pairs library="cimie-km" with
name_space="manufactur-profession" as the plain-agent case, and it's the
largest real namespace (9,998 of 27,024 vectors) after promotion-profession,
so it's a reasonable stand-in default for ad hoc CLI queries.
"""

from tools.general.build import load_env

load_env()

import asyncio

from config.vector import EnsembleSearcher, HybridRRFSearcher, resolve_index_names
from nodes.general.utils import detect_language
from tools.general.build import KM_LIBRARY, resolve_km_key

DEFAULT_NAMESPACE = "manufactur-profession"


def build_km_searcher(
    name_space: str | list[str], language: str = "Thai"
) -> HybridRRFSearcher | EnsembleSearcher:
    """A single namespace uses HybridRRFSearcher directly (dense+sparse RRF,
    reranked). A list of namespaces uses EnsembleSearcher instead — Pinecone's
    query() only accepts one namespace per call, so searching several means
    fanning out one HybridRRFSearcher per namespace and merging via rerank,
    which is exactly what EnsembleSearcher already does (see config/vector.py).
    """
    api_key = resolve_km_key()

    if isinstance(name_space, list):
        return EnsembleSearcher(
            index_names=KM_LIBRARY,
            name_spaces=name_space,
            language=language,
            api_key_dense=api_key,
            api_key_sparse=api_key,
        )

    dense_idx, sparse_idx = resolve_index_names(KM_LIBRARY, language)
    return HybridRRFSearcher(
        name_space=name_space,
        index_name_dense=dense_idx,
        index_name_sparse=sparse_idx,
        api_key_dense=api_key,
        api_key_sparse=api_key,
        rerank=True,
    )


async def run_query(
    query: str, name_space: str | list[str], language: str | None = None
):
    """`language=None` auto-detects from `query` via langdetect, which now
    determines whether cimie-km-th-* or cimie-km-en-* gets queried (see
    module docstring)."""
    if language is None:
        language = detect_language(query)
    searcher = build_km_searcher(name_space, language)
    return await searcher.search(query)


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "ทดสอบระบบค้นหา"
    raw_ns = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NAMESPACE
    # Comma-separated CLI arg -> list, so the EnsembleSearcher path is reachable
    # from the command line too, e.g.: python km_search.py "q" "ns1,ns2"
    name_space = raw_ns.split(",") if "," in raw_ns else raw_ns
    docs = asyncio.run(run_query(query, name_space))
    for d in docs:
        print(f"[{d['id']}] score={d['score']:.4f} source={d['source']}")
        print(d["text"][:200])
        print("-" * 60)
