"""Per-request retrieval config — the slice of cbm-system's tools/general/build.py
this benchmark needs. The real file also assembles the full LangChain tool list
(RAG relevance gate, skill switches, FoundationAgent) for the agent stack, which
this repo doesn't have.

Each library reads a different Pinecone project; anything not listed (all of
cimie) uses PINECONE_CIMIE. Copied verbatim from cbm-system's
PROFESSION_API_KEY_ENV.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

PROFESSION_API_KEY_ENV = {"mro-mrs": "PINECONE_MRO_MRS", "km": "PINECONE_SL"}

# Deviation from cbm-system: the real KM_LIBRARY is "km", which
# config/vector.py::resolve_index_names() maps straight to "km-dense" /
# "km-sparse" (it's in `ignore_suffix`). Those indexes do not exist in this
# Pinecone project (verified against the live project — see BASELINE_PLAN.md).
# The real indexes are "cimie-km-{th,en}-{dense,sparse}", so the base index
# name here must be "cimie-km" — and "cimie-km" is NOT in `ignore_suffix`,
# so it gets the -th/-en split for free once resolve_index_names() runs.
#
# "cimie-km" is also deliberately NOT added to PROFESSION_API_KEY_ENV above —
# that dict is kept verbatim from cbm-system for fidelity — so resolve_km_key()
# below falls through to the generic candidate list rather than a dict lookup.
KM_LIBRARY = "cimie-km"

# The real .env in this repo does not define PINECONE_SL at all; it has
# PINECONE instead (see BASELINE_PLAN.md's verified facts). Tried in order,
# first hit wins. PINECONE_CIMIE is tried first because that's the generic
# "everything not explicitly mapped" key cbm-system itself would fall back to.
_KM_KEY_CANDIDATES = ["PINECONE_CIMIE", "PINECONE", "PINECONE_SL"]


def load_env(dotenv_path: str | Path | None = None) -> None:
    """Load `.env` into os.environ, normalising whitespace around key names.

    This repo's `.env` was written with padded keys and quoted values, e.g.
    `PINECONE = "pcsk_..."` (see BASELINE_PLAN.md). Empirically, the installed
    python-dotenv already strips that padding when parsing (`dotenv_values()`
    returns clean keys), so a plain `load_dotenv()` happens to work today —
    but that's an implementation detail of one library version, not a
    guarantee. This does the normalisation explicitly (strip both key and
    value) so key lookup doesn't silently break if that changes, mirroring
    the same defensive load `test/sparse_index_check.py` already does.
    Never edits `.env` itself — only how it's read.
    """
    path = (
        Path(dotenv_path)
        if dotenv_path
        else Path(__file__).resolve().parents[3] / ".env"
    )
    raw = dotenv_values(path)
    for k, v in raw.items():
        key = (k or "").strip()
        if not key:
            continue
        os.environ.setdefault(key, (v or "").strip())


def resolve_km_key() -> str:
    """KM's Pinecone key.

    cbm-system maps "km" -> PINECONE_SL via PROFESSION_API_KEY_ENV, but
    KM_LIBRARY is "cimie-km" here (see comment above) and isn't a key in that
    dict, so this doesn't do the dict lookup at all — it tries the real
    .env's candidates directly. Raises if none are set.

    Side effect: also mirrors the resolved value into os.environ under the
    plain "PINECONE_API_KEY" name (setdefault, so an explicit value already
    in the environment always wins). langchain_pinecone's PineconeRerank
    (used by config/vector.py's get_reranker_client(), copied verbatim from
    cbm-system) takes no api_key argument at all — it only ever reads
    PINECONE_API_KEY from the environment. cbm-system's real deployment must
    set that var directly; this repo's .env doesn't (see BASELINE_PLAN.md),
    so without this, every reranked search (all of HybridRRFSearcher,
    EnsembleSearcher) fails at reranker construction even though the plain
    dense/sparse index calls -- which do take an explicit api_key -- work
    fine. The one Pinecone key in this project is used for both purposes.
    """
    for name in _KM_KEY_CANDIDATES:
        val = os.getenv(name)
        if val:
            os.environ.setdefault("PINECONE_API_KEY", val)
            return val
    raise RuntimeError(
        f"No Pinecone key set — tried {', '.join(_KM_KEY_CANDIDATES)}. "
        "Add one to .env (see BASELINE_PLAN.md)."
    )


# Below this rerank score (HybridRRFSearcher's "score" field, sourced from the
# reranker's relevance_score once rerank=True) the knowledge base is treated
# as not covering the question, rather than returned as a weak match. Ported
# verbatim from cbm-system's tools/general/build.py — same threshold, same
# semantics (it gates on the *reranked* score, not raw cosine similarity).
# Applied in service/graph_llm.py to km_chat.py's top result, rather than via
# cbm-system's LangChain-tool-wrapping approach (_gate_rag_tool), since this
# repo has no agent/tool-calling layer to wrap.
RAG_RELEVANCE_FLOOR = 0.15
