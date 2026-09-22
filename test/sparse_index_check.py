"""Diagnose whether the sparse half of hybrid search actually works.

The suspicion: config/vector.py builds query-side sparse vectors by CRC32-hashing
tokens with uniform 1.0 weights. If the Pinecone sparse index was built with a
different encoding (real BM25, a different tokenizer, a different hash), the
query term IDs never match the indexed term IDs — so sparse search returns
nothing useful and RRF silently degrades to dense-only. No error is raised;
it just quietly retrieves worse.

This script answers that with three checks, strongest last:
  1. What indexes actually exist, and do they hold vectors?
  2. Does a normal query return anything from sparse at all?
  3. SELF-RETRIEVAL: take text verbatim out of a document the dense index
     returned, feed it back as a sparse query. A working keyword index MUST
     return that same document — the query is literally copied from it. If it
     doesn't come back, the encodings don't match. This is the decisive test,
     because it removes "maybe the query was just bad" as an explanation.

Run:
    cd AI-KM-Agent_BaseCase
    PYTHONPATH=code python test/sparse_index_check.py
    PYTHONPATH=code python test/sparse_index_check.py "<query>" "<namespace>"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from dotenv import dotenv_values

from config.vector import DenseSearcher, SparseSearcher

# .env in this repo has keys written with trailing spaces ("PINECONE ="), and
# the key is named PINECONE while the code expects PINECONE_SL. Rather than
# edit .env, normalise here so the diagnostic runs as-is.
CANDIDATE_KEY_NAMES = ["PINECONE_SL", "PINECONE", "PINECONE_CIMIE", "PINECONE_API_KEY"]


def load_env() -> dict:
    raw = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    return {(k or "").strip(): (v or "").strip() for k, v in raw.items()}


def mask(value: str) -> str:
    return f"{value[:6]}...{value[-4:]} ({len(value)} chars)" if value else "(empty)"


def resolve_api_key(env: dict) -> tuple[str, str]:
    for name in CANDIDATE_KEY_NAMES:
        if env.get(name):
            return name, env[name]
    raise SystemExit(f"No Pinecone key found. Tried: {', '.join(CANDIDATE_KEY_NAMES)}")


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {n}: {title}\n{'=' * 70}")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "ทดสอบระบบค้นหา"
    namespace = sys.argv[2] if len(sys.argv) > 2 else "thai-namespace"

    env = load_env()
    key_name, api_key = resolve_api_key(env)
    # DenseSearcher/SparseSearcher read os.getenv as a fallback; set both the
    # canonical name and whatever we found so either path works.
    os.environ["PINECONE_SL"] = api_key
    os.environ["PINECONE_CIMIE"] = api_key
    for k, v in env.items():
        os.environ.setdefault(k, v)

    step(0, "Environment")
    print(f"Pinecone key found as: {key_name} -> {mask(api_key)}")
    if key_name != "PINECONE_SL":
        print(f"  ⚠️  code expects PINECONE_SL; found {key_name}. Patched for this run only.")
    for name in ["GOOGLE_API_KEY", "EMBEDDING_MODEL", "GOOGLE_APPLICATION_CREDENTIALS_BASE64"]:
        print(f"{name}: {mask(env.get(name, ''))}")

    from pinecone import Pinecone

    pc = Pinecone(api_key=api_key)

    step(1, "What indexes actually exist in this Pinecone project?")
    names = [i["name"] for i in pc.list_indexes()]
    for n in names:
        print(f"  - {n}")
    if not names:
        raise SystemExit("No indexes in this project — wrong Pinecone key?")

    dense_name = next((n for n in names if "dense" in n), None)
    sparse_name = next((n for n in names if "sparse" in n), None)
    print(f"\nUsing dense index : {dense_name}")
    print(f"Using sparse index: {sparse_name}")
    if not sparse_name:
        raise SystemExit("No sparse index found — hybrid search cannot work at all.")

    step(2, "Index stats (namespaces + vector counts)")
    for label, idx_name in [("DENSE", dense_name), ("SPARSE", sparse_name)]:
        if not idx_name:
            continue
        stats = pc.Index(idx_name).describe_index_stats()
        print(f"\n{label} ({idx_name}): {stats.get('total_vector_count')} vectors")
        for ns, meta in (stats.get("namespaces") or {}).items():
            print(f"    namespace {ns!r}: {meta.get('vector_count')} vectors")

    step(3, f"Normal query — dense vs sparse   (query: {query!r}, ns: {namespace!r})")
    dense = DenseSearcher(name_space=namespace, index_name=dense_name, api_key=api_key)
    sparse = SparseSearcher(name_space=namespace, index_name=sparse_name, api_key=api_key)

    dense_hits = dense.search_dense(query)
    sparse_hits = sparse.search_sparse(query)
    print(f"dense  -> {len(dense_hits)} hits")
    for h in dense_hits[:3]:
        print(f"    {h['score']:.4f}  {h['source']}  {h['text'][:70]!r}")
    print(f"sparse -> {len(sparse_hits)} hits")
    for h in sparse_hits[:3]:
        print(f"    {h['score']:.4f}  {h['source']}  {h['text'][:70]!r}")

    step(4, "SELF-RETRIEVAL — the decisive test")
    if not dense_hits:
        print("Dense returned nothing, so there's no known-good text to test with.")
        print("Try a different query/namespace before drawing conclusions.")
        return

    target = dense_hits[0]
    probe = " ".join(str(target["text"]).split()[:25])
    print(f"Taking text verbatim from a real indexed document:\n  source: {target['source']}\n  text  : {probe[:120]!r}\n")
    print("A working keyword index MUST return this document for this query —")
    print("the query IS the document's own words.\n")

    self_hits = sparse.search_sparse(probe)
    print(f"sparse self-retrieval -> {len(self_hits)} hits")
    for h in self_hits[:5]:
        marker = "  <-- FOUND IT" if h["source"] == target["source"] else ""
        print(f"    {h['score']:.4f}  {h['source']}{marker}")

    found = any(h["source"] == target["source"] for h in self_hits)

    step(5, "Verdict")
    if found:
        print("✅ Sparse index and query encoding MATCH.")
        print("   The document was retrieved using its own text. Keyword search works.")
        print("   Note: it's still uniform-weight matching, not BM25 — weaker ranking,")
        print("   but functional. Hybrid search is genuinely using both halves.")
    elif self_hits:
        print("⚠️  Sparse returns hits, but NOT the document the text came from.")
        print("   Encoding mismatch is likely — results look like noise, not keyword matches.")
        print("   Effect: RRF is fusing dense with garbage, which can rank worse than dense alone.")
    else:
        print("❌ Sparse returned NOTHING for a document's own text.")
        print("   The query encoding does not match the index encoding.")
        print("   Effect: the sparse half is dead. Hybrid search = dense-only, silently.")
        print("   Your baseline has been running on one leg.")


if __name__ == "__main__":
    main()
