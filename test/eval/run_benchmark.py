"""Replay the frozen 22-turn conversation through a system, write JSONL.

Same file, byte-for-byte, runs BOTH systems — only the adapter differs. That is
what makes the two result files comparable: same script, same order, same
session boundaries, same config fingerprint.

  line 1   {"_config": true, ...}   fingerprint the scorer checks before it will
                                    compare two systems
  line 2+  one line per turn

Turn order and session boundaries come from evalset.json and must not change:
the memory system is stateful, so order is itself a variable.

Usage (repo root):
    uv run python test/eval/run_benchmark.py
    uv run python test/eval/run_benchmark.py --adapter benchmark_adapter --limit 4
"""

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE))

# Files whose bytes define "the same retrieval, prompt and floor". A mismatch in
# any of these means the two systems are not running the same pipeline.
LOCKED_FILES = [
    "code/config/vector.py",
    "code/config/pinecone_core.py",
    "code/nodes/general/prompts.py",
    "code/nodes/general/utils.py",
    "code/tools/general/build.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(adapter, evalset_path: Path, evalset: dict) -> dict:
    from config.pinecone_core import EMBEDDING_MODEL
    from config.vector import HybridRRFSearcher
    from tools.general.build import RAG_RELEVANCE_FLOOR

    init = inspect.signature(HybridRRFSearcher.__init__).parameters
    rrf = inspect.signature(HybridRRFSearcher._rrf_fusion).parameters

    return {
        "_config": True,
        "system": getattr(adapter, "SYSTEM_LABEL", adapter.__name__),
        "memory_enabled": bool(getattr(adapter, "MEMORY_ENABLED", False)),
        "code_hashes": {
            f: sha256(ROOT / f) for f in LOCKED_FILES if (ROOT / f).exists()
        },
        "evalset_sha256": sha256(evalset_path),
        "evalset_version": evalset.get("version"),
        "namespace": getattr(adapter, "NAMESPACE", None),
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": getattr(adapter, "GENERATION_MODEL", None),
        "top_k": init["top_k"].default,
        "fusion_pool": init["top_n"].default,
        "rerank_top_n": init["rerank_top_n"].default,
        "rrf_k": rrf["k"].default,
        "relevance_floor": RAG_RELEVANCE_FLOOR,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def call(fn, *args):
    out = fn(*args)
    return await out if inspect.isawaitable(out) else out


REQUIRED = ["answer", "floor_triggered", "retrieved_chunk_ids"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="benchmark_adapter")
    ap.add_argument("--evalset", default=str(HERE / "evalset.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N turns only (smoke test)")
    args = ap.parse_args()

    adapter = importlib.import_module(args.adapter)
    evalset_path = Path(args.evalset)
    evalset = json.loads(evalset_path.read_text())

    if getattr(adapter, "NAMESPACE", None) != evalset["namespace"]:
        sys.exit(
            f"adapter NAMESPACE {getattr(adapter, 'NAMESPACE', None)!r} != "
            f"evalset namespace {evalset['namespace']!r}"
        )

    config = build_config(adapter, evalset_path, evalset)
    label = config["system"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else HERE / "results" / f"{label}_{ts}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    user_id = evalset["user_id"]
    turns = evalset["turns"][: args.limit] if args.limit else evalset["turns"]

    if hasattr(adapter, "reset_memory"):
        await call(adapter.reset_memory, user_id)

    print(f"system={label} memory_enabled={config['memory_enabled']} turns={len(turns)}")
    with out.open("w") as f:
        f.write(json.dumps(config, ensure_ascii=False) + "\n")

        prev_session = None
        for i, turn in enumerate(turns, 1):
            if prev_session and turn["session_id"] != prev_session:
                if hasattr(adapter, "end_session"):
                    await call(adapter.end_session, user_id, prev_session)
                print(f"   -- session boundary: {prev_session} -> {turn['session_id']}")

            record = {
                "turn_id": turn["turn_id"],
                "session_id": turn["session_id"],
                "user_id": user_id,
                "type": turn["type"],
                "question": turn["question"],
            }
            try:
                result = await call(
                    adapter.answer, turn["question"], user_id, turn["session_id"]
                )
                missing = [k for k in REQUIRED if k not in result]
                if missing:
                    record["error"] = f"adapter result missing keys: {missing}"
                else:
                    record |= result
            except Exception as exc:  # a crashed turn is data, not a reason to stop
                record["error"] = repr(exc)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            state = "ERROR" if "error" in record else (
                "refused" if record.get("floor_triggered") else "answered"
            )
            print(f"[{i:2}/{len(turns)}] {turn['turn_id']:4} {turn['type']:8} {state:8} {turn['question'][:44]}")
            prev_session = turn["session_id"]

        if prev_session and hasattr(adapter, "end_session"):
            await call(adapter.end_session, user_id, prev_session)

    print(f"\n-> {out}")
    print("next: uv run python test/eval/score_benchmark.py --baseline", out)


if __name__ == "__main__":
    asyncio.run(main())
