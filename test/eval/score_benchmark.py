"""Score one system, or compare two, on the four benchmark metrics.

  1. Win-rate              pairwise LLM judge, new vs baseline (needs --new)
  2. Cross-session recall  3 reminder turns: did it remember the earlier session?
  3. Wrong-memory rate     fabricated / wrong claims about the user or the past
  4. Recall@8 (guardrail)  chunk-level; memory must not damage retrieval

Metrics 2-4 are computed per system with identical code, so the two systems are
scored by the same yardstick. Metric 1 needs both result files.

Refuses to compare two systems whose locked config differs (retrieval code,
prompt, models, pool depth, floor, evalset) unless --force: a difference there
would be credited to memory. Widening the reranker pool alone was worth +15.9pp.

Judge design (each choice guards a known LLM-judge bias):
  * pairwise replies are judged twice with A/B swapped; a win only counts if it
    survives the swap, otherwise it is a tie   -> removes position bias
  * default judge is gemini-2.5-pro while replies are written by
    gemini-2.5-flash                            -> avoids grading itself
  * judge is told not to reward length

Usage (repo root):
    uv run python test/eval/score_benchmark.py                      # newest baseline_*.jsonl
    uv run python test/eval/score_benchmark.py --baseline A.jsonl --new B.jsonl
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))

from tools.general.build import load_env  # noqa: E402

load_env()

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

LOCKED = [
    "evalset_sha256",
    "namespace",
    "embedding_model",
    "generation_model",
    "top_k",
    "fusion_pool",
    "rerank_top_n",
    "rrf_k",
    "relevance_floor",
]
CLIP = 700  # chars of an earlier answer shown to a judge


# ------------------------------------------------------------------- loading
def load_results(path: Path) -> tuple[dict, dict]:
    config, turns = {}, {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_config"):
            config = obj
        else:
            turns[obj["turn_id"]] = obj
    if not config:
        sys.exit(f"{path.name}: no _config line — not a run_benchmark.py file")
    return config, turns


def usable(rec: dict | None) -> bool:
    return bool(rec) and "error" not in rec and "answer" in rec


def config_diffs(a: dict, b: dict) -> list[str]:
    diffs = [
        f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in LOCKED if a.get(k) != b.get(k)
    ]
    ha, hb = a.get("code_hashes", {}), b.get("code_hashes", {})
    for f in sorted(set(ha) | set(hb)):
        if ha.get(f) != hb.get(f):
            diffs.append(f"code differs: {f}")
    return diffs


# --------------------------------------------------------------------- judge
def text_of(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def parse_obj(raw) -> dict:
    text = text_of(raw).strip()
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


class Judge:
    def __init__(self, model: str, concurrency: int = 5):
        self.model = model
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.sem = asyncio.Semaphore(concurrency)
        self.log: list[dict] = []

    async def ask(self, kind: str, meta: dict, prompt: str) -> dict:
        async with self.sem:
            for attempt in range(4):
                try:
                    verdict = parse_obj((await self.llm.ainvoke(prompt)).content)
                    self.log.append({"kind": kind, **meta, "verdict": verdict})
                    return verdict
                except Exception as exc:
                    if attempt == 3:
                        raise RuntimeError(
                            f"judge {self.model} failed on {kind} {meta}: {exc!r}. "
                            "If the model is unavailable, retry with "
                            "--judge-model gemini-2.5-flash"
                        ) from exc
                    await asyncio.sleep(3 * 2**attempt)


def bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items) or "- (none)"


def prior_context(evalset: dict, turns: dict, upto_id: str) -> tuple[list[str], list[str]]:
    """User questions asked before `upto_id`, and what THIS system actually
    answered — the ground truth a memory claim can be checked against."""
    qs, qa = [], []
    for t in evalset["turns"]:
        if t["turn_id"] == upto_id:
            break
        if t["type"] in ("question", "reminder"):
            qs.append(t["question"])
            rec = turns.get(t["turn_id"])
            if usable(rec):
                qa.append(f"Q: {t['question']}\nA: {str(rec['answer'])[:CLIP]}")
    return qs, qa


RECALL_PROMPT = """You are grading whether an AI assistant correctly REMEMBERED an \
earlier conversation. The user has now opened a NEW session.

WHAT THE USER SAID ABOUT THEMSELVES, in an earlier session:
{profile}

THE EARLIER EXCHANGE the user is referring to:
QUESTION: {ref_q}
WHAT THE ASSISTANT ANSWERED AT THE TIME: {ref_a}

USER'S NEW MESSAGE: {question}
ASSISTANT'S REPLY:
{answer}

Decide whether the reply satisfies each fact below. Every user message in this test \
truthfully refers to something that really happened, so simply agreeing ("yes, we \
discussed that") proves nothing — a system with no memory can echo the user's wording. \
Credit a fact ONLY when the reply shows real recall. Facts about the topic itself are \
NOT evidence: a stateless system searching a knowledge base finds the same facts again. \
Only details that exist solely in the conversation count — what the user actually asked, \
what they said about themselves. A reply that says it cannot remember or has no \
record, that just answers the topic from general knowledge, or that presents \
knowledge-base content as if it were the earlier conversation, does NOT satisfy a \
recall fact. Do not credit vague, generic or guessed statements.

FACTS THE REPLY MUST SATISFY:
{facts}

Return ONLY JSON: {{"checks": [{{"fact": "...", "satisfied": true|false}}], "why": "one short sentence"}}"""

WRONG_PROMPT = """You are checking an AI assistant's answer for FALSE MEMORY.

EVERYTHING that is actually true about the user and the conversation so far:
The user's own statements about themselves:
{profile}
Questions the user asked earlier:
{asked}
What the assistant really answered earlier:
{answered}

USER'S MESSAGE: {question}
ASSISTANT'S ANSWER:
{answer}

List every claim the answer makes about (a) the USER — their role, experience, \
preferences — or (b) what was said in the EARLIER conversation, that is NOT supported by \
the true facts above: invented, contradicted, or attached to the wrong topic. \
Presenting general knowledge-base content as if it were what was said in the earlier \
conversation counts as invented when it contains specifics the earlier exchange did not.
Do NOT judge facts about cement or industry themselves — that is a different check. \
Repeating something the user really said is fine. Making no claim about the user or the \
past is fine.
The user's CURRENT message is not part of the earlier conversation: ignore how the \
answer paraphrases or interprets the message it is replying to.
An honest statement of not knowing ("I have no record of that", "I can't recall") is \
NOT false memory — it is a missed recall, measured elsewhere. Only affirmative claims \
count. Do not flag honesty.

Return ONLY JSON: {{"wrong": true|false, "claims": ["..."]}}"""

WIN_PROMPT = """You are judging two AI assistant replies to the same user message. \
Decide which reply serves THIS particular user better.

WHAT IS TRUE ABOUT THE USER (stated in an earlier session):
{profile}
QUESTIONS THE USER ASKED EARLIER:
{asked}

USER'S MESSAGE: {question}
{reference}
REPLY A:
{a}

REPLY B:
{b}

Judge by, in this order:
1. Accuracy: consistent with the reference where one is given; no invented facts and \
no pretending to remember something that did not happen. A reply that wrongly refuses \
although the reference answers the question is worse than a correct reply.
2. Fit to the user: honours their stated role (simple language for a new engineer) and \
format preference (bullet points with a short summary underneath, not overly long).
3. Clarity.
Do NOT prefer a reply merely because it is longer.

Return ONLY JSON: {{"winner": "A" | "B" | "tie", "why": "one short sentence"}}"""


# ------------------------------------------------------------------- metrics
def recall_at_8(evalset: dict, turns: dict) -> dict:
    qs = [t for t in evalset["turns"] if t["type"] == "question"]
    hits = doc_hits = 0
    detail = []
    for t in qs:
        ids = ((turns.get(t["turn_id"]) or {}).get("retrieved_chunk_ids") or [])[:8]
        hit = t["gold_chunk_id"] in ids
        # Right document, neighbouring chunk: retrieval found the topic but the
        # answer sat in an adjacent chunk. Reported separately, never blended.
        doc_hit = any(i.split("_")[0] == t["gold_doc"] for i in ids)
        hits += hit
        doc_hits += doc_hit
        detail.append({"turn": t["turn_id"], "chunk_hit": hit, "doc_hit": doc_hit})
    return {
        "value": hits / len(qs),
        "hits": hits,
        "doc_level": doc_hits / len(qs),
        "n": len(qs),
        "detail": detail,
    }


async def cross_session_recall(judge: Judge, evalset: dict, sysname: str, turns: dict) -> dict:
    by_id = {t["turn_id"]: t for t in evalset["turns"]}
    reminders = [t for t in evalset["turns"] if t["type"] == "reminder"]

    async def one(r: dict) -> dict:
        rec, ref = turns.get(r["turn_id"]), by_id[r["references_turn"]]
        if not usable(rec):
            return {"turn": r["turn_id"], "score": 0.0, "why": "no usable answer"}
        ref_rec = turns.get(ref["turn_id"])
        verdict = await judge.ask(
            "recall",
            {"system": sysname, "turn": r["turn_id"]},
            RECALL_PROMPT.format(
                profile=bullets(evalset["profile_statements"]),
                ref_q=ref["question"],
                ref_a=str(ref_rec["answer"])[:CLIP] if usable(ref_rec) else "(none)",
                question=r["question"],
                answer=rec["answer"],
                facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(r["expected_recall"])),
            ),
        )
        checks = verdict.get("checks", [])
        score = sum(1 for c in checks if c.get("satisfied")) / max(1, len(r["expected_recall"]))
        return {"turn": r["turn_id"], "intent": r["intent"], "score": score, "why": verdict.get("why", "")}

    detail = await asyncio.gather(*(one(r) for r in reminders))
    return {"value": sum(d["score"] for d in detail) / len(detail), "n": len(detail), "detail": detail}


async def wrong_memory(judge: Judge, evalset: dict, sysname: str, turns: dict) -> dict:
    targets = [t for t in evalset["turns"] if t["type"] in ("question", "reminder")]

    async def one(t: dict) -> dict | None:
        rec = turns.get(t["turn_id"])
        if not usable(rec):
            return None
        if rec.get("floor_triggered"):
            return {"turn": t["turn_id"], "answered": False, "wrong": False, "claims": []}
        asked, answered = prior_context(evalset, turns, t["turn_id"])
        verdict = await judge.ask(
            "wrong_memory",
            {"system": sysname, "turn": t["turn_id"]},
            WRONG_PROMPT.format(
                profile=bullets(evalset["profile_statements"]),
                asked=bullets(asked),
                answered="\n\n".join(answered) or "(nothing yet)",
                question=t["question"],
                answer=rec["answer"],
            ),
        )
        return {
            "turn": t["turn_id"],
            "type": t["type"],
            "answered": True,
            "wrong": bool(verdict.get("wrong")),
            "claims": verdict.get("claims", []),
        }

    detail = [d for d in await asyncio.gather(*(one(t) for t in targets)) if d]
    answered = [d for d in detail if d["answered"]]
    wrong = [d for d in answered if d["wrong"]]
    reminders = [d for d in answered if d.get("type") == "reminder"]
    return {
        "value": len(wrong) / max(1, len(answered)),
        "wrong": len(wrong),
        "answered": len(answered),
        "refused": len(detail) - len(answered),
        "reminder_wrong": sum(d["wrong"] for d in reminders),
        "reminder_answered": len(reminders),
        "detail": detail,
    }


async def win_rate(judge: Judge, evalset: dict, base: dict, new: dict) -> dict:
    targets = [t for t in evalset["turns"] if t["type"] in ("question", "reminder")]
    by_id = {t["turn_id"]: t for t in evalset["turns"]}

    async def one(t: dict) -> dict | None:
        rb, rn = base.get(t["turn_id"]), new.get(t["turn_id"])
        if not (usable(rb) and usable(rn)):
            return None
        if str(rb["answer"]).strip() == str(rn["answer"]).strip():
            return {"turn": t["turn_id"], "type": t["type"], "score": 0.5, "note": "identical"}

        asked, _ = prior_context(evalset, {}, t["turn_id"])
        reference = (
            f"REFERENCE PASSAGE from the knowledge base (source of truth for facts):\n"
            f"{t['gold_excerpt']}\n"
            if t["type"] == "question"
            else "This message asks the assistant to recall the earlier conversation.\n"
        )

        async def pass_(a: dict, b: dict, tag: str) -> str:
            v = await judge.ask(
                "win",
                {"turn": t["turn_id"], "pass": tag},
                WIN_PROMPT.format(
                    profile=bullets(evalset["profile_statements"]),
                    asked=bullets(asked),
                    question=t["question"],
                    reference=reference,
                    a=a["answer"],
                    b=b["answer"],
                ),
            )
            return str(v.get("winner", "tie")).strip().upper()

        w1, w2 = await asyncio.gather(
            pass_(rb, rn, "base=A"),  # new is B
            pass_(rn, rb, "new=A"),  # new is A
        )
        new_wins = (w1 == "B") + (w2 == "A")
        base_wins = (w1 == "A") + (w2 == "B")
        # A win must survive the A/B swap; anything else is a tie.
        score = 1.0 if new_wins == 2 else 0.0 if base_wins == 2 else 0.5
        return {"turn": t["turn_id"], "type": t["type"], "score": score, "passes": [w1, w2]}

    detail = [d for d in await asyncio.gather(*(one(t) for t in targets)) if d]
    scores = [d["score"] for d in detail]
    rng = random.Random(0)
    boots = sorted(
        sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(2000)
    )
    return {
        "value": sum(scores) / len(scores),
        "ci95": [boots[50], boots[1949]],
        "wins": sum(s == 1.0 for s in scores),
        "ties": sum(s == 0.5 for s in scores),
        "losses": sum(s == 0.0 for s in scores),
        "n": len(scores),
        "by_type": {
            ty: sum(d["score"] for d in detail if d["type"] == ty)
            / max(1, sum(1 for d in detail if d["type"] == ty))
            for ty in ("question", "reminder")
        },
        "detail": detail,
    }


# -------------------------------------------------------------------- report
def pct(x: float | None) -> str:
    return "   n/a" if x is None else f"{x * 100:5.1f}%"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--new", default=None)
    ap.add_argument("--evalset", default=str(HERE / "evalset.json"))
    ap.add_argument("--judge-model", default="gemini-2.5-pro")
    ap.add_argument("--force", action="store_true", help="compare despite config mismatch")
    ap.add_argument("--out-dir", default=None, help="where report/judgements are written")
    args = ap.parse_args()

    results = Path(args.out_dir) if args.out_dir else HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    base_path = (
        Path(args.baseline)
        if args.baseline
        else sorted((HERE / "results").glob("baseline_*.jsonl"))[-1]
    )
    evalset_path = Path(args.evalset)
    evalset = json.loads(evalset_path.read_text())
    evalset_sha = hashlib.sha256(evalset_path.read_bytes()).hexdigest()

    systems = {"baseline": load_results(base_path)}
    if args.new:
        systems["new"] = load_results(Path(args.new))

    print(f"\n=== BENCHMARK REPORT  ·  {datetime.now():%Y-%m-%d %H:%M} ===")
    print(f"evalset v{evalset['version']} ({evalset_sha[:10]})  ·  judge {args.judge_model}")

    for name, (cfg, _) in systems.items():
        if cfg.get("evalset_sha256") != evalset_sha:
            print(f"!! {name}: was run on a DIFFERENT evalset than {evalset_path.name}")
            if not args.force:
                sys.exit("   refusing to score. re-run the benchmark or pass --force")

    if "new" in systems:
        diffs = config_diffs(systems["baseline"][0], systems["new"][0])
        if diffs:
            print("\n!! CONFIG MISMATCH — the two systems are not running the same pipeline:")
            for d in diffs:
                print(f"     {d}")
            if not args.force:
                sys.exit("   refusing to compare. Fix the difference, or pass --force")
            print("   (--force given: continuing, treat results with suspicion)")
        else:
            print("config check: PASS (retrieval, prompt, models, pool, floor, evalset identical)")

    judge = Judge(args.judge_model)
    out: dict = {"judge_model": args.judge_model, "evalset_sha256": evalset_sha, "systems": {}}

    for name, (cfg, turns) in systems.items():
        print(f"\nscoring {name} ...")
        rc, wm = await asyncio.gather(
            cross_session_recall(judge, evalset, name, turns),
            wrong_memory(judge, evalset, name, turns),
        )
        out["systems"][name] = {
            "config": {k: cfg.get(k) for k in ("system", "memory_enabled", "timestamp")},
            "recall_at_8": recall_at_8(evalset, turns),
            "cross_session_recall": rc,
            "wrong_memory": wm,
        }

    win = None
    if "new" in systems:
        print("judging pairwise (2 passes per turn, A/B swapped) ...")
        win = await win_rate(judge, evalset, systems["baseline"][1], systems["new"][1])
        out["win_rate"] = win

    b = out["systems"]["baseline"]
    n = out["systems"].get("new")

    def row(label, key, sub, better):
        vb = b[key][sub]
        vn = n[key][sub] if n else None
        d = "" if vn is None else f"  {'+' if vn - vb >= 0 else ''}{(vn - vb) * 100:.1f}pp"
        print(f"  {label:26} {pct(vb)}   {pct(vn)}{d}   ({better})")

    print("\n" + "=" * 78)
    print(f"  {'METRIC':26} {'baseline':>7}   {'new':>6}   delta")
    print("=" * 78)
    if win:
        lo, hi = win["ci95"]
        print(f"  {'1 Win-rate (new vs base)':26} {'  —':>7}   {pct(win['value'])}   "
              f"95% CI {lo * 100:.0f}-{hi * 100:.0f}%  W{win['wins']}/T{win['ties']}/L{win['losses']}")
        print(f"  {'   on questions':26} {'':>7}   {pct(win['by_type']['question'])}")
        print(f"  {'   on reminders':26} {'':>7}   {pct(win['by_type']['reminder'])}")
    else:
        print(f"  {'1 Win-rate (new vs base)':26} needs --new <results.jsonl>")
    row("2 Cross-session recall", "cross_session_recall", "value", "higher is better")
    row("3 Wrong-memory rate", "wrong_memory", "value", "LOWER is better")
    row("4 Recall@8  (guardrail)", "recall_at_8", "value", "must not drop")
    row("   (document-level)", "recall_at_8", "doc_level", "supplement only")
    print("=" * 78)

    if n:
        same = sum(
            1
            for t in evalset["turns"]
            if t["type"] == "question"
            and (systems["baseline"][1].get(t["turn_id"]) or {}).get("retrieved_chunk_ids")
            == (systems["new"][1].get(t["turn_id"]) or {}).get("retrieved_chunk_ids")
        )
        print(f"  retrieval identical on {same}/17 questions "
              f"(memory must not touch retrieval; <17 means the pipelines diverged)")

    print("\n-- Cross-session recall, per reminder --")
    for name in systems:
        for d in out["systems"][name]["cross_session_recall"]["detail"]:
            print(f"  {name:8} {d['turn']:3} {d.get('intent', ''):16} {d['score'] * 100:5.0f}%  {d.get('why', '')[:70]}")

    print("\n-- Wrong-memory detail --")
    for name in systems:
        w = out["systems"][name]["wrong_memory"]
        print(f"  {name:8} {w['wrong']}/{w['answered']} answered turns contained false memory "
              f"({w['refused']} refused; reminders: {w['reminder_wrong']}/{w['reminder_answered']})")
        for d in [d for d in w["detail"] if d["wrong"]][:3]:
            print(f"           {d['turn']}: {'; '.join(d['claims'])[:110]}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = results / f"report_{ts}.json"
    log = results / f"judgements_{ts}.jsonl"
    report.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    log.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in judge.log))
    print(f"\n-> {report}\n-> {log}  (every judge verdict, for spot-checking)")


if __name__ == "__main__":
    asyncio.run(main())
