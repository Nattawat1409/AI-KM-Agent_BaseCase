"""Build evalset.json (v2): one Thai user, 22 scripted turns, two sessions.

  17 questions  - each written by an LLM from a different real Pinecone chunk
   3 reminders  - "did we talk about X?" — only answerable by a system with memory
   2 profile    - statements the user makes about themselves (given by the human)

Why this shape: the comparison target has bank + episodic memory. A pure
retrieval eval cannot see that at all, so the script deliberately (a) states a
profile up front, (b) asks questions across two sessions, then (c) asks the
system to recall the earlier conversation. Recall@8 on the 17 questions guards
against memory quietly damaging retrieval.

Pipeline (every LLM step is checked, nothing is trusted blindly):
  1. sample many candidate chunks from the Thai dense index, one per source
     document so the 17 questions span 17 different documents
  2. LLM keeps only chunks with substantive business/operations knowledge
     (drops bibliographies, tables of contents, credits, image-only pages)
  3. LLM writes a short casual Thai question per kept chunk
  4. LLM validator rejects questions that are vague or not answerable from
     that chunk alone
  5. seeded shuffle -> first 17 survive; session split 9 / 8
  6. 3 reminder questions written about topics from session 1

Deterministic given SEED, but LLM output is not — so the FROZEN evalset.json is
the artifact. Do not regenerate it once benchmarking has started.

Usage (repo root):
    uv run python test/eval/build_conversation_evalset.py
"""

import json
import math
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from tools.general.build import KM_LIBRARY, load_env, resolve_km_key  # noqa: E402

load_env()

from config.vector import resolve_index_names  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from pinecone import Pinecone  # noqa: E402

HERE = Path(__file__).parent
NAMESPACE = "manufactur-profession"
USER_ID = "u1"
SEED = 2026
N_QUESTIONS = 17
N_SESSION1_QUESTIONS = 9
N_CANDIDATES = 70  # over-sample: three filters below each discard some
DIM = 3072
GEN_MODEL = "gemini-2.5-flash"

# Provided by the human. Normalised from double-เ typing to the real แ (U+0E41)
# so langdetect / embeddings see proper Thai; wording is otherwise verbatim.
PROFILE_STATEMENTS = [
    "ผมคือวิศวกรหน้าใหม่ ช่วยอธิบายเรียบเรียงแบบเข้าใจง่ายโดยไม่ใช่ภาษาที่ยากไป",
    "ผมชอบให้อธิบายเป็น bullet point พร้อมประโยคสรุปด้านใต้ไม่ยาวจนเกินไป",
]

STYLE_EXAMPLES = [
    "Silo pilot วัดระดับ ทำงานไง?",
    "ต้นทุนคุณภาพโดยทั่วไปแบ่งเป็นกี่องค์ประกอบ และมีอะไรบ้าง?",
]
REMINDER_EXAMPLES = [
    "เราเคยคุยกันเรื่อง silo pilot แล้วหรือว่าชอบให้อธิบายแบบไหน",
    "เราเคยคุยกันเรื่องต้นทุนคุณภาพหรือว่าแบ่งเป็นยังไงบ้าง",
]


def llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEN_MODEL, temperature=0)


def parse_json(raw) -> list | dict:
    text = raw if isinstance(raw, str) else str(raw)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json")
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=0)
    end = max(text.rfind("]"), text.rfind("}"))
    return json.loads(text[start : end + 1])


def as_text(value) -> str:
    if isinstance(value, list):
        value = "\n".join(str(v) for v in value)
    return str(value or "").strip()


# ---------------------------------------------------------------- 1. sampling
def sample_candidates(rng: random.Random) -> list[dict]:
    """Pinecone has no random-sample call, and index.list() is insertion-ordered
    (biased to whatever was ingested first). Random unit vectors instead: each
    random direction lands in a different region of embedding space, so the
    neighbours span topics rather than ingestion time."""
    dense_index, _ = resolve_index_names(KM_LIBRARY, "Thai")
    index = Pinecone(api_key=resolve_km_key()).Index(dense_index)

    by_id: dict[str, dict] = {}
    for _ in range(60):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in vec))
        res = index.query(
            namespace=NAMESPACE,
            vector=[v / norm for v in vec],
            top_k=10,
            include_metadata=True,
        )
        for match in res.get("matches", []):
            meta = match.get("metadata") or {}
            text = as_text(meta.get("chunk_text") or meta.get("content"))
            if len(text) < 400:  # too short to support a specific question
                continue
            by_id.setdefault(
                match["id"],
                {
                    "chunk_id": match["id"],
                    "source": str(meta.get("source", "Unknown")),
                    "text": text,
                },
            )

    # One chunk per source document: 17 questions from 17 different documents,
    # instead of several questions competing over one document's content.
    by_source: dict[str, dict] = {}
    pool = list(by_id.values())
    rng.shuffle(pool)
    for chunk in pool:
        by_source.setdefault(chunk["source"], chunk)

    chosen = list(by_source.values())
    print(f"sampled {len(by_id)} unique chunks from {len(by_source)} documents")
    return chosen[:N_CANDIDATES]


# --------------------------------------------------- 2. business-relevance filter
KEEP_PROMPT = """You are curating a Q&A test set for a cement / industrial manufacturing \
company's internal knowledge base.

For each numbered excerpt decide keep=true only if it contains concrete, substantive \
knowledge an employee could ask about: a process, equipment, quality, cost, safety, \
maintenance, technology, sustainability or business fact.

keep=false for: reference lists / bibliography, table of contents, credits or logos, \
confidentiality banners, meeting logistics, pages that are mostly image descriptions, \
text too fragmentary to make sense on its own, and generic software / AI / IT \
tutorials or textbooks that are not about this company's own operations (cement, \
concrete, kilns, mills, quality, maintenance, safety, logistics, smart factory, \
sustainability, or the business side of building materials).

Return ONLY JSON: [{{"n": <number>, "keep": true|false}}]

{excerpts}"""


def filter_relevant(chunks: list[dict]) -> list[dict]:
    model, kept = llm(), []
    for i in range(0, len(chunks), 8):
        batch = chunks[i : i + 8]
        excerpts = "\n\n".join(
            f"[{j + 1}]\n{c['text'][:1500]}" for j, c in enumerate(batch)
        )
        verdicts = parse_json(model.invoke(KEEP_PROMPT.format(excerpts=excerpts)).content)
        keep = {int(v["n"]) - 1 for v in verdicts if v.get("keep")}
        kept += [c for j, c in enumerate(batch) if j in keep]
    print(f"relevance filter: kept {len(kept)}/{len(chunks)}")
    return kept


# ------------------------------------------------------- 3. question generation
ASK_PROMPT = """You are writing test questions for a Thai company's knowledge-base \
assistant.

For each numbered excerpt write ONE question, in Thai, that a busy engineer or new \
employee would actually type into a chat box.

Style: short and casual, like these (do NOT reuse these topics):
{style}

Rules:
- Answerable from that excerpt alone, about a concrete fact, number, mechanism, step \
or definition.
- Understandable by a colleague who has NOT seen the excerpt: name the subject \
explicitly, never say "this document" / "the excerpt" / "the above".
- Name the specific equipment, system, programme, company or year the fact belongs \
to, so the question has ONE clear target. Never write "the fan", "the experiment", \
"the system" without saying which one.
- Never refer to a table, figure, chart or page number, and never depend on an \
unexplained local nickname or internal code: the question must make sense on its own.
- Do not copy long distinctive phrases from the excerpt; plain wording is better.

Return ONLY JSON: [{{"n": <number>, "question": "..."}}]

{excerpts}"""


def write_questions(chunks: list[dict]) -> None:
    model = llm()
    for i in range(0, len(chunks), 6):
        batch = chunks[i : i + 6]
        excerpts = "\n\n".join(
            f"[{j + 1}]\n{c['text'][:1800]}" for j, c in enumerate(batch)
        )
        prompt = ASK_PROMPT.format(
            style="\n".join(f"- {s}" for s in STYLE_EXAMPLES), excerpts=excerpts
        )
        for item in parse_json(model.invoke(prompt).content):
            batch[int(item["n"]) - 1]["question"] = str(item["question"]).strip()


# ------------------------------------------------------------- 4. validation
CHECK_PROMPT = """You are auditing test questions. For each numbered pair, judge the \
question against its excerpt.

- answerable: the excerpt alone contains enough to answer the question.
- specific: a colleague who has NOT seen the excerpt would understand exactly what is \
being asked. Reject if the subject is generic ("the fan", "the experiment", "the \
module") or the wording is unclear, so that many different passages could match. \
Also reject any question that cites a table / figure / page number or depends on an \
unexplained code name.
- relevant: an employee of a cement / building-materials / industrial company would \
plausibly ask this about their own work. Reject generic AI / software / IT topics.

Return ONLY JSON: \
[{{"n": <number>, "answerable": true|false, "specific": true|false, "relevant": true|false}}]

{pairs}"""


def validate(chunks: list[dict]) -> list[dict]:
    model, ok = llm(), []
    pending = [c for c in chunks if c.get("question")]
    for i in range(0, len(pending), 6):
        batch = pending[i : i + 6]
        pairs = "\n\n".join(
            f"[{j + 1}] QUESTION: {c['question']}\nEXCERPT: {c['text'][:1500]}"
            for j, c in enumerate(batch)
        )
        for v in parse_json(model.invoke(CHECK_PROMPT.format(pairs=pairs)).content):
            if v.get("answerable") and v.get("specific") and v.get("relevant"):
                ok.append(batch[int(v["n"]) - 1])
    print(f"validation: kept {len(ok)}/{len(pending)}")
    return ok


# ---------------------------------------------------------------- 6. reminders
INTENTS = [
    (
        "topic_and_style",
        "asks whether the topic was discussed before AND how the user likes things explained",
        lambda t: [
            f"Confirms '{t}' was discussed earlier AND correctly states what the user "
            "asked about it at the time (their actual question). Facts about the topic, "
            "or agreeing with the user's own wording, do not count as evidence of recall",
            "States the user's preferred explanation style: simple language for a new "
            "engineer, and/or bullet points with a short summary underneath",
        ],
    ),
    (
        "topic_content",
        "asks whether the topic was discussed before AND what was said about it",
        lambda t: [
            f"Confirms '{t}' was discussed earlier AND correctly states what the user "
            "asked about it at the time (their actual question). Facts about the topic, "
            "or agreeing with the user's own wording, do not count as evidence of recall",
            f"Summarises what was said about '{t}' consistently with the earlier "
            "answer, without adding facts the earlier answer did not contain",
        ],
    ),
    (
        "topic_and_role",
        "asks whether the topic was discussed before AND who the user said they are",
        lambda t: [
            f"Confirms '{t}' was discussed earlier AND correctly states what the user "
            "asked about it at the time (their actual question). Facts about the topic, "
            "or agreeing with the user's own wording, do not count as evidence of recall",
            "States that the user said they are a new (junior) engineer",
        ],
    ),
]

REMIND_PROMPT = """Write three short Thai follow-up messages a user would send in a NEW \
chat session, referring back to something discussed earlier. Casual tone, like:
{examples}

For each item below, write one message that mentions the topic and does what the \
intent says. Use the topic words exactly as given.

{items}

Return ONLY JSON: [{{"n": <number>, "message": "..."}}]"""

TOPIC_PROMPT = """For each question give its topic as ONE natural noun phrase (2-5 words) \
exactly as a person would say it aloud in conversation, e.g. "silo pilot" or \
"ต้นทุนคุณภาพ". Never a list of keywords, never a full sentence. Use the words as they \
appear in the question. Return ONLY JSON: [{{"n": <number>, "topic": "..."}}]

{questions}"""

FALLBACK = {
    "topic_and_style": "เราเคยคุยกันเรื่อง{t}แล้วหรือว่าชอบให้อธิบายแบบไหน",
    "topic_content": "เราเคยคุยกันเรื่อง{t}ไปแล้วใช่มั้ย ตอนนั้นสรุปว่ายังไงบ้าง",
    "topic_and_role": "เราเคยคุยกันเรื่อง{t}ไปแล้วใช่มั้ย แล้วผมบอกไปว่าผมเป็นใคร",
}


def write_reminders(s1_questions: list[dict], rng: random.Random) -> list[dict]:
    picks = sorted(rng.sample(s1_questions, 3), key=lambda c: c["turn_id"])
    model = llm()

    topics = parse_json(
        model.invoke(
            TOPIC_PROMPT.format(
                questions="\n".join(f"[{i + 1}] {c['question']}" for i, c in enumerate(picks))
            )
        ).content
    )
    for item in topics:
        picks[int(item["n"]) - 1]["topic"] = str(item["topic"]).strip()

    items = "\n".join(
        f"[{i + 1}] topic: {p['topic']} | intent: {INTENTS[i][1]}"
        for i, p in enumerate(picks)
    )
    msgs = parse_json(
        model.invoke(
            REMIND_PROMPT.format(
                examples="\n".join(f"- {e}" for e in REMINDER_EXAMPLES), items=items
            )
        ).content
    )
    by_n = {int(m["n"]) - 1: str(m["message"]).strip() for m in msgs}

    reminders = []
    for i, pick in enumerate(picks):
        key, _, facts = INTENTS[i]
        text = by_n.get(i, "")
        # A reminder that doesn't name its topic can't be scored — fall back.
        # The role reminder must be in the user's voice ("ผมบอกว่าผมเป็นใคร"); an
        # LLM once flipped it to "คุณบอกว่าเป็นใคร", which asks the wrong question.
        wrong_voice = key == "topic_and_role" and "ผม" not in text
        if pick["topic"].lower() not in text.lower() or wrong_voice:
            text = FALLBACK[key].format(t=pick["topic"])
        reminders.append(
            {
                "turn_id": f"r{i + 1}",
                "session_id": f"{USER_ID}-s2",
                "type": "reminder",
                "question": text,
                "references_turn": pick["turn_id"],
                "topic": pick["topic"],
                "intent": key,
                "expected_recall": facts(pick["topic"]),
            }
        )
    return reminders


# ------------------------------------------------------------------- assemble
def main() -> None:
    rng = random.Random(SEED)

    candidates = filter_relevant(sample_candidates(rng))
    write_questions(candidates)
    good = validate(candidates)
    if len(good) < N_QUESTIONS:
        sys.exit(f"only {len(good)} valid questions, need {N_QUESTIONS} — rerun")

    rng.shuffle(good)
    chosen = good[:N_QUESTIONS]

    s1_id, s2_id = f"{USER_ID}-s1", f"{USER_ID}-s2"
    q_turns = []
    for i, c in enumerate(chosen, 1):
        q_turns.append(
            {
                "turn_id": f"q{i:02d}",
                "session_id": s1_id if i <= N_SESSION1_QUESTIONS else s2_id,
                "type": "question",
                "question": c["question"],
                "gold_chunk_id": c["chunk_id"],
                "gold_doc": c["chunk_id"].split("_")[0],
                "gold_source": c["source"],
                "gold_excerpt": c["text"][:1500],
            }
        )

    reminders = write_reminders(q_turns[:N_SESSION1_QUESTIONS], rng)

    profile = [
        {
            "turn_id": f"p{i}",
            "session_id": s1_id,
            "type": "profile",
            "question": text,
        }
        for i, text in enumerate(PROFILE_STATEMENTS, 1)
    ]

    turns = (
        profile
        + q_turns[:N_SESSION1_QUESTIONS]
        + q_turns[N_SESSION1_QUESTIONS:]
        + reminders
    )

    evalset = {
        "version": 2,
        "frozen": date.today().isoformat(),
        "language": "th",
        "namespace": NAMESPACE,
        "user_id": USER_ID,
        "seed": SEED,
        "generator_model": GEN_MODEL,
        "note": (
            "Turn order is fixed and must be replayed exactly — the system under "
            "comparison is stateful. Profile statements come first (session 1); "
            "reminders come last (session 2) so they test recall across a session "
            "boundary and past 8 unrelated intervening questions."
        ),
        "profile_statements": PROFILE_STATEMENTS,
        "sessions": [s1_id, s2_id],
        "turns": turns,
    }

    out = HERE / "evalset.json"
    out.write_text(json.dumps(evalset, ensure_ascii=False, indent=2))

    counts = {t: sum(1 for x in turns if x["type"] == t) for t in ("profile", "question", "reminder")}
    print(f"\n{len(turns)} turns  {counts}  -> {out}\n")
    for t in turns:
        extra = f"  <- {t['references_turn']}" if t["type"] == "reminder" else ""
        src = f"  [{t['gold_source'][:26]}]" if t["type"] == "question" else ""
        print(f"{t['turn_id']:4} {t['session_id'][-2:]}  {t['question'][:70]}{src}{extra}")


if __name__ == "__main__":
    main()
