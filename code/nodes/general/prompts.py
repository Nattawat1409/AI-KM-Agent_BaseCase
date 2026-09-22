"""System prompt for the baseline KM Q&A (code/km_chat.py -> service/graph_llm.py).

Deviation from cbm-system: the real nodes/general/prompts.py is a registry +
Langfuse fetcher — it holds no prompt text itself, only a key -> Langfuse-name
map (see that file's docstring: "There are deliberately no local fallbacks...
a failed fetch puts the app into maintenance mode"). Its `profile_for()`
equivalent also picks a different *persona* from (library, name_space) —
Greenie for green-industrial-knowledge, GIBI for sric*, Buildee for
library="km", CiMie otherwise — so the system prompt itself changes depending
on which namespace is queried.

This benchmark deliberately does NOT port that persona-switching. One fixed
persona is used for every namespace, so the comparison against Personalize-AI
never has "which persona fired" as a hidden confound alongside the memory
layer. There is also no Langfuse in this repo's .env, so the actual CiMie
identity/knowledge_domain text is not reachable from here at all — it lives
remotely, populated into cbm-system's `_LIVE` dict at startup, not in that
repo's git history either. The text below is written for this benchmark; it
covers the same functional requirements cbm-system's prompts do (context-only
answers, citation by Document ID, explicit "not covered" over guessing,
language-matched replies) but is NOT the real CiMie prompt and must not be
mistaken for it. See RAG_PLAN.md §3/§8 for the confound-risk note this
implies if Personalize-AI ends up using the real Langfuse-hosted prompt.
"""

SYSTEM_PROMPT = """You are a knowledge-management assistant answering questions from a company's internal knowledge base (cement manufacturing, industrial, and related professional content).

Rules — follow all of them:
1. Answer using ONLY the information in the "Context" section below. Do not use outside knowledge, even if you are confident it is correct.
2. Every factual claim must cite the Document ID(s) it came from, in the form [Document ID: N], matching the "[Document ID: N]" headers in the context.
3. If the context does not contain enough information to answer the question (fully or partially), say so plainly instead of guessing or filling gaps with outside knowledge.
4. Reply in the same language the question was asked in: a Thai question gets a Thai answer, an English question gets an English answer.

Context:
{context}
"""

# Returned directly (no LLM call) when the relevance floor in
# tools/general/build.py::RAG_RELEVANCE_FLOOR isn't met — the retrieved
# documents are treated as not covering the question at all, so this text
# replaces the answer rather than being appended to a prompt for the model to
# rephrase (which would risk the model still trying to answer from weak
# context). Keyed by the same th/en bucket resolve_index_names() uses, so the
# refusal is always in the question's own language too.
NOT_COVERED_MESSAGE = {
    "th": "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้ในฐานความรู้",
    "en": "The knowledge base does not contain information relevant to this question.",
}
