"""Query language detection — the slice of cbm-system's nodes/general/utils.py
this benchmark needs. The real file also holds tool-visibility/state helpers
tied to the full agent stack (FastAPI/LangGraph), which this repo doesn't have.

Used to pick the -th/-en index suffix via config.vector.resolve_index_names().
Note: KM itself is in resolve_index_names()'s ignore_suffix list, so detected
language currently has no effect on which KM index gets queried (see the
caveat in README.md) — this exists for when that changes, or for other
libraries that do split by language.
"""

from langdetect import DetectorFactory, LangDetectException, detect

# Pinned: langdetect is otherwise non-deterministic between runs on the same text.
DetectorFactory.seed = 0

# Fallback when detect_language() cannot classify the text at all (empty, too
# short, digits only). A plain fallback value, not a second Thai/English test.
DEFAULT_LANGUAGE = "en"


def detect_language(text: str) -> str:
    """Detect the query's language, returning langdetect's ISO code.

    No Thai/English mapping here: resolve_index_names() already reduces the
    code to a bool via `language.strip().lower() in THAI_LANGS`, so "th" lands
    in the Thai bucket and everything else falls through to -en.
    """
    try:
        return detect(text)
    except LangDetectException:
        return DEFAULT_LANGUAGE
