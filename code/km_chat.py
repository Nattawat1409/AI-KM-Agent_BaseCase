"""KM Q&A CLI — the generation-capable counterpart to km_search.py (which
stays retrieval-only, per BASELINE_PLAN.md). Runs one query end to end:
detect language -> retrieve -> relevance-floor check -> generate -> print.

No direct cbm-system equivalent, same reasoning as km_search.py: the real app
drives this through controller/llm.py -> service/graph_llm.py's LangGraph
workflow (see service/graph_llm.py's docstring for the deviations). This is a
plain CLI stand-in, kept flat at code/ root.

Usage:
    PYTHONPATH=code python code/km_chat.py "<query>" "<namespace>"

Default namespace matches km_search.py's (manufactur-profession) — see that
file's docstring for why.
"""

from tools.general.build import load_env

load_env()

import asyncio
import sys

from service.graph_llm import answer_query

DEFAULT_NAMESPACE = "manufactur-profession"


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "ปูนซีเมนต์ปอร์ตแลนด์คืออะไร"
    name_space = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NAMESPACE

    result = await answer_query(query, name_space)

    print(
        f"[language={result['language']}] "
        f"[indexes={', '.join(result['indexes_used'])}] "
        f"[top_score={result['top_score']:.4f}] "
        f"[floor_triggered={result['floor_triggered']}]"
    )
    print()
    print(result["answer"])

    if result["sources"]:
        print()
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['source']} (score={s['score']:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
