import ast
import asyncio
import functools
import os
import random
import zlib
import re
from typing import Any, List, Dict, Literal, Optional, Union
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from config.pinecone_core import (
    get_pinecone_client,
    get_embedding_client,
    get_reranker_client,
)

# --- Helper Functions ---

THAI_LANGS = {"thai", "th", "ไทย"}
_LANG_SUFFIX_RE = re.compile(r"-(th|en)$", re.IGNORECASE)
ignore_suffix: list[str] = ["mro-mrs", "km"]


def resolve_index_names(base_index: str, language: str) -> tuple[str, str]:
    base_index = _LANG_SUFFIX_RE.sub("", base_index or "")
    if base_index in ignore_suffix:
        return f"{base_index}-dense", f"{base_index}-sparse"
    is_thai = (language or "Thai").strip().lower() in THAI_LANGS
    suffix = "-th" if is_thai else "-en"
    return f"{base_index}{suffix}-dense", f"{base_index}{suffix}-sparse"


def _normalize_pages(value: Any) -> List[str]:
    """Coerce a page value into a list of strings.

    Page metadata round-trips through ``str()`` in _format_pinecone_match and
    back through ``ast.literal_eval`` below, so a numeric ``original_pages``
    ("5", 5, [1, 2]) comes back as int/float rather than the list of strings the
    rest of the pipeline assumes -- nodes/general/utils.map_return_sources sorts
    these with ``str.isdigit``. Normalise once, where pages are produced.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    pages: List[str] = []
    for page in value:
        if page is None:
            continue
        if isinstance(page, float) and page.is_integer():
            page = int(page)
        text = str(page).strip()
        if text:
            pages.append(text)
    return pages


def _format_pinecone_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to standardize Pinecone match results across all searchers."""
    meta = match.get("metadata", {})
    chunk_text = meta.get("chunk_text", "")
    content = meta.get("content", "")
    return {
        "id": match["id"],
        "score": match.get("score", 0.0),
        "text": chunk_text or content,
        "source": meta.get("source", "Unknown"),
        "page": str(meta.get("original_pages", "N/A")),
        "link": meta.get("link", ""),
        "metadata": meta,
    }


# --- Searcher Classes ---


class BaseSearcher:
    """Base class providing uniform formatting for all searchers."""

    def _format_document(
        self, idx: int, result: Dict, score_key: str = "score"
    ) -> Dict[str, Any]:
        """Formats text content for AI/LLM ingestion."""
        text = result.get("text", "")
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        elif not isinstance(text, str):
            text = str(text)

        # A1: contextual_text is already indexed on the chunk's metadata but was
        # never read; prepending it here improves grounding/citation quality with
        # no reindex. `result["metadata"]` only exists on the first format pass
        # (straight from _format_pinecone_match) -- ENSEMBLE mode's post-rerank
        # pass reuses this method's own output as `result`, which has no
        # "metadata" key, so this can't double-prepend on that second pass.
        contextual_text = (result.get("metadata") or {}).get("contextual_text", "")
        if contextual_text and contextual_text.strip():
            contextual_text = contextual_text.strip()
            text = f"{contextual_text}\n\n{text}" if text else contextual_text

        source_name = result.get("source", "Unknown")
        page = result.get("page", "N/A")
        parsed_page: Any = []
        link = result.get("link", "")

        if isinstance(page, list):
            parsed_page = page
        elif isinstance(page, str) and page and page != "N/A":
            try:
                parsed_page = ast.literal_eval(page)
            except (ValueError, SyntaxError):
                parsed_page = []

        final_page: List[str] = _normalize_pages(parsed_page)

        header = f"[Document ID: {idx}] | Source: {source_name} | Page: {page if page != 'N/A' else 'N/A'}\n"

        return {
            "id": idx,
            "content": header + text,
            "text": text,
            "source": source_name,
            "page": final_page,
            "link": link,
            "score": result.get(score_key, 0.0),
        }

    def _extract_images(self, idx: int, result: Dict) -> List[Dict[str, Any]]:
        """Extracts and formats image metadata."""
        images = []
        meta = result.get("metadata", {})
        img_paths = meta.get("image_paths", [])

        if isinstance(img_paths, str):
            try:
                import json

                img_paths = json.loads(img_paths.replace("'", '"'))
            except Exception as e:
                print(f"Failed to parse image paths: {e}")
                img_paths = [img_paths] if img_paths else []

        page = result.get("page", "N/A")
        parsed_page: Any = []
        if page and page != "N/A":
            try:
                parsed_page = ast.literal_eval(page) if isinstance(page, str) else page
            except (ValueError, SyntaxError):
                # non-literal page labels ("1-2", "cover") are not fatal here
                parsed_page = []
        final_page = _normalize_pages(parsed_page)

        for img_path in img_paths:
            if img_path:
                images.append(
                    {
                        "id": idx,
                        "source": result.get("source", "Unknown"),
                        "page": final_page,
                        "link": img_path,
                    }
                )

        return images


class DenseSearcher(BaseSearcher):
    def __init__(
        self,
        name_space: str = "thai-namespace",
        index_name: str = "general-knowledge-dense",
        top_k: int = 20,
        api_key: Optional[str] = None,
    ):
        self.name_space = name_space
        self.dense_index_name = index_name
        self.top_k = top_k
        self.api_key = api_key or os.getenv("PINECONE_CIMIE")

    def search_dense(self, query: str, filters: dict = None) -> List[Dict[str, Any]]:
        """Perform dense (semantic) vector search"""
        try:
            embedding_client = get_embedding_client()
            dense_vec = embedding_client.embed_query(query)

            dense_client = get_pinecone_client(api_key=self.api_key)
            dense_index = dense_client.Index(self.dense_index_name)

            response = dense_index.query(
                namespace=self.name_space,
                top_k=self.top_k,
                vector=dense_vec,
                include_metadata=True,
                filter=filters,
            )

            return [
                _format_pinecone_match(match) for match in response.get("matches", [])
            ]

        except Exception as e:
            print(f"Dense search failed: {e}")
            return []

    async def search(
        self, search_input: str, filters: dict = None
    ) -> List[Dict[str, Any]]:
        raw_results = await asyncio.to_thread(self.search_dense, search_input, filters)
        return [
            self._format_document(idx, res, "score")
            for idx, res in enumerate(raw_results, 1)
        ]

    async def search_with_images(
        self, search_input: str, filters: dict = None
    ) -> Dict[str, Any]:
        raw_results = await asyncio.to_thread(self.search_dense, search_input, filters)
        documents, images = [], []

        for idx, res in enumerate(raw_results, 1):
            documents.append(self._format_document(idx, res, "score"))
            images.extend(self._extract_images(idx, res))

        return {"documents": documents, "images": images}


class SparseSearcher(BaseSearcher):
    def __init__(
        self,
        name_space: str = "thai-namespace",
        index_name: str = "general-knowledge-sparse",
        top_k: int = 20,
        api_key: Optional[str] = None,
    ):
        self.name_space = name_space
        self.sparse_index_name = index_name
        self.top_k = top_k
        self.api_key = api_key or os.getenv("PINECONE_CIMIE")

    def _generate_sparse_vector(self, query: str) -> Dict[str, Any]:
        """Generate sparse vector using CRC32 hashing for keyword-based search"""
        clean_q = re.sub(r"[^\w\s]", "", query.lower())
        tokens = list(
            set(clean_q.split() + [clean_q[i : i + 3] for i in range(len(clean_q) - 2)])
        )
        indices = [zlib.crc32(t.encode("utf-8")) for t in tokens]
        sparse_vec = {"indices": indices, "values": [1.0] * len(indices)}
        return sparse_vec

    def search_sparse(self, query: str, filters: dict = None) -> List[Dict[str, Any]]:
        """Perform sparse (keyword-based) search"""
        try:
            sparse_vec = self._generate_sparse_vector(query)
            sparse_client = get_pinecone_client(api_key=self.api_key)
            sparse_index = sparse_client.Index(self.sparse_index_name)

            response = sparse_index.query(
                namespace=self.name_space,
                top_k=self.top_k,
                sparse_vector=sparse_vec,
                include_metadata=True,
                filter=filters,
            )

            return [
                _format_pinecone_match(match) for match in response.get("matches", [])
            ]

        except Exception as e:
            print(f"Sparse search failed: {e}")
            return []

    async def search(
        self, search_input: str, filters: dict = None
    ) -> List[Dict[str, Any]]:
        raw_results = await asyncio.to_thread(self.search_sparse, search_input, filters)
        return [
            self._format_document(idx, res, "score")
            for idx, res in enumerate(raw_results, 1)
        ]

    async def search_with_images(
        self, search_input: str, filters: dict = None
    ) -> Dict[str, Any]:
        raw_results = await asyncio.to_thread(self.search_sparse, search_input, filters)
        documents, images = [], []

        for idx, res in enumerate(raw_results, 1):
            documents.append(self._format_document(idx, res, "score"))
            images.extend(self._extract_images(idx, res))

        return {"documents": documents, "images": images}


class HybridRRFSearcher(BaseSearcher):
    """Uses Composition instead of Inheritance to manage underlying searchers."""

    def __init__(
        self,
        name_space: str = "thai-namespace",
        index_name_dense: str = "general-knowledge-dense",
        index_name_sparse: str = "general-knowledge-sparse",
        api_key_dense: Optional[str] = None,
        api_key_sparse: Optional[str] = None,
        top_k: int = 20,
        top_n: int = 8,
        rerank: bool = False,
        rerank_top_n: int = 8,
    ):
        self.dense_searcher = DenseSearcher(
            name_space=name_space,
            index_name=index_name_dense,
            top_k=top_k,
            api_key=api_key_dense,
        )
        self.sparse_searcher = SparseSearcher(
            name_space=name_space,
            index_name=index_name_sparse,
            top_k=top_k,
            api_key=api_key_sparse,
        )
        self.top_n = top_n
        self.rerank = rerank
        self.rerank_top_n = rerank_top_n
        self.reranker = get_reranker_client() if rerank else None  # C2: shared, top_n=8

    def _rrf_fusion(
        self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) algorithm to merge dense and sparse search results."""
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            docs[doc_id] = doc

        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in docs:
                docs[doc_id] = doc

        # Sort by RRF score descending
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_scores[: self.top_n]:
            doc = docs[doc_id].copy()  # Copy to avoid mutating original dictionary
            doc["rrf_score"] = rrf_score
            results.append(doc)

        return results

    async def _rerank_documents(self, docs: List[Dict], query: str) -> List[Dict]:
        """A2: rerank the RRF-fused pool, same pattern EnsembleSearcher already
        uses for its own post-fanout rerank (search() below). No-op when this
        searcher was constructed with rerank=False."""
        if not self.rerank or not docs:
            return docs

        input_docs = [
            Document(page_content=d.get("text") or d.get("content") or "", metadata=d)
            for d in docs
        ]
        reranked = await asyncio.to_thread(
            self.reranker.compress_documents, documents=input_docs, query=query
        )
        final = []
        for idx, doc in enumerate(reranked[: self.rerank_top_n], 1):
            res = dict(doc.metadata)
            res["text"] = doc.page_content
            final.append(self._format_document(idx, res, "relevance_score"))
        return final

    async def search_raw(
        self, search_input: str, filters: dict = None
    ) -> List[Dict[str, Any]]:
        """RRF-fused results with their original Pinecone metadata dict still
        attached (unlike search(), which formats through _format_document and
        drops "metadata" -- see A1). Used by SummaryNamespaceClient, which
        needs to read raw fields like hash_file/summary_text directly rather
        than the LLM-facing content/text shape."""
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(self.dense_searcher.search_dense, search_input, filters),
            asyncio.to_thread(
                self.sparse_searcher.search_sparse, search_input, filters
            ),
        )
        return self._rrf_fusion(dense_results, sparse_results, k=60)

    async def search(
        self, search_input: str, filters: dict = None
    ) -> List[Dict[str, Any]]:
        rrf_results = await self.search_raw(search_input, filters)
        documents = [
            self._format_document(idx, res, "rrf_score")
            for idx, res in enumerate(rrf_results, 1)
        ]
        return await self._rerank_documents(documents, search_input)

    async def search_with_images(
        self, search_input: str, filters: dict = None
    ) -> Dict[str, Any]:
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(self.dense_searcher.search_dense, search_input, filters),
            asyncio.to_thread(
                self.sparse_searcher.search_sparse, search_input, filters
            ),
        )
        rrf_results = self._rrf_fusion(dense_results, sparse_results, k=60)

        documents, images = [], []
        for idx, res in enumerate(rrf_results, 1):
            documents.append(self._format_document(idx, res, "rrf_score"))
            images.extend(self._extract_images(idx, res))

        if not self.rerank:
            return {"documents": documents, "images": images}

        reranked_documents = await self._rerank_documents(documents, search_input)
        kept_sources = {d["source"] for d in reranked_documents}
        images = [img for img in images if img.get("source") in kept_sources]
        return {"documents": reranked_documents, "images": images}


class EnsembleSearcher(BaseSearcher):
    def __init__(
        self,
        index_names: Union[str, List[str]],
        name_spaces: Union[str, List[str]] = "thai-namespace",
        language: str = "Thai",
        top_k: int = 20,
        top_n: int = 8,
        top_n_per_prof: int = 3,
        min_local_score: float = 0.015,
        api_key_dense: Optional[str] = None,
        api_key_sparse: Optional[str] = None,
    ):
        self.top_n_per_prof = top_n_per_prof
        self.min_local_score = min_local_score
        self.reranker = get_reranker_client()
        self.searchers = []

        if isinstance(index_names, str):
            index_names = [index_names]
        if isinstance(name_spaces, str):
            name_spaces = [name_spaces]

        for name in index_names:
            dense_idx, sparse_idx = resolve_index_names(name, language)
            for ns in name_spaces:
                s = HybridRRFSearcher(
                    name_space=ns,
                    index_name_dense=dense_idx,
                    index_name_sparse=sparse_idx,
                    api_key_dense=api_key_dense,
                    api_key_sparse=api_key_sparse,
                    top_k=top_k,
                    top_n=top_n,
                )
                self.searchers.append(s)

        self.reranker = get_reranker_client()

    def _get_top_per_index(self, all_index_results: List[List[Dict]]) -> List[Dict]:
        diverse_results = []
        for index_results in all_index_results:
            if not index_results:
                continue

            # Simplified: This safely handles empty results or low scores without crashing
            valid_docs = [
                doc
                for doc in index_results
                if doc.get("score", 0.0) >= self.min_local_score
            ]
            diverse_results.extend(valid_docs[: self.top_n_per_prof])

        return diverse_results

    async def search(
        self, search_input: str, filters: dict = None
    ) -> List[Dict[str, Any]]:
        # 1. Parallel Search across all professions
        tasks = [s.search(search_input, filters) for s in self.searchers]
        all_profession_results = await asyncio.gather(*tasks)

        # 2. Stratified Sampling with Safety Check
        diverse_candidates = self._get_top_per_index(all_profession_results)

        if not diverse_candidates:
            return []

        # 3. Convert to LangChain Documents for the Reranker
        input_docs = []
        for d in diverse_candidates:
            content = d.get("text") or d.get("content") or ""
            input_docs.append(Document(page_content=content, metadata=d))

        # 4. The Reranker (Sorting the diverse, validated pool)
        reranked_docs = self.reranker.compress_documents(
            documents=input_docs, query=search_input
        )

        # 5. Final Formatting
        documents = []
        for idx, doc in enumerate(reranked_docs, 1):
            res = doc.metadata
            res["text"] = doc.page_content
            documents.append(self._format_document(idx, res, "relevance_score"))

        return documents

    async def search_with_images(
        self, search_input: str, filters: dict = None
    ) -> Dict[str, Any]:
        # 1. Parallel Search across all professions
        tasks = [s.search(search_input, filters) for s in self.searchers]
        all_profession_results = await asyncio.gather(*tasks)

        # 2. Stratified Sampling with Safety Check
        diverse_candidates = self._get_top_per_index(all_profession_results)

        if not diverse_candidates:
            return {"documents": [], "images": []}

        # 3. Convert to LangChain Documents for the Reranker
        input_docs = []
        for d in diverse_candidates:
            content = d.get("text") or ""
            input_docs.append(Document(page_content=content, metadata=d))

        # 4. The Reranker (Sorting the diverse, validated pool)
        reranked_docs = self.reranker.compress_documents(
            documents=input_docs, query=search_input
        )

        # 5. Final Formatting
        documents, images = [], []
        for idx, doc in enumerate(reranked_docs, 1):
            res = doc.metadata
            res["text"] = doc.page_content
            documents.append(self._format_document(idx, res, "relevance_score"))
            images.extend(self._extract_images(idx, res))

        return {"documents": documents, "images": images}


# --- Summary namespace -------------

SUMMARY_NAMESPACE = "summary-namespace"
LIST_PAGE_SIZE = 100  # Pinecone caps both list() pages and fetch() id batches at 100
SUMMARY_FIELD_ALIASES = {
    "text": ["summary_text"],
    "takeaways": ["summary_key_takeaways"],
    "file_name": ["file_name", "source"],
    "hash_file": ["hash_file"],
    "category": ["category"],
    "namespace": ["namespace"],
    "link": ["link"],
}

@functools.lru_cache(maxsize=None)
def _dense_index_dimension(index_name: str, api_key: Optional[str]) -> int:
    """Dense index dimension, cached -- a probe vector must match it exactly."""
    client = get_pinecone_client(api_key=api_key)
    stats = client.Index(index_name).describe_index_stats()
    dim = getattr(stats, "dimension", None)
    if dim is None and isinstance(stats, dict):
        dim = stats.get("dimension")
    return int(dim)


def _probe_vector(dim: int) -> List[float]:
    """A random direction, drawn fresh per call, used to sample the summary
    namespace without a search text. Pinecone rejects all-zero vectors, and a
    fixed direction would pin every catalog answer to the same corner of the
    index -- Gaussian components give a uniformly random direction instead, so
    repeat asks surface different documents."""
    return [random.gauss(0.0, 1.0) for _ in range(dim)]


def summary_field(meta: Dict, field: str, default=None):
    """Read a summary-namespace record field by its logical name, trying
    each alias in SUMMARY_FIELD_ALIASES in order."""
    for key in SUMMARY_FIELD_ALIASES.get(field, [field]):
        val = (meta or {}).get(key)
        if val:
            return val
    return default


class SummaryNamespaceClient:
    """Reads the shared summary-namespace via two access patterns:
      - semantic_search: embedding-based hybrid search scoped to this namespace
      - sample: a bounded, unranked spread of documents -- for the open
        "what do you have" catalog query, where there's no natural search text
    """

    def __init__(
        self,
        profession: str,
        language: str = "Thai",
        api_key: Optional[str] = None,
        top_k: int = 8,
        name_space: Optional[Union[str, List[str]]] = None,
    ):
        dense_idx, sparse_idx = resolve_index_names(profession, language)  # A5
        self._dense_index_name = dense_idx
        self._api_key = api_key
        self._ns_filter = self._build_ns_filter(name_space)
        self._searcher = HybridRRFSearcher(
            name_space=SUMMARY_NAMESPACE,
            index_name_dense=dense_idx,
            index_name_sparse=sparse_idx,
            api_key_dense=api_key,
            api_key_sparse=api_key,
            top_k=top_k,
            rerank=False,  # small candidate pool already; nothing to gain from reranking here
        )

    @staticmethod
    def _build_ns_filter(name_space: Optional[Union[str, List[str]]]) -> Optional[Dict]:
        if not name_space:
            return None
        values = name_space if isinstance(name_space, list) else [name_space]
        return {"namespace": {"$in": values}}

    async def semantic_search(self, query: str) -> List[Dict]:
        """Raw Pinecone match dicts (id/score/text/source/page/link/metadata)
        -- callers need the raw `metadata` dict (hash_file, summary_text, ...),
        which search_raw() preserves and search() would drop."""
        fused = await self._searcher.search_raw(query, self._ns_filter)
        for doc in fused:
            doc["score"] = doc.get("rrf_score", doc.get("score", 0.0))
        return fused

    @staticmethod
    def _fetch_metadata(index, batch_ids: List[str]) -> List[Dict]:
        """Blocking fetch of one id batch -> its records' raw metadata dicts."""
        fetched = index.fetch(ids=batch_ids, namespace=SUMMARY_NAMESPACE)
        vectors = getattr(fetched, "vectors", None)
        if vectors is None and isinstance(fetched, dict):
            vectors = fetched.get("vectors", {})

        metas = []
        for rec in (vectors or {}).values():
            meta = getattr(rec, "metadata", None)
            if meta is None and isinstance(rec, dict):
                meta = rec.get("metadata")
            if meta:
                metas.append(meta)
        return metas

    def _list_ids(self, index, cap: int) -> List[str]:
        """Blocking: first `cap` record ids. list() is a lazy pager, so this
        costs the same against a 10k-doc namespace as a 100-doc one."""
        ids: List[str] = []
        for id_batch in index.list(
            namespace=SUMMARY_NAMESPACE, limit=min(cap, LIST_PAGE_SIZE)
        ):
            ids.extend(id_batch)
            if len(ids) >= cap:
                break
        return ids[:cap]

    def _query_sample(self, index, limit: int) -> List[Dict]:
        """Blocking: `limit` records matching the category scope, in one call.

        list() can't filter server-side, so scoping it means over-listing and
        discarding -- a ratio that only holds while the allowed categories are
        a large share of the corpus. Querying a random direction under a
        metadata filter asks Pinecone for `limit` *matching* records directly,
        so a rare category costs the same as a common one."""
        dim = _dense_index_dimension(self._dense_index_name, self._api_key)
        response = index.query(
            namespace=SUMMARY_NAMESPACE,
            top_k=limit,
            vector=_probe_vector(dim),
            include_metadata=True,
            filter=self._ns_filter,
        )
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        metas = []
        for match in matches or []:
            meta = getattr(match, "metadata", None)
            if meta is None and isinstance(match, dict):
                meta = match.get("metadata")
            if meta:
                metas.append(meta)
        return metas

    async def sample(self, limit: int = 30) -> List[Dict]:
        """Up to `limit` doc_summary metadata dicts -- a spread of what's in the
        namespace, for the open "what do you have" catalog case where there's no
        search text to rank by. Not ranked, not exhaustive, and not stable across
        calls; callers wanting relevance want semantic_search instead.

        Cost is bounded by `limit` alone, never by corpus size: scoped calls are
        one filtered query, unscoped calls are one list page plus one fetch."""
        try:
            client = get_pinecone_client(api_key=self._api_key)
            index = client.Index(self._dense_index_name)

            if self._ns_filter:
                return await asyncio.to_thread(self._query_sample, index, limit)

            ids = await asyncio.to_thread(self._list_ids, index, limit)
            if not ids:
                return []

            batches = [
                ids[i : i + LIST_PAGE_SIZE] for i in range(0, len(ids), LIST_PAGE_SIZE)
            ]
            fetched = await asyncio.gather(
                *(
                    asyncio.to_thread(self._fetch_metadata, index, batch)
                    for batch in batches
                )
            )
            return [meta for metas in fetched for meta in metas][:limit]
        except Exception as e:
            print(f"[SummaryNamespaceClient.sample] failed ({e}); returning [].")
            return []


class VectorDbCore:
    def __init__(
        self,
        name_space: Union[str, List[str]] = "thai-namespace",
        index_name: Union[str, List[str]] = "general-knowledge",
        top_k: int = 20,
        mode: Literal["DENSE", "SPARSE", "HYBRID", "ENSEMBLE"] = "HYBRID",
        language: str = "Thai",
        api_key_dense: Optional[str] = None,
        api_key_sparse: Optional[str] = None,
    ):
        self.name_space = name_space
        self.index_name = index_name
        self.mode = mode.upper()
        self.top_k = top_k
        self.language = language

        idx_single = index_name[0] if isinstance(index_name, list) else index_name
        ns_single = name_space[0] if isinstance(name_space, list) else name_space

        if self.mode == "DENSE":
            self.searcher = DenseSearcher(
                name_space=ns_single,
                index_name=idx_single,
                top_k=top_k,
                api_key=api_key_dense,
            )
        elif self.mode == "SPARSE":
            self.searcher = SparseSearcher(
                name_space=ns_single,
                index_name=idx_single,
                top_k=top_k,
                api_key=api_key_sparse,
            )
        elif self.mode == "HYBRID":
            dense_idx, sparse_idx = resolve_index_names(idx_single, language)
            self.searcher = HybridRRFSearcher(
                name_space=ns_single,
                index_name_dense=dense_idx,
                index_name_sparse=sparse_idx,
                api_key_dense=api_key_dense,
                api_key_sparse=api_key_sparse,
                top_k=top_k,
                top_n=12,
                rerank=True,
            )
        elif self.mode == "ENSEMBLE":
            self.searcher = EnsembleSearcher(
                index_names=index_name,
                name_spaces=name_space,
                language=language,  # A5
                top_k=top_k,
                api_key_dense=api_key_dense,
                api_key_sparse=api_key_sparse,
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _init_vector_store(self, embedding):
        """Initializes Langchain Pinecone store."""
        # Note: If this relies on Langchain, ensure PineconeVectorStore is imported
        client = get_pinecone_client()  # Need API key handling here if necessary
        index = client.Index(
            f"{self.index_name}-dense"
        )  # Assuming Langchain uses Dense
        return PineconeVectorStore(
            index=index, embedding=embedding, namespace=self.name_space
        )

    def similar_search_tools(self, search_input: str):
        """Call this tool when the user explicitly asks to know the information."""
        embedding = get_embedding_client()
        reranker = get_reranker_client()

        vectorstore = self._init_vector_store(embedding)
        k = 12
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 2}
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )

        return compression_retriever.invoke(search_input)
