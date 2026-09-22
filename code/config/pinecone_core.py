import functools
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeRerank

# === Lazy-loaded clients with LRU cache ===
# Deviation from cbm-system: that file hardcodes this constant. .env in this
# repo already sets EMBEDDING_MODEL (see BASELINE_PLAN.md), and the code was
# silently ignoring it -- read it here instead, falling back to the same
# value cbm-system hardcodes so behaviour is unchanged when the var is unset.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


@functools.lru_cache(maxsize=None)
def get_embedding_client():
    # print("--- LAZY INIT: Creating GoogleGenerativeAIEmbeddings Client ---")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@functools.lru_cache(maxsize=None)
def get_reranker_client():
    # print("--- LAZY INIT: Creating PineconeRerank Client ---")
    return PineconeRerank(model="bge-reranker-v2-m3", top_n=8)


@functools.lru_cache(maxsize=None)
def get_pinecone_client(api_key: any = None):
    # print("--- LAZY INIT: Creating Pinecone Client (Default) ---")
    if api_key:
        return Pinecone(api_key=api_key)
    return Pinecone()


# ==============================================================
