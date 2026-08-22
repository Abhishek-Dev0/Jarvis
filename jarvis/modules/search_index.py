"""
search_index.py — vector search over locally-saved documents (currently
web_growth's saved pages, data/web/*.txt). From the 2026-08-22 systems
audit (P3): retrieval by keyword already existed (web.py's WebSearchSkill,
against the live web); nothing retrieved by *meaning* over content JARVIS
had already saved locally.

Model measured, not guessed, same discipline as every other model choice
in this project: nomic-embed-text via Ollama, ~2-3.6s per embedding once
warm on this machine (768-dim vectors).

Index is a flat JSON file (embedding as a list of floats + source text +
metadata), with numpy cosine-similarity search at query time — not a real
vector database (chromadb/faiss/etc.), on purpose. The corpus this indexes
is currently a handful of saved web pages, not millions of documents; a
flat file plus numpy is the right scope until there's enough content to
need more. See the audit's own anti-overengineering framing.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SOURCE_DIR = os.path.join(_PKG_DIR, "data", "web")
_DEFAULT_INDEX_PATH = os.path.join(_PKG_DIR, "data", "search_index.json")


def embed(text: str, model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> list[float]:
    import requests
    r = requests.post(f"{host.rstrip('/')}/api/embeddings",
                       json={"model": model, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple sliding-window chunking on characters, not tokens — good
    enough at this corpus size; a smarter (sentence-aware) splitter is a
    reasonable future upgrade if chunk boundaries start cutting content
    awkwardly in practice, not something to guess-optimize now."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def build_index(source_dir: str | None = None, index_path: str | None = None,
                 model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> int:
    """Rebuilds the index from every .txt file in source_dir. Returns the
    number of chunks indexed. Skips WebGrowthSkill's manifest.json and the
    "# source:.../# title:.../# fetched:..." header lines it prepends to
    each saved page, so those don't pollute the embedded text."""
    source_dir = source_dir or _DEFAULT_SOURCE_DIR
    index_path = index_path or _DEFAULT_INDEX_PATH
    records = []

    for path in sorted(glob.glob(os.path.join(source_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        body = "".join(lines[4:]) if len(lines) > 4 and lines[0].startswith("# source:") else "".join(lines)
        for chunk in chunk_text(body):
            try:
                vector = embed(chunk, model=model, host=host)
            except Exception as e:
                print(f"[search_index] couldn't embed a chunk from {os.path.basename(path)}: {e}")
                continue
            records.append({"source": os.path.basename(path), "text": chunk, "embedding": vector})

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return len(records)


def search(query: str, index_path: str | None = None, top_k: int = 5,
           model: str = "nomic-embed-text", host: str = "http://localhost:11434") -> list[dict]:
    index_path = index_path or _DEFAULT_INDEX_PATH
    if not os.path.exists(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return []

    query_vec = np.array(embed(query, model=model, host=host))
    matrix = np.array([r["embedding"] for r in records])
    # cosine similarity, vectorized: normalize once, then a single dot product
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    scores = matrix_norm @ query_norm

    ranked = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"source": records[i]["source"], "text": records[i]["text"], "score": float(scores[i])}
            for i in ranked]


_SEARCH_TRIGGERS = ("search my notes for", "search my documents for", "search saved pages for")
_INDEX_TRIGGERS = {"index my documents", "rebuild search index", "index saved pages"}


class SearchIndexSkill(SkillModule):
    """"search my notes for X" (vector search over data/web/) / "rebuild
    search index" — ungated, read-only, operates only on JARVIS's own
    already-saved local files."""

    name = "search_index"
    description = "vector search over locally-saved web pages (data/web/), by meaning not keyword"
    priority = 8

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    @property
    def available(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.host}/api/version", timeout=2).ok
        except Exception:
            return False

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        if t in _INDEX_TRIGGERS:
            return True
        return any(t.startswith(p) for p in _SEARCH_TRIGGERS)

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()

        if low in _INDEX_TRIGGERS:
            try:
                count = build_index(model=self.model, host=self.host)
            except Exception as e:
                return f"Couldn't build the search index: {e}"
            if count == 0:
                return "Nothing to index yet — data/web/ has no saved pages. Try \"research <topic>\" first."
            return f"Indexed {count} chunk(s) from saved pages."

        for prefix in _SEARCH_TRIGGERS:
            if low.startswith(prefix):
                query = t[len(prefix):].strip()
                if not query:
                    return "Search your notes for what?"
                try:
                    results = search(query, model=self.model, host=self.host)
                except Exception as e:
                    return f"Search failed: {e}"
                if not results:
                    return ("No search index yet (or nothing matched). "
                            "Say \"index my documents\" to build one from your saved pages.")
                lines = [f"Top matches for '{query}':"]
                for r in results:
                    snippet = r["text"][:200].replace("\n", " ")
                    lines.append(f"\n[{r['score']:.2f}] {r['source']}\n  {snippet}...")
                return "\n".join(lines)

        return "I didn't catch what to search for."
