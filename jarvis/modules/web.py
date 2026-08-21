"""
web.py — search the live web and grow the corpus from it.

Two separate concerns, deliberately kept apart:

  WebSearchSkill   answers a question right now, by retrieval, not generation.
                    Same philosophy as CalculatorSkill: a 25M-param model
                    cannot know today's news or verify a fact. Don't ask it
                    to. Fetch real results and hand them back verbatim.

  WebGrowthSkill   saves full page text to data/web/ for later training.
                    Searching and learning are different actions with
                    different costs (a growth fetch pulls full pages, not
                    just snippets) so they need different trigger phrases.

Both use DuckDuckGo's HTML endpoint (html.duckduckgo.com) — no API key, no
account, no rate-limit contract to break. That also means it's not a stable
API: if DuckDuckGo changes their markup, _parse_results breaks. It's wrapped
in try/except so a broken parser degrades to "search failed" rather than
crashing the assistant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base import SkillModule

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_TIMEOUT = 10


def search(query: str, max_results: int = 5) -> list[dict]:
    """Query DuckDuckGo's no-JS HTML endpoint. Returns [{title, url, snippet}]."""
    resp = requests.post(
        _SEARCH_URL, data={"q": query}, headers=_HEADERS, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for row in soup.select("div.result"):
        if "result--ad" in row.get("class", []):
            continue
        link = row.select_one("a.result__a")
        snippet = row.select_one("a.result__snippet, div.result__snippet")
        if link is None or not link.get("href"):
            continue
        url = link["href"]
        if urlparse(url).netloc.endswith("duckduckgo.com"):
            continue   # ad redirect (y.js) or other DDG-internal link, not an organic result
        results.append({
            "title": link.get_text(strip=True),
            "url": url,
            "snippet": snippet.get_text(strip=True) if snippet else "",
        })
        if len(results) >= max_results:
            break
    return results


def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a page and return its visible text, stripped of markup/scripts."""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars]


# --------------------------------------------------------------------- search

class WebSearchSkill(SkillModule):
    """Answers 'search for X' / 'look up X' style requests by retrieval."""

    name = "web_search"
    description = "searches the web for current information"
    priority = 8   # below calculator (10), so arithmetic still wins ties

    _TRIGGERS = (
        "search for", "search the web for", "look up", "look for",
        "google", "what's the latest on", "what is the latest on",
        "find information about", "who is", "who was",
    )

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) or f" {p} " in f" {t} " for p in self._TRIGGERS)

    def _strip_trigger(self, text: str) -> str:
        t = text.strip().rstrip("?")
        low = t.lower()
        for p in sorted(self._TRIGGERS, key=len, reverse=True):
            if low.startswith(p):
                return t[len(p):].strip()
        return t

    def handle(self, text: str) -> str:
        query = self._strip_trigger(text)
        if not query:
            return "Search for what?"
        try:
            results = search(query, max_results=5)
        except requests.RequestException as e:
            return f"Search failed — network error ({e})."
        except Exception as e:
            return f"Search failed — couldn't parse results ({e})."

        if not results:
            return f"No results for '{query}'."

        lines = [f"Top results for '{query}':"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r['title']}\n   {r['snippet']}\n   {r['url']}")
        return "\n".join(lines)


# --------------------------------------------------------------------- growth

class WebGrowthSkill(SkillModule):
    """Answers 'research X' / 'learn about X' by fetching full pages and
    saving them under data/web/ for a future retrain — separate from
    data/corpus.txt so a bad fetch never contaminates the curated corpus
    without a deliberate merge step."""

    name = "web_growth"
    description = "researches a topic and saves sources for future training"
    priority = 8

    _TRIGGERS = ("research", "learn about", "read up on", "study")

    def __init__(self, data_dir: str = "data/web", pages_per_topic: int = 3):
        self.data_dir = data_dir
        self.pages_per_topic = pages_per_topic
        self._manifest_path = os.path.join(data_dir, "manifest.json")
        self._manifest = {}

    def setup(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, encoding="utf-8") as f:
                self._manifest = json.load(f)

    def _save_manifest(self):
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) for p in self._TRIGGERS)

    def _strip_trigger(self, text: str) -> str:
        t = text.strip().rstrip("?")
        low = t.lower()
        for p in sorted(self._TRIGGERS, key=len, reverse=True):
            if low.startswith(p):
                return t[len(p):].strip()
        return t

    def handle(self, text: str) -> str:
        topic = self._strip_trigger(text)
        if not topic:
            return "Research what?"
        try:
            results = search(topic, max_results=self.pages_per_topic)
        except requests.RequestException as e:
            return f"Research failed — network error ({e})."

        if not results:
            return f"No sources found for '{topic}'."

        saved, skipped = [], 0
        for r in results:
            url_hash = hashlib.sha256(r["url"].encode()).hexdigest()[:16]
            if url_hash in self._manifest:
                skipped += 1
                continue
            try:
                body = fetch_page_text(r["url"])
            except Exception:
                continue
            if len(body) < 200:      # too thin to be worth keeping
                continue

            fname = f"{url_hash}.txt"
            with open(os.path.join(self.data_dir, fname), "w", encoding="utf-8") as f:
                f.write(f"# source: {r['url']}\n# title: {r['title']}\n"
                        f"# fetched: {datetime.now(timezone.utc).isoformat()}\n\n{body}")

            self._manifest[url_hash] = {
                "url": r["url"], "title": r["title"],
                "fetched": datetime.now(timezone.utc).isoformat(),
                "chars": len(body), "file": fname, "topic": topic,
            }
            saved.append(r["title"])
            time.sleep(0.5)   # don't hammer whatever site we're pulling from

        self._save_manifest()

        if not saved:
            note = " (already had all of them)" if skipped else ""
            return f"Found sources on '{topic}' but nothing new to save{note}."
        summary = "\n".join(f"  - {t}" for t in saved)
        return (f"Saved {len(saved)} source(s) on '{topic}' to {self.data_dir}/"
                f"{f' ({skipped} already had)' if skipped else ''}:\n{summary}\n\n"
                f"Run core.data absorb-web to fold these into the training corpus "
                f"when you're ready to retrain.")
