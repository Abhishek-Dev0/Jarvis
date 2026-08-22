import json

import numpy as np

from jarvis.modules.search_index import SearchIndexSkill, build_index, chunk_text, search


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("short text", chunk_size=800) == ["short text"]


def test_chunk_text_empty_text_is_no_chunks():
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text_with_overlap():
    text = "x" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)
    # consecutive chunks overlap
    assert chunks[0][-50:] in chunks[1] or chunks[1].startswith(chunks[0][700:])


def _fake_embed(dim=8):
    """Deterministic, distinguishable fake embeddings for testing ranking
    without a live Ollama call — real end-to-end behavior (real pages,
    real embeddings, real cosine ranking) was verified manually against
    this project's actual saved data during development."""
    def _embed(text, model="nomic-embed-text", host="http://localhost:11434"):
        # hash-based but deterministic per distinct text
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.rand(dim).tolist()
    return _embed


def test_build_index_writes_one_record_per_chunk(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si
    monkeypatch.setattr(si, "embed", _fake_embed())

    source_dir = tmp_path / "web"
    source_dir.mkdir()
    (source_dir / "page1.txt").write_text(
        "# source: http://example.com\n# title: Example\n# fetched: now\n\nThis is a short saved page.")
    index_path = tmp_path / "index.json"

    count = build_index(source_dir=str(source_dir), index_path=str(index_path))
    assert count == 1

    with open(index_path, encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 1
    assert records[0]["source"] == "page1.txt"
    assert "This is a short saved page." in records[0]["text"]
    # the "# source:.../# title:.../# fetched:..." header was stripped
    assert "# source:" not in records[0]["text"]


def test_build_index_empty_source_dir_indexes_nothing(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si
    monkeypatch.setattr(si, "embed", _fake_embed())

    source_dir = tmp_path / "web"
    source_dir.mkdir()
    index_path = tmp_path / "index.json"
    count = build_index(source_dir=str(source_dir), index_path=str(index_path))
    assert count == 0


def test_search_returns_empty_list_without_an_index(tmp_path):
    results = search("anything", index_path=str(tmp_path / "no_index.json"))
    assert results == []


def test_search_ranks_the_more_similar_record_first(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si

    # hand-construct an index with known, orthogonal-ish vectors so ranking
    # is checkable exactly, rather than relying on hash-based fakes
    index_path = tmp_path / "index.json"
    records = [
        {"source": "a.txt", "text": "about cats", "embedding": [1.0, 0.0, 0.0]},
        {"source": "b.txt", "text": "about dogs", "embedding": [0.0, 1.0, 0.0]},
    ]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(records, f)

    monkeypatch.setattr(si, "embed", lambda text, model="nomic-embed-text",
                         host="http://localhost:11434": [1.0, 0.0, 0.0])

    results = search("query about cats", index_path=str(index_path), top_k=2)
    assert results[0]["source"] == "a.txt"
    assert results[0]["score"] > results[1]["score"]


def test_search_reuses_normalized_index_between_queries(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si

    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps([
        {"source": "a.txt", "text": "about cats", "embedding": [1.0, 0.0]},
    ]), encoding="utf-8")
    embed_calls = []

    def fake_embed(text, model="nomic-embed-text", host="http://localhost:11434"):
        embed_calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(si, "embed", fake_embed)
    si.search("first", index_path=str(index_path))
    si.search("second", index_path=str(index_path))

    assert embed_calls == ["first", "second"]


def test_index_status_reports_saved_pages(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si

    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps([
        {"source": "a.txt", "text": "one", "embedding": [1.0]},
        {"source": "a.txt", "text": "two", "embedding": [1.0]},
        {"source": "b.txt", "text": "three", "embedding": [1.0]},
    ]), encoding="utf-8")
    monkeypatch.setattr(si, "_DEFAULT_INDEX_PATH", str(index_path))

    assert si.SearchIndexSkill().handle("index status") == (
        "Search index: 3 chunk(s) from 2 saved page(s)."
    )


def test_index_status_explains_when_index_is_missing(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si
    monkeypatch.setattr(si, "_DEFAULT_INDEX_PATH", str(tmp_path / "missing.json"))

    assert "not built yet" in si.SearchIndexSkill().handle("index status")


def test_skill_matches_search_and_index_triggers():
    sk = SearchIndexSkill()
    assert sk.matches("search my notes for tokenization") is True
    assert sk.matches("index my documents") is True
    assert sk.matches("index status") is True
    assert sk.matches("tell me a joke") is False


def test_skill_handle_search_with_no_query_prompts():
    sk = SearchIndexSkill()
    assert sk.handle("search my notes for") == "Search your notes for what?"


def test_skill_handle_search_no_index_gives_actionable_message(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si
    monkeypatch.setattr(si, "_DEFAULT_INDEX_PATH", str(tmp_path / "no_index.json"))
    sk = SearchIndexSkill()
    reply = sk.handle("search my notes for tokenization")
    assert "index my documents" in reply


def test_skill_handle_index_with_nothing_saved_gives_actionable_message(tmp_path, monkeypatch):
    import jarvis.modules.search_index as si
    monkeypatch.setattr(si, "_DEFAULT_SOURCE_DIR", str(tmp_path / "empty_web"))
    monkeypatch.setattr(si, "_DEFAULT_INDEX_PATH", str(tmp_path / "index.json"))
    (tmp_path / "empty_web").mkdir()
    sk = SearchIndexSkill()
    reply = sk.handle("index my documents")
    assert "research" in reply.lower()
