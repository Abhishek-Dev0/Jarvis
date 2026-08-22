import json
import os

from jarvis import vendor_models


def test_status_reports_present_and_missing_sources(tmp_path, monkeypatch, capsys):
    present = tmp_path / "present_source"
    present.mkdir()
    (present / "f.bin").write_bytes(b"x" * 100)
    missing = tmp_path / "does_not_exist"

    monkeypatch.setattr(vendor_models, "sources", lambda: [
        {"name": "present", "path": str(present)},
        {"name": "missing", "path": str(missing)},
    ])
    vendor_models.status()
    out = capsys.readouterr().out
    assert "present" in out and "not found" not in out.split("present")[0]
    assert "missing" in out


def test_backup_only_copies_sources_that_exist(tmp_path, monkeypatch):
    present = tmp_path / "present_source"
    present.mkdir()
    (present / "f.bin").write_bytes(b"weights" * 10)
    missing = tmp_path / "does_not_exist"

    monkeypatch.setattr(vendor_models, "sources", lambda: [
        {"name": "present", "path": str(present)},
        {"name": "missing", "path": str(missing)},
    ])

    dest = tmp_path / "vendor_backup"
    vendor_models.backup(str(dest))

    assert (dest / "present" / "f.bin").exists()
    assert not (dest / "missing").exists()

    manifest = json.loads((dest / "manifest.json").read_text())
    names = {s["name"] for s in manifest["sources"]}
    assert names == {"present"}


def test_restore_round_trips_after_source_is_deleted(tmp_path, monkeypatch):
    src = tmp_path / "original_location"
    src.mkdir()
    (src / "weights.bin").write_bytes(b"not real weights" * 5)

    monkeypatch.setattr(vendor_models, "sources", lambda: [{"name": "fake_source", "path": str(src)}])

    dest = tmp_path / "vendor_backup"
    vendor_models.backup(str(dest))

    # simulate a fresh machine: the original location is gone
    import shutil
    shutil.rmtree(src)
    assert not src.exists()

    vendor_models.restore(str(dest))
    assert (src / "weights.bin").exists()
    assert (src / "weights.bin").read_bytes() == b"not real weights" * 5


def test_restore_raises_clearly_without_a_manifest(tmp_path):
    empty_dir = tmp_path / "no_manifest_here"
    empty_dir.mkdir()
    try:
        vendor_models.restore(str(empty_dir))
        assert False, "should have raised without a manifest.json present"
    except FileNotFoundError:
        pass
