from __future__ import annotations

from active_diagnosis_v13.audit_source_snapshot import source_inventory


def test_source_inventory_is_order_independent_and_ignores_caches(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.py").write_text("b\n")
    (source / "a.py").write_text("a\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    first, first_hash = source_inventory(tmp_path, [source / "b.py", source])
    second, second_hash = source_inventory(tmp_path, [source, source / "a.py"])
    assert first == second
    assert first_hash == second_hash
    assert [entry["path"] for entry in first] == ["source/a.py", "source/b.py"]


def test_source_inventory_rejects_external_source(tmp_path) -> None:
    outside = tmp_path.parent / "outside_snapshot_test.py"
    outside.write_text("outside\n")
    try:
        try:
            source_inventory(tmp_path, [outside])
        except ValueError as error:
            assert "outside repository" in str(error)
        else:
            raise AssertionError("external source should be rejected")
    finally:
        outside.unlink()
