from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from aimedicine import data
from aimedicine.document_index import covered_reference_ids, verified_source_records
from aimedicine.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]
PROOF_FILES = (
    "data/source/manifest.json", "data/source/extracted_pages.json",
    "data/document_index_manifest.json", "data/document_chunks.json", "data/chunks.json",
)


@pytest.fixture
def copied_source(tmp_path):
    for relative in PROOF_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_removed_dose_text_cannot_enter_the_search_index():
    records = data.load_chunks()
    anchor = next(record["text"] for record in records if record["id"] == "cap-dose-ampicillin")
    mutated = []
    for record in records:
        if record.get("origin") == "automatic" and anchor in record["text"]:
            record["text"] = record["text"].replace(anchor, "[фрагмент отсутствует]")
            mutated.append(record)
    assert mutated
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        Retriever(records)
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        covered_reference_ids(mutated)


def test_changed_source_cannot_bypass_validation_by_removing_its_origin():
    records = data.load_chunks()
    chunk = next(record for record in records if record.get("origin") == "automatic"
                 and "cap-dose-ampicillin" in record["coverage_ids"])
    chunk["text"] = "changed"
    del chunk["origin"]
    chunk["source_type"] = "demo_policy"
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        Retriever(records)
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        covered_reference_ids([chunk])


@pytest.mark.parametrize("field", ["text", "end_char", "reference_spans"])
def test_mutation_after_indexing_cannot_prove_a_reference(field):
    retriever = Retriever(data.load_chunks())
    chunk = next(record for record in retriever.chunks if record.get("origin") == "automatic"
                 and "cap-dose-ampicillin" in record["coverage_ids"])
    if field == "text":
        chunk[field] = chunk[field].replace("2,0", "9,9", 1)
    elif field == "end_char":
        chunk[field] += 1
    else:
        chunk[field] = [{"id": "cap-dose-ampicillin", "start_char": 0, "end_char": 1}]
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        retriever.covered_references([chunk])


@pytest.mark.parametrize("remove_origin", [False, True])
def test_mutated_reference_alias_is_rejected_after_indexing(remove_origin):
    retriever = Retriever(data.load_chunks())
    alias = retriever.by_id["cap-dose-ampicillin"]
    alias["text"] = retriever.by_id["cap-dose-levofloxacin"]["text"]
    if remove_origin:
        del alias["origin"]
    delivered = [chunk for chunk in retriever.chunks if chunk.get("origin") == "automatic"
                 and "cap-dose-ampicillin" in chunk["coverage_ids"]]
    with pytest.raises(ValueError, match="Source integrity mismatch"):
        retriever.covered_references(delivered)


@pytest.mark.parametrize("relative,error", [
    ("data/document_chunks.json", "Document index integrity"),
    ("data/source/extracted_pages.json", "Extracted-page integrity"),
])
def test_load_rejects_corrupted_source_files(copied_source, monkeypatch, relative, error):
    monkeypatch.setattr(data, "ROOT", copied_source)
    assert data.load_chunks()
    path = copied_source / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=error):
        data.load_chunks()


def test_rehashed_index_still_has_to_match_original_page_intervals(copied_source, monkeypatch):
    monkeypatch.setattr(data, "ROOT", copied_source)
    index_path = copied_source / "data/document_chunks.json"
    records = json.loads(index_path.read_bytes())
    target = next(record for record in records if "cap-dose-ampicillin" in record["coverage_ids"])
    target["text"] = target["text"].replace("2,0", "9,9", 1)
    index_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    manifest_path = copied_source / "data/document_index_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["document_chunks_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match source text"):
        data.load_chunks()


def test_reference_id_cannot_be_rebound_to_another_verbatim_excerpt(copied_source, monkeypatch):
    monkeypatch.setattr(data, "ROOT", copied_source)
    path = copied_source / "data/chunks.json"
    references = json.loads(path.read_bytes())
    by_id = {reference["id"]: reference for reference in references}
    by_id["cap-dose-ampicillin"]["text"] = by_id["cap-dose-levofloxacin"]["text"]
    path.write_text(json.dumps(references, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match source text"):
        data.load_chunks()


def test_callers_cannot_mutate_the_cached_source_snapshot(copied_source):
    first, references = verified_source_records(copied_source)
    saved = deepcopy(first)
    saved_references = deepcopy(references)
    first[0]["text"] = "changed"
    references[0]["text"] = "changed"
    next_index, next_references = verified_source_records(copied_source)
    assert next_index == saved
    assert next_references == saved_references


def test_preserving_size_and_mtime_does_not_bypass_cache_invalidation(copied_source):
    verified_source_records(copied_source)
    path = copied_source / "data/document_chunks.json"
    before = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"origin": "automatic"', b'"origin": "reference"', 1))
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="Document index integrity"):
        verified_source_records(copied_source)
