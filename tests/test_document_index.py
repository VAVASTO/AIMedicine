from copy import deepcopy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader
import pytest

from aimedicine.document_index import (
    build_document_chunks, covered_reference_ids, page_spans, reference_coverage_from_intervals,
)
from aimedicine.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return (read("data/source/extracted_pages.json"), read("data/source/manifest.json"),
            read("data/chunks.json"), read("data/document_chunks.json"))


def test_every_page_and_every_character_are_covered_verbatim(corpus):
    pages, manifest, _, chunks = corpus
    grouped = defaultdict(list)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        grouped[chunk["page"]].append(chunk)
    assert set(grouped) == set(range(1, manifest["page_count"] + 1))
    for page in pages:
        frontier = 0
        for chunk in sorted(grouped[page["page"]], key=lambda item: item["start_char"]):
            start, end = chunk["start_char"], chunk["end_char"]
            assert 0 <= start <= frontier <= end <= len(page["text"])
            assert chunk["text"] == page["text"][start:end]
            assert len(chunk["text"].split()) <= 500
            assert chunk["origin"] == "automatic"
            assert chunk["source_type"] == "guideline"
            assert chunk["document_id"] == manifest["document_id"]
            assert chunk["version"] == manifest["version"]
            assert "tags" not in chunk
            frontier = end
        assert frontier == len(page["text"])


def test_index_is_reproducible_and_pinned_to_the_source(corpus):
    pages, manifest, references, saved = corpus
    assert build_document_chunks(pages, manifest, references) == saved
    assert build_document_chunks(deepcopy(pages), deepcopy(manifest), deepcopy(references)) == saved
    provenance = read("data/document_index_manifest.json")
    assert provenance["document_chunks_sha256"] == hashlib.sha256(
        (ROOT / "data/document_chunks.json").read_bytes()).hexdigest()
    assert provenance["source_pdf_sha256"] == hashlib.sha256(
        (ROOT / manifest["source_file"]).read_bytes()).hexdigest()
    assert provenance["extracted_pages_sha256"] == hashlib.sha256(
        (ROOT / "data/source/extracted_pages.json").read_bytes()).hexdigest()


def test_all_cached_pages_match_a_fresh_extraction_of_the_original_pdf(corpus):
    pages, manifest, _, _ = corpus
    reader = PdfReader(ROOT / manifest["source_file"])
    normalize = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    for cached, original in zip(pages, reader.pages, strict=True):
        assert normalize(cached["text"]) == normalize(original.extract_text()), cached["page"]
    actual_images = [number for number, page in enumerate(reader.pages, 1) if page.images]
    assert read("data/document_index_manifest.json")["pages_with_images"] == actual_images


def test_empty_extracted_pages_are_really_blank_in_original_pdf(corpus):
    pages, manifest, _, _ = corpus
    empty = [page["page"] for page in pages if not page["text"].strip()]
    reader = PdfReader(ROOT / manifest["source_file"])
    for number in empty:
        page = reader.pages[number - 1]
        contents = page.get_contents()
        assert contents is None or not contents.get_data()
        assert not page.images
    provenance = read("data/document_index_manifest.json")
    assert provenance["blank_pdf_pages"] == empty == [72, 73]
    assert provenance["pages_requiring_ocr"] == []


def test_catalog_reference_metadata_cannot_change_chunking_or_search_text(corpus):
    pages, manifest, references, saved = corpus
    altered = deepcopy(references)
    for reference in altered:
        reference["title"] = "invented-catalog-hint"
        reference["section"] = "invented-catalog-section"
        reference["tags"] = ["invented-catalog-tags"]
    assert build_document_chunks(pages, manifest, altered) == saved
    automatic = build_document_chunks(pages, manifest)
    excluded = {"coverage_ids", "reference_spans"}
    for with_refs, without_refs in zip(saved, automatic, strict=True):
        assert {key: value for key, value in with_refs.items() if key not in excluded} == {
            key: value for key, value in without_refs.items() if key not in excluded}


def test_all_reference_anchors_are_exact_and_covered_without_catalog_injection(corpus):
    pages, _, references, chunks = corpus
    page_text = {page["page"]: page["text"] for page in pages}
    anchor_by_id = {reference["id"]: reference for reference in references}
    assert set(covered_reference_ids(chunks)) == set(anchor_by_id)
    for chunk in chunks:
        direct = set()
        for span in chunk["reference_spans"]:
            anchor = anchor_by_id[span["id"]]
            assert anchor["page"] == chunk["page"]
            assert page_text[chunk["page"]][span["start_char"]:span["end_char"]] == anchor["text"]
            if chunk["start_char"] <= span["start_char"] and span["end_char"] <= chunk["end_char"]:
                direct.add(span["id"])
                assert anchor["text"] in chunk["text"]
        assert set(chunk["coverage_ids"]) == direct
        assert set(covered_reference_ids([chunk])) == direct


def test_partial_anchor_requires_all_text_and_does_not_bridge_a_gap_or_page():
    first = {"document_id": "doc", "version": "1", "page": 1, "origin": "automatic",
             "start_char": 0, "end_char": 10,
             "reference_spans": [{"id": "reference", "start_char": 5, "end_char": 20}]}
    second = {**first, "start_char": 10, "end_char": 25}
    assert reference_coverage_from_intervals([first]) == []
    assert reference_coverage_from_intervals([second]) == []
    assert reference_coverage_from_intervals([first, second]) == ["reference"]
    for changed in ({"start_char": 11}, {"page": 2}, {"document_id": "another"}, {"version": "2"}):
        assert reference_coverage_from_intervals([first, {**second, **changed}]) == []
    assert reference_coverage_from_intervals([first, second, first]) == ["reference"]


def test_unselected_prevention_page_is_searchable(corpus):
    _, _, references, chunks = corpus
    query = "пневмококковых инфекций иммунодефицитами кохлеарными имплантами вакцинации"
    assert 37 not in {reference["page"] for reference in references}
    hits = Retriever(chunks).search(query, k=5)
    assert any(hit["page"] == 37 for hit in hits)
    assert any("кохлеарными" in hit["text"] for hit in hits)
    assert all(hit["origin"] == "automatic" for hit in hits)


@pytest.mark.parametrize("text", ["", " \n\t ", "одно слово", "\n".join(
    "Абзац " + " ".join(f"слово{word}" for word in range(87)) + "." for _ in range(12))])
def test_split_boundaries_preserve_whitespace_unicode_and_overlap(text):
    spans = page_spans(text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (start, end), (next_start, _) in zip(spans, spans[1:]):
        assert start < next_start < end
        assert len(text[next_start:end].split()) == 60
    assert all(len(text[start:end].split()) <= 500 for start, end in spans)


def test_missing_or_reordered_pages_and_invalid_size_are_rejected(corpus):
    pages, manifest, _, _ = corpus
    with pytest.raises(ValueError, match="Pages must be"):
        build_document_chunks(pages[:-1], manifest)
    with pytest.raises(ValueError, match="Pages must be"):
        build_document_chunks(list(reversed(pages)), manifest)
    with pytest.raises(ValueError, match="overlap_words"):
        page_spans("Text", target_words=50, overlap_words=50)


def test_neighbor_links_follow_document_order(corpus):
    chunks = corpus[-1]
    assert chunks[0]["previous_id"] is None
    assert chunks[-1]["next_id"] is None
    for first, second in zip(chunks, chunks[1:]):
        assert first["next_id"] == second["id"]
        assert second["previous_id"] == first["id"]
