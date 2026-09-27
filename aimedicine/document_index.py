from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


DEFAULT_TARGET_WORDS = 350
DEFAULT_MAX_WORDS = 500
DEFAULT_OVERLAP_WORDS = 60
MIN_WORDS = 250
ROOT = Path(__file__).resolve().parents[1]
_PROOF_FILES = (
    "data/source/manifest.json", "data/source/extracted_pages.json",
    "data/document_index_manifest.json", "data/document_chunks.json", "data/chunks.json",
)

_HEADING = re.compile(
    r"^(?:[1-9](?:\.\d+){0,4}\.?\s+[А-ЯЁ][^\n]{3,175}"
    r"|Приложение\s+[А-ЯA-Z]\d*[^\n]{0,150})$"
)


def page_spans(text: str, *, target_words=DEFAULT_TARGET_WORDS,
               max_words=DEFAULT_MAX_WORDS,
               overlap_words=DEFAULT_OVERLAP_WORDS) -> list[tuple[int, int]]:
    if not 0 <= overlap_words < target_words <= max_words:
        raise ValueError("Require 0 <= overlap_words < target_words <= max_words")
    words = list(re.finditer(r"\S+", text))
    if not words:
        return [(0, len(text))]
    starts = [word.start() for word in words]
    boundaries = set()
    paragraphs = set()
    for match in re.finditer(r"\n+", text):
        index = bisect_right(starts, match.end() - 1)
        boundaries.add(index)
        if len(match.group()) > 1 or text[:match.start()].rstrip().endswith((".", ";", ":")):
            paragraphs.add(index)
    spans = []
    left = 0
    while left < len(words):
        remaining = len(words) - left
        if remaining <= max_words:
            right = len(words)
        else:
            minimum = min(MIN_WORDS, target_words)
            desired = left + min(target_words, remaining - minimum + overlap_words)
            lower = left + minimum
            upper = min(left + max_words, len(words) - minimum + overlap_words)
            candidates = [x for x in boundaries if lower <= x <= upper]
            preferred = [x for x in candidates if x in paragraphs and abs(x - desired) <= 50]
            right = min(preferred or candidates or [desired], key=lambda x: (abs(x - desired), x))
        start = 0 if left == 0 else words[left].start()
        end = len(text) if right == len(words) else words[right].start()
        spans.append((start, end))
        if right == len(words):
            break
        left = right - overlap_words
    return spans


def _reference_spans(pages: list[dict], references: list[dict]) -> dict[int, list[dict]]:
    text_by_page = {page["page"]: page["text"] for page in pages}
    spans = defaultdict(list)
    for reference in references:
        text = reference["text"]
        raw = text_by_page[reference["page"]]
        start = raw.find(text)
        if not text or start < 0:
            raise ValueError(f"Reference is not verbatim on its page: {reference['id']}")
        if raw.find(text, start + 1) >= 0:
            raise ValueError(f"Ambiguous reference on its page: {reference['id']}")
        spans[reference["page"]].append({
            "id": reference["id"], "start_char": start, "end_char": start + len(text),
        })
    return spans


def build_document_chunks(pages: list[dict], manifest: dict, references=(), *,
                          target_words=DEFAULT_TARGET_WORDS,
                          max_words=DEFAULT_MAX_WORDS,
                          overlap_words=DEFAULT_OVERLAP_WORDS) -> list[dict]:
    numbers = [page["page"] for page in pages]
    if numbers != list(range(1, manifest["page_count"] + 1)):
        raise ValueError("Pages must be consecutive, complete, and one-based")
    anchors = _reference_spans(pages, list(references))
    chunks = []
    section = ""
    for page in pages:
        text = page["text"]
        headings = []
        for match in re.finditer(r"[^\n]+", text):
            line = match.group().strip()
            if _HEADING.fullmatch(line):
                headings.append((match.start(), line))
        prior_section = section
        for start, end in page_spans(text, target_words=target_words,
                                     max_words=max_words, overlap_words=overlap_words):
            before = [heading for offset, heading in headings if offset <= start]
            current = before[-1] if before else prior_section
            content = text[start:end]
            digest = hashlib.sha256(content.encode()).hexdigest()[:10]
            reference_spans = [dict(anchor) for anchor in anchors[page["page"]]
                               if anchor["start_char"] < end and anchor["end_char"] > start]
            coverage = [anchor["id"] for anchor in reference_spans
                        if start <= anchor["start_char"] and anchor["end_char"] <= end]
            chunks.append({
                "id": f"{manifest['document_id']}:p{page['page']:03d}:{start}-{end}:{digest}",
                "document_id": manifest["document_id"],
                "version": manifest["version"],
                "source_type": "guideline",
                "origin": "automatic",
                "page": page["page"],
                "start_char": start,
                "end_char": end,
                "text": content,
                "title": current or manifest["title"],
                "section": current,
                "source_url": manifest["source_url"],
                "text_status": "extracted" if content.strip() else "no_extractable_text",
                "coverage_ids": coverage,
                "reference_spans": reference_spans,
            })
        if headings:
            section = headings[-1][1]
    for i, chunk in enumerate(chunks):
        chunk["previous_id"] = chunks[i - 1]["id"] if i else None
        chunk["next_id"] = chunks[i + 1]["id"] if i + 1 < len(chunks) else None
    return chunks


def reference_coverage_from_intervals(chunks: list[dict]) -> list[str]:
    intervals = defaultdict(list)
    references = {}
    for chunk in chunks:
        if chunk.get("origin") != "automatic":
            continue
        key = (chunk["document_id"], chunk["version"], chunk["page"])
        intervals[key].append((chunk["start_char"], chunk["end_char"]))
        for reference in chunk.get("reference_spans", []):
            references[(key, reference["id"])] = reference
    merged = {}
    for key, spans in intervals.items():
        result = []
        for start, end in sorted(spans):
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], end))
            else:
                result.append((start, end))
        merged[key] = result
    return sorted({reference_id for (key, reference_id), reference in references.items()
                   if any(start <= reference["start_char"] and reference["end_char"] <= end
                          for start, end in merged[key])})


@lru_cache(maxsize=4)
def _verified_snapshot(root: str, signatures: tuple) -> tuple[bytes, bytes]:
    raw = {name: (Path(root) / name).read_bytes() for name in _PROOF_FILES}
    source = json.loads(raw["data/source/manifest.json"])
    manifest = json.loads(raw["data/document_index_manifest.json"])
    page_digest = hashlib.sha256(raw["data/source/extracted_pages.json"]).hexdigest()
    if page_digest != source["extracted_pages_sha256"] or page_digest != manifest["extracted_pages_sha256"]:
        raise ValueError("Extracted-page integrity check failed")
    if hashlib.sha256(raw["data/document_chunks.json"]).hexdigest() != manifest["document_chunks_sha256"]:
        raise ValueError("Document index integrity check failed")
    if any(manifest[key] != source[key] for key in ("document_id", "version")):
        raise ValueError("Document identity mismatch")
    if manifest["source_pdf_sha256"] != source["sha256"]:
        raise ValueError("Document PDF identity mismatch")
    pages = json.loads(raw["data/source/extracted_pages.json"])
    references = json.loads(raw["data/chunks.json"])
    for reference in references:
        if (reference["document_id"] != source["document_id"]
                or reference["version"] != source["version"]
                or reference["source_type"] != "guideline"):
            raise ValueError(f"Reference identity mismatch: {reference['id']}")
    expected = build_document_chunks(pages, source, references, **manifest["parameters"])
    if json.loads(raw["data/document_chunks.json"]) != expected:
        raise ValueError("Document chunks do not match source text, offsets, or reference anchors")
    return raw["data/document_chunks.json"], raw["data/chunks.json"]


def verified_source_records(root: Path = ROOT) -> tuple[list[dict], list[dict]]:
    root = Path(root).resolve()
    signatures = []
    for name in _PROOF_FILES:
        stat = (root / name).stat()
        signatures.append((stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    index, references = _verified_snapshot(str(root), tuple(signatures))
    return json.loads(index), [{**item, "origin": "reference"} for item in json.loads(references)]


def validate_source_records(chunks: list[dict], root: Path = ROOT) -> None:
    if not chunks:
        return
    index, references = verified_source_records(root)
    canonical = {chunk["id"]: chunk for chunk in index + references}
    supplied = [chunk for chunk in chunks if chunk.get("origin") in {"automatic", "reference"}
                or chunk.get("id") in canonical]
    source_fields = ("id", "document_id", "version", "source_type", "origin", "page", "text")
    interval_fields = ("start_char", "end_char", "reference_spans", "coverage_ids", "previous_id", "next_id")
    for chunk in supplied:
        expected = canonical.get(chunk.get("id"))
        if expected is None:
            raise ValueError(f"Unknown source chunk: {chunk.get('id')}")
        fields = source_fields + (interval_fields if expected["origin"] == "automatic" else ())
        if any(chunk.get(field) != expected.get(field) for field in fields):
            raise ValueError(f"Source integrity mismatch: {chunk['id']}")


def covered_reference_ids(chunks: list[dict]) -> list[str]:
    validate_source_records(chunks)
    automatic = [chunk for chunk in chunks if chunk.get("origin") == "automatic"]
    return reference_coverage_from_intervals(automatic)
