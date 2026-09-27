#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimedicine.document_index import (
    DEFAULT_MAX_WORDS, DEFAULT_OVERLAP_WORDS, DEFAULT_TARGET_WORDS,
    build_document_chunks, reference_coverage_from_intervals,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = ROOT / "data/source"
    manifest = json.loads((source / "manifest.json").read_text())
    extracted = source / "extracted_pages.json"
    if hashlib.sha256(extracted.read_bytes()).hexdigest() != manifest["extracted_pages_sha256"]:
        raise ValueError("Extracted-page checksum mismatch")
    pdf = ROOT / manifest["source_file"]
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("PDF checksum mismatch")
    pages = json.loads(extracted.read_text())
    references = json.loads((ROOT / "data/chunks.json").read_text())
    chunks = build_document_chunks(pages, manifest, references)
    empty_pages = [page["page"] for page in pages if not page["text"].strip()]
    blank_pdf_pages = []
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    pages_with_images = [number for number, page in enumerate(reader.pages, 1) if page.images]
    for number in empty_pages:
        page = reader.pages[number - 1]
        contents = page.get_contents()
        if (contents is None or not contents.get_data()) and number not in pages_with_images:
            blank_pdf_pages.append(number)
    serialized = json.dumps(chunks, ensure_ascii=False, indent=2) + "\n"
    index_manifest = {
        "document_id": manifest["document_id"], "version": manifest["version"],
        "origin": "automatic", "source_pdf_sha256": manifest["sha256"],
        "extracted_pages_sha256": manifest["extracted_pages_sha256"],
        "document_chunks_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "parameters": {"target_words": DEFAULT_TARGET_WORDS, "max_words": DEFAULT_MAX_WORDS,
                       "overlap_words": DEFAULT_OVERLAP_WORDS},
        "page_count": len(pages), "chunk_count": len(chunks),
        "text_page_count": sum(bool(page["text"].strip()) for page in pages),
        "pages_without_extractable_text": empty_pages,
        "blank_pdf_pages": blank_pdf_pages,
        "pages_requiring_ocr": sorted(set(empty_pages) - set(blank_pdf_pages)),
        "pages_with_images": pages_with_images,
        "covered_reference_ids": reference_coverage_from_intervals(chunks),
        "offset_unit": "Unicode code points; half-open intervals within physical PDF page",
        "coverage": "All extracted characters; images have not been OCR-transcribed",
        "limitations": ["Текст внутри растровых схем отдельно не распознаётся."],
    }
    outputs = {
        ROOT / "data/document_chunks.json": serialized,
        ROOT / "data/document_index_manifest.json": json.dumps(index_manifest, ensure_ascii=False, indent=2) + "\n",
    }
    for path, text in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError(f"Index is out of date: {path.name}")
        else:
            path.write_text(text, encoding="utf-8")
    print(f"{len(chunks)} chunks; {index_manifest['text_page_count']}/{len(pages)} pages with text; "
          f"{len(index_manifest['covered_reference_ids'])} reference anchors covered")


if __name__ == "__main__":
    main()
