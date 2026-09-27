import hashlib
import json
from pathlib import Path
import re
import runpy
import unicodedata

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def normalized(text):


    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def test_quote_normalization_tolerates_pdf_spacing_but_preserves_clinical_content():
    assert normalized("Ампициллин** 2,0 г в/в каждые 6 ч") == normalized("Ампициллин**2,0г в/в\nкаждые6ч")
    source = normalized("Ампициллин** 2,0 г в/в каждые 6 ч")
    assert source != normalized("Ампициллин** 20 г в/в каждые 6 ч")
    assert source != normalized("Ампициллин** 2,0 г в/м каждые 6 ч")
    assert source != normalized("Ампициллин** 2,0 г в/в каждые 8 ч")


def test_source_integrity_and_physical_page_numbering():
    manifest = load("data/source/manifest.json")
    source = ROOT / manifest["source_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(PdfReader(source).pages) == manifest["page_count"] == 73
    extracted = ROOT / "data/source/extracted_pages.json"
    assert hashlib.sha256(extracted.read_bytes()).hexdigest() == manifest["extracted_pages_sha256"]
    pages = load("data/source/extracted_pages.json")
    assert [p["page"] for p in pages] == list(range(1, 74))


def test_every_guideline_quote_is_on_its_claimed_original_pdf_page():

    manifest = load("data/source/manifest.json")
    reader = PdfReader(ROOT / manifest["source_file"])
    chunks = [c for c in load("data/chunks.json") if c["source_type"] == "guideline"]
    page_text = {c["page"]: normalized(reader.pages[c["page"] - 1].extract_text()) for c in chunks}
    assert 20 <= len(chunks) <= 35
    for chunk in chunks:
        assert chunk["document_id"] == manifest["document_id"]
        assert chunk["version"] == manifest["version"]
        assert chunk["source_url"] == manifest["source_url"]
        assert normalized(chunk["text"]) in page_text[chunk["page"]], chunk["id"]
        assert chunk["section"] and chunk["tags"]


def test_catalog_evidence_references_exist_and_have_no_duplicate_ids():
    chunks = load("data/chunks.json")
    catalog = load("data/catalog.json")
    evidence = {c["id"]: c for c in chunks}
    assert len(evidence) == len(chunks)
    assert len({a["id"] for a in catalog}) == len(catalog)
    for action in catalog:
        assert action["evidence_ids"], action["id"]
        assert set(action["evidence_ids"]) <= set(evidence), action["id"]
        assert action["category"] in {"medication", "investigation", "monitoring", "regimen", "procedure", "review"}


def test_doses_retain_the_visually_checked_table_route_and_organ_scope():
    actions = {a["id"]: a for a in load("data/catalog.json")}


    expected = {
        "ampicillin_iv": ("ampicillin", "2,0 г", "каждые 6 ч"),
        "amoxiclav_iv": ("amoxiclav", "1,2 г", "каждые 8 ч"),
        "levofloxacin_iv": ("levofloxacin", "0,5 г", "каждые 12 ч"),
    }
    for action_id, (drug, dose, frequency) in expected.items():
        action = actions[action_id]
        assert (action["drug_id"], action["dose"], action["frequency"]) == (drug, dose, frequency)
        assert action["route"] == "в/в"
        assert {"normal_renal_function", "normal_hepatic_function"} <= set(action["requirements"])
        assert "cap-dose-scope" in action["evidence_ids"]


def test_cross_page_clinical_stability_rule_is_not_truncated():
    actions = {a["id"]: a for a in load("data/catalog.json")}
    assert {"cap-oral-switch-1", "cap-oral-switch-2"} <= set(actions["oral_switch_review"]["evidence_ids"])
    chunks = {c["id"]: c for c in load("data/chunks.json")}
    assert "двух измерениях с интервалом 8 ч" in chunks["cap-oral-switch-1"]["text"]
    assert "отсутствие нарушений всасывания" in chunks["cap-oral-switch-2"]["text"]


def test_hydration_has_no_invented_quantitative_dose():
    action = next(a for a in load("data/catalog.json") if a["id"] == "hydration")
    assert action["category"] == "regimen"
    assert action["dose"] is None
    assert action["frequency"] is None
    assert action["route"] is None
    assert action["evidence_ids"] == ["cap-hydration"]


def test_public_catalog_descriptions_do_not_reveal_hidden_future_case():


    generate_catalog = runpy.run_path(str(ROOT / "scripts/build_corpus.py"))["catalog"]
    forbidden = ("в сценарии", "haemophilus influenzae", "h. influenzae",
                 "устойчивого к ампициллину", "patient_003")
    for public_catalog in (load("data/catalog.json"), generate_catalog()):
        for action in public_catalog:
            description = action["description"].casefold()
            assert not any(token in description for token in forbidden), action["id"]


def test_retrieval_gold_targets_are_real_source_fragments():

    rows = load("data/retrieval_gold.json")
    known = {c["id"] for c in load("data/chunks.json")}
    assert 8 <= len(rows) <= 10
    assert len({row["query"] for row in rows}) == len(rows)
    for row in rows:
        assert row["query"].strip()
        assert row["expected_ids"]
        assert set(row["expected_ids"]) <= known


def test_build_corpus_handles_parser_table_linebreaks_without_rewriting_source():
    ns = runpy.run_path(str(ROOT / "scripts/build_corpus.py"))
    pages = load("data/source/extracted_pages.json")
    wrapped = "Амоксициллин+\nклавулановая кислота**\n  0,5"
    marker = re.search(r"Амоксициллин\+\s*клавулановая\s+кислота\*\*\s*0,5", pages[63]["text"])
    assert marker is not None
    pages[63]["text"] = pages[63]["text"].replace(marker.group(), wrapped, 1)
    result = ns["build_chunks"](pages)
    chunk = next(c for c in result if c["id"] == "cap-dose-amoxiclav")
    assert wrapped in chunk["text"]
    assert chunk["text"] in pages[63]["text"]
    assert normalized("1,2 г в/в каждые 6-8 ч") in normalized(chunk["text"])
