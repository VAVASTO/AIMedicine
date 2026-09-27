from pathlib import Path
import json
from .document_index import verified_source_records

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_chunks():
    document_chunks, references = verified_source_records(ROOT)
    items = document_chunks + references
    for name in ("supplemental_chunks.json", "policies.json"):
        p = ROOT / "data" / name
        if p.exists():
            records = read_json(p)
            items.extend(records)
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence identifiers")
    return items


def load_catalog():
    return read_json(ROOT / "data/catalog.json")


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(p)
