from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

from .document_index import covered_reference_ids, validate_source_records


def tokenize(text: str) -> list[str]:
    return [word[:7] for word in re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е"))
            if len(word) > 2]


class Retriever:
    def __init__(self, chunks):
        validate_source_records(chunks)
        self.chunks = chunks
        self.by_id = {chunk["id"]: chunk for chunk in chunks}
        self.indexed_chunks = sorted(
            (chunk for chunk in chunks if chunk.get("text", "").strip()
             and (chunk.get("origin") == "automatic" or chunk.get("source_type") != "guideline")),
            key=lambda chunk: chunk["id"],
        )
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in self.indexed_chunks]) if self.indexed_chunks else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index is None or k <= 0:
            return []
        scores = self.index.get_scores(tokenize(query))
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], self.indexed_chunks[i]["id"]))
        return [{**self.indexed_chunks[i], "score": round(float(scores[i]), 5)}
                for i in ordered[:k] if scores[i] > 0]

    def lookup(self, ids):
        missing = set(ids) - self.by_id.keys()
        if missing:
            raise ValueError("Unknown evidence: " + ", ".join(sorted(missing)))
        return [self.by_id[item] for item in dict.fromkeys(ids)]

    def covered_references(self, hits):
        references = self.lookup([item for item in covered_reference_ids(hits) if item in self.by_id])
        validate_source_records(references)
        return references

    def retrieve(self, queries, k=3):
        searches, seed_ids = [], []
        for query in queries:
            hits = self.search(query, k)
            searches.append({"query": query, "hits": [{"id": hit["id"], "score": hit["score"]} for hit in hits]})
            seed_ids.extend(hit["id"] for hit in hits)
        seed_ids = list(dict.fromkeys(seed_ids))
        ids = list(seed_ids)
        for chunk in self.lookup(seed_ids):
            if chunk.get("origin") == "automatic":
                ids.extend(item for item in (chunk.get("previous_id"), chunk.get("next_id"))
                           if item in self.by_id and self.by_id[item].get("text", "").strip())
        retrieved = self.lookup(ids)
        references = self.covered_references(retrieved)
        log = {
            "searches": searches, "search_ids": seed_ids,
            "expanded_ids": [chunk["id"] for chunk in retrieved if chunk["id"] not in seed_ids],
            "retrieved_ids": [chunk["id"] for chunk in retrieved],
            "covered_reference_ids": [chunk["id"] for chunk in references],
            "reference_map": [{"id": reference["id"], "page": reference["page"],
                               "section": reference["section"],
                               "source_chunk_ids": [chunk["id"] for chunk in retrieved
                                                    if any(span["id"] == reference["id"]
                                                           for span in chunk.get("reference_spans", []))]}
                              for reference in references],
            "index": "full-document-bm25-v2", "top_k": k, "neighbor_radius": 1,
        }
        return retrieved, references, log


def patient_queries(patient, day=0):
    queries = [
        "Нетяжелая внебольничная пневмония госпитализированные пациенты стартовая антибактериальная терапия парентерально",
        "Общий клинический анализ крови биохимический креатинин С-реактивный белок госпитализированные пациенты",
        "Мокрота микроскопия посев чувствительность до начала антибактериальной терапии интерпретация результатов",
        "Рентгенография органов грудной клетки пульсоксиметрия общий осмотр жизненные показатели",
        "Электрокардиография ЭКГ интервал QT фторхинолоны нежелательные лекарственные реакции",
        "Дозы антибактериальных препаратов нормальная функция почек печени внутривенно каждые",
        "Пациенту пневмония пить достаточное количество жидкости физические нагрузки",
        "Оценка эффективности безопасности антибактериальной терапии через 48 72 часа СРБ неэффективность",
    ]
    if patient.allergies:
        queries.append("Аллергические реакции немедленного типа бета-лактамные антибиотики альтернативные препараты")
    if patient.current_medications:
        queries.append(" ".join(patient.current_medications) + " противопоказания взаимодействия QT")
    if day:
        queries.append("Критерии клинической стабильности ступенчатая терапия переход на пероральный прием")
    if patient.culture and patient.culture.available_day <= day:
        queries.append(patient.culture.organism + " этиотропная терапия результаты микробиологических исследований чувствительность устойчивость")
    return queries
