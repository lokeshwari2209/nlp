"""Flask application for the agriculture debate intelligence dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

from nlp_core import (
    DEFAULT_DATASET,
    build_analytics,
    display_title,
    extract_entities,
    load_records,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("DATASET_PATH", DEFAULT_DATASET)).resolve()
ARTIFACT_DIR = Path(
    os.environ.get("ARTIFACT_DIR", BASE_DIR / "artifacts")
).resolve()
MODEL_NAME = "all-MiniLM-L6-v2"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


class ArtifactUnavailable(RuntimeError):
    """Raised when the model/index has not been built for the current dataset."""


@lru_cache(maxsize=1)
def get_records():
    return load_records(DATASET_PATH)


@lru_cache(maxsize=1)
def get_analytics() -> dict[str, Any]:
    return build_analytics(get_records())


def _dataset_digest() -> str:
    digest = hashlib.sha256()
    with DATASET_PATH.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_artifact_lock = threading.Lock()
_artifacts: dict[str, Any] | None = None


def _read_artifacts() -> dict[str, Any]:
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    with _artifact_lock:
        if _artifacts is not None:
            return _artifacts
        manifest_path = ARTIFACT_DIR / "manifest.json"
        required = [
            manifest_path,
            ARTIFACT_DIR / "embeddings.npy",
            ARTIFACT_DIR / "search_records.json",
            ARTIFACT_DIR / "topics.json",
            ARTIFACT_DIR / "sentence_transformer",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ArtifactUnavailable(
                "Search and BERTopic artifacts are not initialized. "
                "Run `python init_artifacts.py` first."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset_sha256") != _dataset_digest():
                raise ArtifactUnavailable(
                    "NLP artifacts were built from a different Excel file. "
                    "Run `python init_artifacts.py` again."
                )
            if manifest.get("model_name") != MODEL_NAME:
                raise ArtifactUnavailable(
                    "The saved embedding model does not match this application."
                )
            embeddings = np.load(ARTIFACT_DIR / "embeddings.npy", mmap_mode="r")
            search_records = json.loads(
                (ARTIFACT_DIR / "search_records.json").read_text(encoding="utf-8")
            )
            topics = json.loads(
                (ARTIFACT_DIR / "topics.json").read_text(encoding="utf-8")
            )
            if embeddings.ndim != 2 or len(embeddings) != len(search_records):
                raise ValueError("Embedding rows do not match search records.")
            if len(topics.get("topics_by_record", [])) != len(get_records()):
                raise ValueError("Topic assignments do not match cleaned records.")

            index = None
            index_path = ARTIFACT_DIR / "faiss.index"
            if index_path.is_file():
                try:
                    import faiss

                    index = faiss.read_index(str(index_path))
                    if index.ntotal != len(search_records):
                        raise ValueError("FAISS index size does not match records.")
                except ImportError:
                    logger.warning(
                        "FAISS is unavailable; using normalized NumPy "
                        "inner-product search."
                    )

            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                str(ARTIFACT_DIR / "sentence_transformer")
            )
            _artifacts = {
                "manifest": manifest,
                "embeddings": embeddings,
                "search_records": search_records,
                "topics": topics,
                "index": index,
                "model": model,
            }
            return _artifacts
        except ArtifactUnavailable:
            raise
        except Exception as exc:
            logger.exception("Could not load generated NLP artifacts.")
            raise ArtifactUnavailable(
                f"Unable to load NLP artifacts: {exc}. "
                "Rebuild them with `python init_artifacts.py`."
            ) from exc


def _serialize_result(record: dict[str, Any], score: float) -> dict[str, Any]:
    row_index = int(record["record_id"])
    row = get_records().iloc[row_index]
    topic_by_record = _artifacts["topics"]["topics_by_record"][row_index]
    return {
        "title": display_title(row),
        "house": str(row["House"]),
        "year": int(row["unified_year"])
        if row["unified_year"] == row["unified_year"]
        else None,
        "date": row["unified_date"].date().isoformat()
        if row["unified_date"] == row["unified_date"]
        else None,
        "subject": _value_text(row.get("debateTitleSubject", "")),
        "ministry": _value_text(row.get("ministry", "")),
        "member": _value_text(
            row.get("memberName", row.get("questionerName", ""))
        ),
        "snippet": str(record["chunk_text"]),
        "similarity_score": round(float(score), 4),
        "entities": extract_entities(row["text"]),
        "topic": topic_by_record.get("label", "Unknown"),
    }


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(np.asarray(value != value).item()):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _answer_from_evidence(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        return {
            "answer": (
                "I could not find a matching record in the available dataset. "
                "Try a broader agriculture topic, policy name, crop, or year."
            ),
            "evidence_scope": "No matching records",
            "evidence_count": 0,
            "themes": [],
            "agricultural_terms": [],
        }

    topic_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    houses: dict[str, int] = {}
    years: list[int] = []
    for item in evidence:
        topic = item["topic"]
        if topic not in {"Other / Outlier", "Unknown", ""}:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        houses[item["house"]] = houses.get(item["house"], 0) + 1
        if item["year"] is not None:
            years.append(item["year"])
        for terms in item["entities"].values():
            for term in terms:
                entity_counts[term] = entity_counts.get(term, 0) + 1

    themes = sorted(topic_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:2]
    agricultural_terms = sorted(
        entity_counts.items(), key=lambda pair: (-pair[1], pair[0])
    )[:5]
    house_summary = ", ".join(
        f"{house} ({count})"
        for house, count in sorted(houses.items(), key=lambda pair: (-pair[1], pair[0]))
    )
    record_word = "record" if len(evidence) == 1 else "records"
    sentences = [
        (
            f"I reviewed the top {len(evidence)} distinct {record_word} retrieved "
            f"for “{query}”."
        )
    ]
    if themes:
        sentences.append(
            "The most common BERTopic themes in this retrieved set are "
            + "; ".join(
                f"{label} ({count} {'record' if count == 1 else 'records'})"
                for label, count in themes
            )
            + "."
        )
    if agricultural_terms:
        sentences.append(
            "The agriculture keywords explicitly appearing in these records include "
            + ", ".join(f"{term} ({count})" for term, count in agricultural_terms)
            + "."
        )
    if years:
        sentences.append(
            f"These retrieved records span {min(years)}–{max(years)} and include "
            f"{house_summary}."
        )
    else:
        sentences.append(f"The retrieved records include {house_summary}.")
    title_examples = "; ".join(
        f"[{position}] “{item['title']}”"
        for position, item in enumerate(evidence[:3], start=1)
    )
    sentences.append(
        f"Relevant record titles in this set include {title_examples}."
    )
    sentences.append(
        "The evidence below comes from these records' titles, subjects, keywords, and "
        "available metadata. The workbook has no full debate transcripts, so this "
        "is a summary of matching records—not a transcript-based or exhaustive "
        "count of every debate on the subject."
    )
    return {
        "answer": " ".join(sentences),
        "evidence_scope": f"Top {len(evidence)} distinct retrieved records",
        "evidence_count": len(evidence),
        "themes": [{"label": label, "records": count} for label, count in themes],
        "agricultural_terms": [
            {"term": term, "records": count}
            for term, count in agricultural_terms
        ],
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "dataset_loaded": DATASET_PATH.is_file(),
            "dataset_file": DATASET_PATH.name,
        }
    )


@app.get("/api/analytics")
def analytics():
    try:
        payload = get_analytics()
        payload["topic_model_ready"] = (ARTIFACT_DIR / "topics.json").is_file()
        payload["search_ready"] = (ARTIFACT_DIR / "embeddings.npy").is_file()
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Unable to calculate dataset analytics.")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/entities")
def entities():
    try:
        return jsonify({"entities": get_analytics()["entities"]})
    except Exception as exc:
        logger.exception("Unable to extract agricultural entities.")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/topics")
def topics():
    try:
        return jsonify(_read_artifacts()["topics"])
    except ArtifactUnavailable as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Enter a search question or phrase."}), 400
    if len(query) > 500:
        return jsonify({"error": "Search queries must be 500 characters or fewer."}), 400
    try:
        limit = min(max(int(request.args.get("limit", 6)), 1), 10)
    except ValueError:
        return jsonify({"error": "Search limit must be a whole number."}), 400

    try:
        artifacts = _read_artifacts()
        query_vector = artifacts["model"].encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        candidate_limit = min(
            len(artifacts["search_records"]), max(limit * 8, 48)
        )
        if artifacts["index"] is not None:
            scores, indices = artifacts["index"].search(
                query_vector, candidate_limit
            )
            scores, indices = scores[0], indices[0]
        else:
            scores_all = np.asarray(artifacts["embeddings"] @ query_vector[0])
            indices = np.argsort(scores_all)[::-1][:candidate_limit]
            scores = scores_all[indices]
        ranked_records: list[dict[str, Any]] = []
        seen_records: set[int] = set()
        for score, index in zip(scores, indices):
            record_index = int(index)
            if record_index < 0:
                continue
            record = artifacts["search_records"][record_index]
            record_id = int(record["record_id"])
            if record_id in seen_records:
                continue
            seen_records.add(record_id)
            ranked_records.append(_serialize_result(record, float(score)))
            if len(ranked_records) >= 12:
                break
        visible_results = ranked_records[:limit]
        answer = _answer_from_evidence(query, visible_results)
        return jsonify(
            {
                "query": query,
                "answer": answer["answer"],
                "evidence_scope": answer["evidence_scope"],
                "evidence_count": answer["evidence_count"],
                "themes": answer["themes"],
                "agricultural_terms": answer["agricultural_terms"],
                "results": visible_results,
            }
        )
    except ArtifactUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Semantic search failed.")
        return jsonify({"error": f"Semantic search failed: {exc}"}), 500


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found."}), 404
    return render_template("index.html"), 404


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
