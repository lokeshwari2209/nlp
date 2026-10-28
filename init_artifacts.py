"""Build the saved BERTopic assignments and Sentence-BERT/FAISS search index."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app import ARTIFACT_DIR, DATASET_PATH, MODEL_NAME, _dataset_digest
from nlp_core import chunk_text, ensure_nltk_resources, load_records


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _record_title(row: Any) -> str:
    for column in ("debateTitle", "title", "debateTitleSubject"):
        value = row.get(column)
        if value is not None and str(value).strip() and str(value) != "nan":
            return str(value).strip()
    return "Untitled parliamentary record"


def _build_topic_data(frame):
    from bertopic import BERTopic

    documents = frame["clean_text"].astype(str).tolist()
    model = BERTopic(language="english", min_topic_size=20, verbose=True)
    topics, _probabilities = model.fit_transform(documents)
    info = model.get_topic_info()
    labels: dict[int, str] = {}
    topic_rows: list[dict[str, Any]] = []
    for _, item in info.iterrows():
        topic_id = int(item["Topic"])
        words = model.get_topic(topic_id) or []
        if topic_id == -1:
            label = "Other / Outlier"
        else:
            label = " / ".join(word for word, _score in words[:5]) or "Unlabeled"
        labels[topic_id] = label
        topic_rows.append(
            {
                "topic_id": topic_id,
                "label": label,
                "count": int(item["Count"]),
                    "keywords": [
                    {"word": str(word), "weight": round(float(score), 6)}
                    for word, score in words[:10]
                ],
            }
        )
    record_topics = [
        {"topic_id": int(topic), "label": labels.get(int(topic), "Unknown")}
        for topic in topics
    ]
    topic_rows.sort(key=lambda item: item["count"], reverse=True)
    return {
        "topics": topic_rows,
        "topics_by_record": record_topics,
        "document_count": len(documents),
        "configuration": {"language": "english", "min_topic_size": 20},
    }


def _build_search_data(frame, artifact_dir: Path):
    from sentence_transformers import SentenceTransformer

    search_records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for chunk in chunk_text(row["text"], chunk_size=250, overlap=40):
            search_records.append(
                {
                    "record_id": int(row["record_id"]),
                    "chunk_text": chunk,
                    "house": str(row["House"]),
                    "year": int(row["unified_year"])
                    if row["unified_year"] == row["unified_year"]
                    else None,
                    "title": _record_title(row),
                }
            )
    if not search_records:
        raise ValueError("No searchable text chunks were generated.")

    logger.info(
        "Loading %s to encode %s source-text chunks.",
        MODEL_NAME,
        len(search_records),
    )
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [record["chunk_text"] for record in search_records],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    if len(embeddings) != len(search_records):
        raise ValueError("Embedding count does not match generated search chunks.")

    model_dir = artifact_dir / "sentence_transformer"
    model.save(str(model_dir))
    np.save(artifact_dir / "embeddings.npy", embeddings)
    (artifact_dir / "search_records.json").write_text(
        json.dumps(search_records, ensure_ascii=False), encoding="utf-8"
    )

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(artifact_dir / "faiss.index"))
        backend = "faiss.IndexFlatIP"
    except ImportError:
        backend = "numpy inner product (FAISS not installed)"
        logger.warning(
            "FAISS is not installed; saved normalized vectors for exact "
            "NumPy inner-product search."
        )
    return len(search_records), int(embeddings.shape[1]), backend


def build_artifacts() -> None:
    ensure_nltk_resources(download=True)
    frame = load_records(DATASET_PATH)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared %s actual Excel records.", len(frame))

    topic_data = _build_topic_data(frame)
    (ARTIFACT_DIR / "topics.json").write_text(
        json.dumps(topic_data, ensure_ascii=False), encoding="utf-8"
    )
    chunk_count, dimension, backend = _build_search_data(frame, ARTIFACT_DIR)
    manifest = {
        "dataset_file": DATASET_PATH.name,
        "dataset_sha256": _dataset_digest(),
        "record_count": len(frame),
        "search_chunk_count": chunk_count,
        "embedding_dimension": dimension,
        "model_name": MODEL_NAME,
        "search_backend": backend,
        "text_columns": list(frame.attrs.get("source_text_fields", [])),
        "topic_configuration": topic_data["configuration"],
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Artifacts ready: %s topic documents, %s search chunks, %s.",
        len(frame),
        chunk_count,
        backend,
    )


if __name__ == "__main__":
    build_artifacts()
