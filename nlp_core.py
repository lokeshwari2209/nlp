"""Shared data preparation and analysis for the parliamentary debate dashboard."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_DATASET = Path(__file__).with_name(
    "Combined_Agriculture_LokSabha_RajyaSabha.xlsx"
)
TEXT_COLUMNS = [
    "debateTitle",
    "debateTypeDesc",
    "keywordUsed",
    "mpPartDetailList",
    "title",
    "debateTitleSubject",
    "type",
    "ministry",
    "content",
    "contents",
    "question",
    "answer",
    "debate",
    "speech",
    "description",
    "subject",
    "detail",
    "text",
]
CUSTOM_STOPWORDS = {
    "question",
    "answer",
    "part",
    "written",
    "unstarred",
    "starred",
    "mpname",
    "mpcode",
    "mppartcode",
    "shri",
    "matter",
    "regarding",
    "affair",
}
AGRICULTURE_KEYWORDS = {
    "CROP": [
        "rice",
        "wheat",
        "turmeric",
        "cotton",
        "sugarcane",
        "maize",
        "millet",
        "pulses",
        "groundnut",
        "banana",
        "coconut",
        "tea",
        "coffee",
    ],
    "ISSUE": [
        "drought",
        "flood",
        "pest",
        "disease",
        "water shortage",
        "climate change",
        "farmer suicide",
    ],
    "POLICY": [
        "msp",
        "minimum support price",
        "pm kisan",
        "crop insurance",
        "subsidy",
        "agricultural loan",
    ],
    "RESOURCE": ["fertilizer", "irrigation", "water", "seed", "pesticide"],
}

_STOP_WORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def ensure_nltk_resources(download: bool = False) -> None:
    """Make the notebook's NLTK tokenization and lemmatization resources available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    missing: list[str] = []
    for location, package in resources:
        zip_location = f"{location}.zip"
        try:
            nltk.data.find(location)
        except LookupError:
            try:
                nltk.data.find(zip_location)
                continue
            except LookupError:
                if not download:
                    missing.append(package)
                    continue
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {package}")
            try:
                nltk.data.find(location)
            except LookupError:
                try:
                    nltk.data.find(zip_location)
                except LookupError as exc:
                    raise RuntimeError(
                        f"NLTK resource was not installed successfully: {package}"
                    ) from exc
    if missing:
        raise RuntimeError(
            "Missing NLTK resources: "
            + ", ".join(missing)
            + ". Run `python init_artifacts.py` to install them."
        )


def _nlp_resources() -> tuple[set[str], WordNetLemmatizer]:
    global _STOP_WORDS, _LEMMATIZER
    if _STOP_WORDS is None or _LEMMATIZER is None:
        ensure_nltk_resources(download=True)
        _STOP_WORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS
        _LEMMATIZER = WordNetLemmatizer()
    return _STOP_WORDS, _LEMMATIZER


def preprocess_text(value: Any) -> str:
    """Apply the same text normalization and lemmatization as the Colab notebook."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_words, lemmatizer = _nlp_resources()
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)


def _first_available_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[available].bfill(axis=1).iloc[:, 0]


def _clean_house_column(frame: pd.DataFrame) -> pd.Series:
    if "House" in frame.columns:
        house = frame["House"]
    elif "house" in frame.columns:
        house = frame["house"]
    else:
        inferred = next(
            (
                column
                for column in frame.columns
                if column.lower() in {"loksabha", "rajyasabha"}
            ),
            None,
        )
        house = pd.Series(inferred or "Unknown", index=frame.index)
    return house.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")


def load_records(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load, normalize, and preprocess the actual Excel records."""
    path = Path(dataset_path or DEFAULT_DATASET)
    if not path.is_file():
        raise FileNotFoundError(f"Excel dataset not found: {path}")

    try:
        frame = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Unable to read Excel dataset {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Excel dataset contains no records: {path}")
    workbook_rows = len(frame)

    frame.columns = [str(column).strip() for column in frame.columns]
    frame["House"] = _clean_house_column(frame)

    date_values = _first_available_series(frame, ["debateDate", "date"])
    frame["unified_date"] = pd.to_datetime(
        date_values, errors="coerce", format="mixed", dayfirst=True
    )
    date_year = frame["unified_date"].dt.year
    if "year" in frame.columns:
        source_year = pd.to_numeric(frame["year"], errors="coerce")
        frame["unified_year"] = date_year.fillna(source_year)
    else:
        frame["unified_year"] = date_year
    frame["unified_year"] = pd.to_numeric(
        frame["unified_year"], errors="coerce"
    )

    member_values = _first_available_series(
        frame, ["memberName", "questionerName", "members"]
    )
    frame["unified_member"] = member_values.fillna("Unknown").astype(str)

    available_text_columns = [
        column for column in TEXT_COLUMNS if column in frame.columns
    ]
    if not available_text_columns:
        raise ValueError(
            "The workbook has none of the notebook's supported text columns."
        )
    frame["text"] = (
        frame[available_text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame = frame.loc[frame["text"].str.len() > 0].copy()
    frame = frame.drop_duplicates(subset=["House", "text"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-empty parliamentary records remain after cleaning.")

    frame["record_id"] = range(len(frame))
    frame["clean_text"] = frame["text"].map(preprocess_text)
    frame = frame.loc[frame["clean_text"].str.len() > 0].reset_index(drop=True)
    frame["record_id"] = range(len(frame))
    if frame.empty:
        raise ValueError("No records remain after NLP preprocessing.")
    frame.attrs["workbook_rows"] = workbook_rows
    frame.attrs["source_text_fields"] = available_text_columns
    return frame


def extract_entities(value: Any) -> dict[str, list[str]]:
    """Apply the notebook's agriculture keyword lists to one record's source text."""
    text = "" if pd.isna(value) else str(value).lower()
    return {
        category: [keyword for keyword in keywords if keyword in text]
        for category, keywords in AGRICULTURE_KEYWORDS.items()
    }


def display_title(row: pd.Series | dict[str, Any]) -> str:
    for column in ("debateTitle", "title", "debateTitleSubject"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "Untitled parliamentary record"


def _entity_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    category_totals: dict[str, Counter[str]] = {
        category: Counter() for category in AGRICULTURE_KEYWORDS
    }
    records_by_category: Counter[str] = Counter()
    for text in frame["text"]:
        found = extract_entities(text)
        for category, terms in found.items():
            if terms:
                records_by_category[category] += 1
                category_totals[category].update(set(terms))

    return [
        {
            "category": category.title(),
            "record_count": records_by_category[category],
            "terms": [
                {"term": term, "count": count}
                for term, count in category_totals[category].most_common()
            ],
        }
        for category in AGRICULTURE_KEYWORDS
    ]


def build_analytics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute real dataset statistics, frequency, TF-IDF, and entity summaries."""
    years = (
        frame["unified_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    houses = frame["House"].value_counts().sort_index()
    all_words = " ".join(frame["clean_text"].astype(str)).split()
    word_counts = Counter(all_words)

    tfidf_terms: list[dict[str, Any]] = []
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2
        )
        matrix = vectorizer.fit_transform(frame["clean_text"].astype(str))
        scores = matrix.sum(axis=0).A1
        features = vectorizer.get_feature_names_out()
        ranked = sorted(
            zip(features, scores), key=lambda item: float(item[1]), reverse=True
        )
        tfidf_terms = [
            {"term": str(term), "score": round(float(score), 5)}
            for term, score in ranked[:20]
        ]
    except ValueError:
        tfidf_terms = []

    year_values = years.index.tolist()
    valid_dates = frame["unified_date"].dropna()
    return {
        "summary": {
            "record_count": int(len(frame)),
            "workbook_rows": int(frame.attrs.get("workbook_rows", len(frame))),
            "house_count": int(frame["House"].nunique()),
            "year_count": int(len(years)),
            "records_with_year": int(frame["unified_year"].notna().sum()),
            "year_start": int(min(year_values)) if year_values else None,
            "year_end": int(max(year_values)) if year_values else None,
            "date_start": valid_dates.min().date().isoformat()
            if not valid_dates.empty
            else None,
            "date_end": valid_dates.max().date().isoformat()
            if not valid_dates.empty
            else None,
            "records_with_date": int(frame["unified_date"].notna().sum()),
            "text_fields_used": list(
                frame.attrs.get(
                    "source_text_fields",
                    [column for column in TEXT_COLUMNS if column in frame.columns],
                )
            ),
        },
        "year_trend": [
            {"year": int(year), "count": int(count)}
            for year, count in years.items()
        ],
        "houses": [
            {"house": str(house), "count": int(count)}
            for house, count in houses.items()
        ],
        "top_words": [
            {"word": word, "count": int(count)}
            for word, count in word_counts.most_common(20)
        ],
        "tfidf_terms": tfidf_terms,
        "entities": _entity_summary(frame),
    }


def chunk_text(text: Any, chunk_size: int = 250, overlap: int = 40) -> list[str]:
    """Use the notebook's word-based chunk size and overlap for retrieval."""
    words = str(text).split()
    if not words:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks
