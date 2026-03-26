from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


def _is_unique_non_null(series: pd.Series) -> bool:
    if series.empty:
        return False
    return series.notna().all() and series.is_unique


def infer_primary_key(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    exact_id_candidates = [
        col for col in df.columns if str(col).strip().lower() == "id"
    ]
    suffix_id_candidates = [
        col for col in df.columns
        if str(col).strip().lower().endswith("_id")
        or str(col).strip().lower().endswith("id")
    ]
    ordered_candidates = list(
        dict.fromkeys(exact_id_candidates + suffix_id_candidates + list(df.columns))
    )

    for column in ordered_candidates:
        if _is_unique_non_null(df[column]):
            return str(column)
    return None


def _normalize_attribute_scores(values: object) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if not isinstance(values, dict):
        return normalized
    for raw_attr, raw_score in values.items():
        attr = str(raw_attr).strip()
        if not attr:
            continue
        try:
            score = round(float(raw_score), 3)
        except (TypeError, ValueError):
            continue
        normalized[attr] = min(max(score, 0.0), 1.0)
    return normalized


def _normalize_origin_attributes(values: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        attr = str(value).strip()
        if not attr or attr in seen:
            continue
        seen.add(attr)
        normalized.append(attr)
    return normalized


def geo_attribute_scores(
    df: pd.DataFrame,
    geo_attribute: str,
    geo_association_details: dict[str, dict[str, object]] | None = None,
) -> dict[str, float]:
    detail = dict((geo_association_details or {}).get(str(geo_attribute), {}) or {})
    scores = _normalize_attribute_scores(detail.get("attribute_scores"))

    primary_key = infer_primary_key(df)
    if primary_key and primary_key not in scores:
        scores[primary_key] = 1.0

    for attribute in _normalize_origin_attributes(detail.get("origin_attributes")):
        if attribute not in scores:
            scores[attribute] = 1.0

    return {
        attribute: round(score, 3)
        for attribute, score in scores.items()
        if attribute
    }


@dataclass(frozen=True)
class NormalizedEntropyResult:
    score: float
    dropped_rows: int
    used_rows: int


def _series_entropy(series: pd.Series) -> float:
    value_counts = series.value_counts(dropna=False)
    total = int(value_counts.sum())
    if total <= 1:
        return 0.0

    entropy = 0.0
    for count in value_counts.tolist():
        probability = count / total
        if probability <= 0:
            continue
        entropy -= probability * math.log2(probability)
    return entropy


def normalized_entropy_score(
    df: pd.DataFrame,
    primary_key: str,
    attribute: str,
) -> NormalizedEntropyResult:
    if (
        not primary_key
        or not attribute
        or primary_key not in df.columns
        or attribute not in df.columns
    ):
        return NormalizedEntropyResult(score=0.0, dropped_rows=0, used_rows=0)

    filtered_df = df[[primary_key, attribute]].dropna()
    dropped_rows = max(0, len(df) - len(filtered_df))
    if filtered_df.empty or not filtered_df[primary_key].is_unique:
        return NormalizedEntropyResult(
            score=0.0,
            dropped_rows=dropped_rows,
            used_rows=len(filtered_df),
        )

    primary_key_entropy = _series_entropy(filtered_df[primary_key])
    if primary_key_entropy <= 0:
        return NormalizedEntropyResult(
            score=0.0,
            dropped_rows=dropped_rows,
            used_rows=len(filtered_df),
        )

    attribute_entropy = _series_entropy(filtered_df[attribute])
    score = round(attribute_entropy / primary_key_entropy, 3)
    score = min(max(score, 0.0), 1.0)
    return NormalizedEntropyResult(
        score=score,
        dropped_rows=dropped_rows,
        used_rows=len(filtered_df),
    )


def geo_associativity_score(
    df: pd.DataFrame,
    attribute: str,
    geo_attribute: str,
    geo_association_details: dict[str, dict[str, object]] | None = None,
) -> float:
    scores = geo_attribute_scores(
        df,
        geo_attribute,
        geo_association_details=geo_association_details,
    )
    return round(scores.get(str(attribute), 0.0), 3)


def top_geo_associativity(
    df: pd.DataFrame,
    geo_columns: list[str] | set[str],
    geo_association_details: dict[str, dict[str, object]] | None = None,
) -> list[tuple[str, str, float]]:
    primary_key = infer_primary_key(df)
    top_scores: list[tuple[str, str, float]] = []
    for geo_column in sorted(dict.fromkeys(str(column) for column in geo_columns)):
        scores = geo_attribute_scores(
            df,
            geo_column,
            geo_association_details=geo_association_details,
        )
        if not scores:
            continue
        best_attribute, best_score = min(
            scores.items(),
            key=lambda item: (
                -item[1],
                0 if primary_key and item[0] == primary_key else 1,
                item[0],
            ),
        )
        top_scores.append((geo_column, best_attribute, round(best_score, 3)))
    return top_scores
