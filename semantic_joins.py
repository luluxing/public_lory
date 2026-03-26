from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


_COLUMN_DOMAIN_HINTS = {
    "state": ("state",),
    "country": ("country",),
}

_HARDCODED_EQUIVALENTS = {
    "state": {
        "il": "illinois",
        "ill": "illinois",
        "illinois": "illinois",
        "ia": "iowa",
        "iowa": "iowa",
        "in": "indiana",
        "ind": "indiana",
        "indiana": "indiana",
        "mn": "minnesota",
        "minnesota": "minnesota",
        "wi": "wisconsin",
        "wisconsin": "wisconsin",
    },
    "country": {
        "us": "usa",
        "usa": "usa",
        "unitedstates": "usa",
        "unitedstatesofamerica": "usa",
    },
}


@dataclass(frozen=True)
class SemanticColumnMatch:
    left_col: str
    right_col: str
    overlap: float


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _normalize_scalar(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return re.sub(r"[^a-z0-9]", "", text)


def _column_domain(column_name: str) -> str | None:
    normalized = normalize_column_name(column_name)
    for domain, hints in _COLUMN_DOMAIN_HINTS.items():
        if any(normalize_column_name(hint) in normalized for hint in hints):
            return domain
    return None


def canonicalize_value(value, column_name: str) -> str | None:
    normalized = _normalize_scalar(value)
    if not normalized:
        return None

    domain = _column_domain(column_name)
    if not domain:
        return normalized
    return _HARDCODED_EQUIVALENTS.get(domain, {}).get(normalized, normalized)


def canonicalize_series(series: pd.Series, column_name: str) -> pd.Series:
    return series.map(lambda value: canonicalize_value(value, column_name))


def parse_semantic_label(label: str) -> list[tuple[str, str]]:
    if not label:
        return []

    pairs: list[tuple[str, str]] = []
    for part in label.replace("…", "").split(","):
        item = part.strip()
        if not item:
            continue
        if "≈" in item:
            left, right = item.split("≈", 1)
            pairs.append((left.strip(), right.strip()))
        else:
            pairs.append((item, item))
    return pairs


def resolve_semantic_pairs(
    label: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    return [
        (left_col, right_col)
        for left_col, right_col in parse_semantic_label(label)
        if left_col in left_df.columns and right_col in right_df.columns
    ]


def format_semantic_label(
    matches: Sequence[SemanticColumnMatch],
    max_parts: int = 3,
) -> str:
    parts = [f"{match.left_col}≈{match.right_col}" for match in matches[:max_parts]]
    label = ", ".join(parts)
    if len(matches) > max_parts:
        label = f"{label}…"
    return label


def _series_signature(
    series: pd.Series,
    column_name: str,
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    raw_values: set[str] = set()
    canonical_values: set[str] = set()
    raw_variants_by_canonical: dict[str, set[str]] = {}

    for value in series.dropna():
        raw_value = _normalize_scalar(value)
        canonical_value = canonicalize_value(value, column_name)
        if not raw_value or not canonical_value:
            continue
        raw_values.add(raw_value)
        canonical_values.add(canonical_value)
        raw_variants_by_canonical.setdefault(canonical_value, set()).add(raw_value)

    return raw_values, canonical_values, raw_variants_by_canonical


def _has_value_level_semantic_gain(
    left_groups: dict[str, set[str]],
    right_groups: dict[str, set[str]],
    shared_canonical_values: Iterable[str],
) -> bool:
    for canonical_value in shared_canonical_values:
        if left_groups.get(canonical_value, set()) != right_groups.get(canonical_value, set()):
            return True
    return False


def find_semantic_column_matches(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    min_overlap: float = 0.0,
) -> list[SemanticColumnMatch]:
    norm_to_cols_left: dict[str, list[str]] = {}
    norm_to_cols_right: dict[str, list[str]] = {}

    for column in left_df.columns:
        norm_to_cols_left.setdefault(normalize_column_name(column), []).append(column)
    for column in right_df.columns:
        norm_to_cols_right.setdefault(normalize_column_name(column), []).append(column)

    left_signatures = {
        column: _series_signature(left_df[column], column) for column in left_df.columns
    }
    right_signatures = {
        column: _series_signature(right_df[column], column) for column in right_df.columns
    }

    matches: list[SemanticColumnMatch] = []
    for normalized_name in sorted(set(norm_to_cols_left) & set(norm_to_cols_right)):
        if not normalized_name:
            continue
        for left_col in norm_to_cols_left[normalized_name]:
            for right_col in norm_to_cols_right[normalized_name]:
                _, left_canonical, left_groups = left_signatures[left_col]
                _, right_canonical, right_groups = right_signatures[right_col]
                shared_canonical = left_canonical & right_canonical
                if not shared_canonical:
                    continue

                overlap = len(shared_canonical) / max(
                    1, min(len(left_canonical), len(right_canonical))
                )
                if overlap < min_overlap:
                    continue

                has_semantic_gain = left_col != right_col or _has_value_level_semantic_gain(
                    left_groups,
                    right_groups,
                    shared_canonical,
                )
                if not has_semantic_gain:
                    continue

                matches.append(
                    SemanticColumnMatch(
                        left_col=left_col,
                        right_col=right_col,
                        overlap=overlap,
                    )
                )

    return sorted(matches, key=lambda match: (-match.overlap, match.left_col, match.right_col))


def semantic_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    column_pairs: Sequence[tuple[str, str]],
    how: str = "inner",
    right_columns: Sequence[str] | None = None,
    suffixes: tuple[str, str] = ("", "_dup"),
) -> pd.DataFrame | None:
    valid_pairs = [
        (left_col, right_col)
        for left_col, right_col in column_pairs
        if left_col in left_df.columns and right_col in right_df.columns
    ]
    if not valid_pairs:
        return None

    left_tmp = left_df.copy()
    right_tmp = right_df.copy()
    left_keys: list[str] = []
    right_keys: list[str] = []

    for idx, (left_col, right_col) in enumerate(valid_pairs):
        left_key = f"__semantic_left_key_{idx}__"
        right_key = f"__semantic_right_key_{idx}__"
        left_tmp[left_key] = canonicalize_series(left_tmp[left_col], left_col)
        right_tmp[right_key] = canonicalize_series(right_tmp[right_col], right_col)

        left_missing = left_tmp[left_key].isna()
        right_missing = right_tmp[right_key].isna()
        if left_missing.any():
            left_tmp.loc[left_missing, left_key] = [
                f"__semantic_left_missing_{idx}_{row_idx}__"
                for row_idx in left_tmp.index[left_missing]
            ]
        if right_missing.any():
            right_tmp.loc[right_missing, right_key] = [
                f"__semantic_right_missing_{idx}_{row_idx}__"
                for row_idx in right_tmp.index[right_missing]
            ]

        left_keys.append(left_key)
        right_keys.append(right_key)

    right_projection = list(right_columns) if right_columns is not None else list(right_tmp.columns)
    right_projection = list(dict.fromkeys(right_keys + right_projection))

    merged = pd.merge(
        left_tmp,
        right_tmp[right_projection],
        left_on=left_keys,
        right_on=right_keys,
        how=how,
        suffixes=suffixes,
    )
    return merged.drop(columns=left_keys + right_keys, errors="ignore")
