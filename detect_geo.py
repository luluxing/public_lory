from __future__ import annotations

import re

import pandas as pd

import geocoding


PLACE_KEYWORDS = frozenset(
    {
        "borough",
        "city",
        "country",
        "county",
        "district",
        "locality",
        "municipality",
        "neighborhood",
        "parish",
        "precinct",
        "province",
        "region",
        "state",
        "territory",
        "town",
        "village",
        "ward",
    }
)


def tokenize_column_name(name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return {part for part in normalized.split() if part}


def implicit_geo_kind(column_name: str) -> str | None:
    lowered = str(column_name).strip().lower()
    tokens = tokenize_column_name(lowered)

    if lowered in geocoding.ADDRESS_COLUMN_CANDIDATES or "address" in tokens:
        return "Address"
    if (
        lowered == geocoding.ZIPCODE_COLUMN
        or "zipcode" in lowered
        or "zip" in tokens
        or "postal" in tokens
        or "postcode" in tokens
    ):
        return "Postal area"
    if tokens.intersection(PLACE_KEYWORDS):
        return "Place name"
    return None


def implicit_geo_columns(
    df: pd.DataFrame,
    *,
    excluded_columns: set[str] | list[str] | tuple[str, ...] | dict[str, object] | None = None,
) -> list[dict[str, str]]:
    excluded = {str(column) for column in (excluded_columns or [])}
    columns: list[dict[str, str]] = []
    for column_name in df.columns:
        column_label = str(column_name)
        kind = implicit_geo_kind(column_label)
        if kind is None or column_label in excluded:
            continue
        columns.append({"name": column_label, "kind": kind})
    return columns


def has_implicit_geo_columns(
    df: pd.DataFrame,
    *,
    excluded_columns: set[str] | list[str] | tuple[str, ...] | dict[str, object] | None = None,
) -> bool:
    return bool(implicit_geo_columns(df, excluded_columns=excluded_columns))
