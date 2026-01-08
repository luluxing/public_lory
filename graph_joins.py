from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Tuple

import pandas as pd


RelEdge = Tuple[str, str, str]


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    shared = sorted(set(left.columns).intersection(right.columns))
    return shared


def find_relational_joins(tables: dict[str, pd.DataFrame]) -> List[RelEdge]:
    """
    Heuristic join discovery based on shared column names.

    Returns a list of (table_a, table_b, label) edges.
    """
    edges: List[RelEdge] = []
    for (name_a, df_a), (name_b, df_b) in combinations(tables.items(), 2):
        shared = _shared_columns(df_a, df_b)
        if not shared:
            continue
        label = ", ".join(shared[:3])
        if len(shared) > 3:
            label = f"{label}…"
        edges.append((name_a, name_b, label))
    return edges


def _looks_like_point(value: str) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith("POINT(")


def _find_spatial_columns(df: pd.DataFrame) -> List[str]:
    spatial_cols: List[str] = []
    lat_col = None
    lon_col = None
    for col in df.columns:
        if "location" in col.lower() or "coordinate" in col.lower():
            spatial_cols.append(col)
            continue
        if lat_col is None and "lat" in col.lower():
            sample = pd.to_numeric(df[col].dropna().head(10), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-90, 90).all():
                lat_col = col
        if lon_col is None and ("lon" in col.lower() or "lng" in col.lower()):
            sample = pd.to_numeric(df[col].dropna().head(10), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-180, 180).all():
                lon_col = col
        sample = df[col].dropna().astype(str).head(10)
        if any(_looks_like_point(v) for v in sample):
            spatial_cols.append(col)
    if lat_col and lon_col:
        spatial_cols.extend([lat_col, lon_col])
    return spatial_cols


def find_spatial_joins(tables: dict[str, pd.DataFrame]) -> List[RelEdge]:
    """
    Heuristic spatial join discovery based on POINT(...) columns.
    """
    spatial_by_table = {
        name: _find_spatial_columns(df) for name, df in tables.items()
    }
    spatial_tables = {
        name: cols for name, cols in spatial_by_table.items() if cols
    }

    edges: List[RelEdge] = []
    for (name_a, cols_a), (name_b, cols_b) in combinations(spatial_tables.items(), 2):
        label = ", ".join(sorted(set(cols_a + cols_b))[:3])
        if len(set(cols_a + cols_b)) > 3:
            label = f"{label}…"
        edges.append((name_a, name_b, label))
    return edges
