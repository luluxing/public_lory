from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

import semantic_joins


RelEdge = Tuple[str, str, str]


def _shared_columns(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    left_spatial = set(_find_spatial_columns(left))
    right_spatial = set(_find_spatial_columns(right))
    shared = sorted(
        col
        for col in set(left.columns).intersection(right.columns)
        if not (col in left_spatial and col in right_spatial)
    )
    return shared


def _normalize_column_name(name: str) -> str:
    return semantic_joins.normalize_column_name(name)


def find_semantic_joins(
    tables: dict[str, pd.DataFrame],
    min_overlap: float = 0.0,
) -> List[RelEdge]:
    """
    Column pairs that align after normalization or value canonicalization, such as
    school_id vs SchoolID or state values like IL vs Illinois.
    """
    edges: List[RelEdge] = []
    for (name_a, df_a), (name_b, df_b) in combinations(tables.items(), 2):
        matched_pairs = semantic_joins.find_semantic_column_matches(
            df_a,
            df_b,
            min_overlap=min_overlap,
        )
        if not matched_pairs:
            continue
        label = semantic_joins.format_semantic_label(matched_pairs)
        edges.append((name_a, name_b, label))
    return edges


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


def _looks_like_polygon(value: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().upper()
    return normalized.startswith("POLYGON(") or normalized.startswith("MULTIPOLYGON(")


def _find_spatial_columns(df: pd.DataFrame) -> List[str]:
    spatial_cols: List[str] = []
    lat_col = None
    lon_col = None
    for col in df.columns:
        if "location" in col.lower() or "coordinate" in col.lower() or "polygon" in col.lower():
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
        if any(_looks_like_point(v) or _looks_like_polygon(v) for v in sample):
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


def relational_adjacency(tables: dict[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    """Undirected graph: edge if two tables share at least one column name."""
    names = list(tables.keys())
    adj: Dict[str, Set[str]] = {n: set() for n in names}
    for i, ni in enumerate(names):
        for nj in names[i + 1 :]:
            if _shared_columns(tables[ni], tables[nj]):
                adj[ni].add(nj)
                adj[nj].add(ni)
    return adj


def bfs_shortest_path_to_geo_table(
    start: str,
    tables: dict[str, pd.DataFrame],
    native_geo_tables: Set[str],
) -> Optional[List[str]]:
    """Return shortest relational path from start to any table in native_geo_tables."""
    if start in native_geo_tables:
        return [start]
    if start not in tables:
        return None
    adj = relational_adjacency(tables)
    q: deque[Tuple[str, List[str]]] = deque([(start, [start])])
    seen: Set[str] = {start}
    while q:
        node, path = q.popleft()
        for nb in adj.get(node, ()):
            if nb in seen:
                continue
            next_path = path + [nb]
            if nb in native_geo_tables:
                return next_path
            seen.add(nb)
            q.append((nb, next_path))
    return None


def physical_geo_augment_along_path(
    tables: dict[str, pd.DataFrame],
    path: List[str],
) -> Optional[pd.DataFrame]:
    """
    Left-merge each hop along `path`, starting from path[0], adding columns from the
    right table at each step. The last table in path should carry spatial columns.
    """
    if len(path) < 2:
        return None
    result = tables[path[0]].copy()
    for i in range(len(path) - 1):
        right = tables[path[i + 1]]
        keys = _shared_columns(result, right)
        if not keys:
            return None
        merge_cols = list(
            dict.fromkeys(keys + [c for c in right.columns if c not in result.columns])
        )
        right_slice = right[merge_cols].drop_duplicates(subset=keys)
        result = result.merge(right_slice, on=keys, how="left")
    return result
