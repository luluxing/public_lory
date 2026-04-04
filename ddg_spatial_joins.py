from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from data_path import data_path
import pandas_helper


POINT_WKT_RE = re.compile(r"^\s*POINT\s*\(", re.IGNORECASE)
POLYGON_WKT_RE = re.compile(r"^\s*(?:MULTI)?POLYGON\s*\(", re.IGNORECASE)

PREDICATE_DISPLAY = {
    "distance": "Distance",
    "containment": "Containment",
    "overlap": "Overlapping",
}

PREDICATE_PRIORITY = {
    "overlap": 3,
    "containment": 2,
    "distance": 1,
}

ORIGIN_RANK = {
    "geo-coded": 3,
    "native": 2,
    "geo-augmented": 1,
}


@dataclass(frozen=True)
class GeometryAttribute:
    table_name: str
    attribute_name: str
    geometry_type: str
    origin: str
    source_kind: str

    @property
    def origin_rank(self) -> int:
        return ORIGIN_RANK.get(self.origin, 0)

    @property
    def is_augmented(self) -> bool:
        return self.origin == "geo-augmented"

    @property
    def display_ref(self) -> str:
        return f"{self.table_name}.{self.attribute_name}"

    @property
    def description(self) -> str:
        return f"{self.display_ref} ({self.geometry_type}, {self.origin})"


@dataclass(frozen=True)
class SpatialCandidate:
    source: GeometryAttribute
    target: GeometryAttribute
    predicate: str
    directed: bool

    @property
    def source_table(self) -> str:
        return self.source.table_name

    @property
    def target_table(self) -> str:
        return self.target.table_name

    @property
    def source_attribute(self) -> str:
        return self.source.attribute_name

    @property
    def target_attribute(self) -> str:
        return self.target.attribute_name

    @property
    def predicate_display(self) -> str:
        return PREDICATE_DISPLAY.get(self.predicate, self.predicate.title())


def _sample_strings(series: pd.Series, limit: int = 20) -> pd.Series:
    return series.dropna().astype(str).head(limit)


def infer_geometry_type_from_name(name: str) -> str | None:
    lowered = str(name).lower()
    if "polygon" in lowered:
        return "polygon"
    if "lat" in lowered and ("lon" in lowered or "lng" in lowered):
        return "point"
    if "coordinate" in lowered or "location" in lowered or "point" in lowered:
        return "point"
    return None


def _infer_geometry_type_from_series(series: pd.Series, column_name: str) -> str | None:
    samples = _sample_strings(series)
    if samples.empty:
        return None
    if samples.map(lambda value: bool(POLYGON_WKT_RE.match(value))).any():
        return "polygon"
    if samples.map(lambda value: bool(POINT_WKT_RE.match(value))).any():
        return "point"
    return infer_geometry_type_from_name(column_name)


def _find_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if lat_col is None and "lat" in col_lower:
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-90, 90).all():
                lat_col = str(col)
        if lon_col is None and ("lon" in col_lower or "lng" in col_lower):
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-180, 180).all():
                lon_col = str(col)
    return lat_col, lon_col


def geometry_attributes_from_dataframe(
    table_name: str,
    df: pd.DataFrame,
    *,
    origin: str,
    source_kind: str,
) -> list[GeometryAttribute]:
    attrs: list[GeometryAttribute] = []
    seen: set[tuple[str, str]] = set()

    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        attr_name = f"{lat_col} + {lon_col}"
        seen.add((attr_name, "point"))
        attrs.append(
            GeometryAttribute(
                table_name=table_name,
                attribute_name=attr_name,
                geometry_type="point",
                origin=origin,
                source_kind=source_kind,
            )
        )

    for column in df.columns:
        column_name = str(column)
        geometry_type = _infer_geometry_type_from_series(df[column], column_name)
        if geometry_type is None:
            continue
        key = (column_name, geometry_type)
        if key in seen:
            continue
        seen.add(key)
        attrs.append(
            GeometryAttribute(
                table_name=table_name,
                attribute_name=column_name,
                geometry_type=geometry_type,
                origin=origin,
                source_kind=source_kind,
            )
        )

    return attrs


def _current_lake_dir(chosen_lake: str | None) -> Path | None:
    if not chosen_lake or chosen_lake == "<no datalakes found>":
        return None
    return data_path() / "datalakes" / chosen_lake


def _load_authoritative_tables(
    chosen_lake: str | None,
    in_memory_tables: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    tables: dict[str, pd.DataFrame] = {}
    origins: dict[str, str] = {}
    memory_tables = dict(in_memory_tables or {})
    lake_dir = _current_lake_dir(chosen_lake)

    base_names: set[str] = set()
    if lake_dir is not None:
        all_tables_dir = lake_dir / "all_tables"
        if all_tables_dir.is_dir():
            base_names = {path.stem for path in all_tables_dir.glob("*.csv")}

            temp_dir = pandas_helper.temp_tables_dir()
            overridden: set[str] = set()
            for temp_path in sorted(temp_dir.glob("*.csv")):
                table_name = temp_path.stem
                if table_name not in base_names and table_name not in memory_tables:
                    continue
                tables[table_name] = pandas_helper.read_csv_safely(temp_path)
                origins[table_name] = "geo-coded"
                overridden.add(table_name)

            for base_path in sorted(all_tables_dir.glob("*.csv")):
                table_name = base_path.stem
                if table_name in overridden:
                    continue
                tables[table_name] = pandas_helper.read_csv_safely(base_path)
                origins[table_name] = "native"

    for table_name, df in memory_tables.items():
        if table_name in tables:
            continue
        tables[table_name] = df
        origins[table_name] = "native"

    return tables, origins


def discover_table_geometries(
    chosen_lake: str | None,
    *,
    in_memory_tables: Mapping[str, pd.DataFrame] | None = None,
    manual_geo_details: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> dict[str, list[GeometryAttribute]]:
    tables, origins = _load_authoritative_tables(
        chosen_lake=chosen_lake,
        in_memory_tables=in_memory_tables,
    )

    geometry_map: dict[str, list[GeometryAttribute]] = {}
    for table_name, df in tables.items():
        geometry_map[table_name] = geometry_attributes_from_dataframe(
            table_name,
            df,
            origin=origins.get(table_name, "native"),
            source_kind="physical",
        )

    for table_name, attr_map in (manual_geo_details or {}).items():
        table_geometries = geometry_map.setdefault(table_name, [])
        existing = {
            (attr.attribute_name, attr.geometry_type)
            for attr in table_geometries
        }
        for attribute_name, detail in sorted(attr_map.items()):
            geometry_type = (
                str(detail.get("geometry_type") or "").strip().lower()
                or infer_geometry_type_from_name(attribute_name)
            )
            if geometry_type not in {"point", "polygon"}:
                continue
            key = (str(attribute_name), geometry_type)
            if key in existing:
                continue
            existing.add(key)
            table_geometries.append(
                GeometryAttribute(
                    table_name=table_name,
                    attribute_name=str(attribute_name),
                    geometry_type=geometry_type,
                    origin="geo-augmented",
                    source_kind="manual",
                )
            )

    for table_name, attrs in geometry_map.items():
        geometry_map[table_name] = sorted(
            attrs,
            key=lambda attr: (
                -attr.origin_rank,
                -PREDICATE_PRIORITY.get(
                    "overlap" if attr.geometry_type == "polygon" else "distance",
                    0,
                ),
                attr.attribute_name,
            ),
        )

    return geometry_map


def _candidate_for_pair(
    left: GeometryAttribute,
    right: GeometryAttribute,
) -> SpatialCandidate | None:
    geometry_pair = (left.geometry_type, right.geometry_type)
    if geometry_pair == ("point", "point"):
        source, target = (
            (left, right)
            if (left.table_name, left.attribute_name) <= (right.table_name, right.attribute_name)
            else (right, left)
        )
        return SpatialCandidate(
            source=source,
            target=target,
            predicate="distance",
            directed=False,
        )
    if geometry_pair == ("polygon", "point"):
        return SpatialCandidate(
            source=left,
            target=right,
            predicate="containment",
            directed=True,
        )
    if geometry_pair == ("point", "polygon"):
        return SpatialCandidate(
            source=right,
            target=left,
            predicate="containment",
            directed=True,
        )
    if geometry_pair == ("polygon", "polygon"):
        source, target = (
            (left, right)
            if (left.table_name, left.attribute_name) <= (right.table_name, right.attribute_name)
            else (right, left)
        )
        return SpatialCandidate(
            source=source,
            target=target,
            predicate="overlap",
            directed=False,
        )
    return None


def _candidate_sort_key(candidate: SpatialCandidate) -> tuple[int, int, int, str, str]:
    return (
        min(candidate.source.origin_rank, candidate.target.origin_rank),
        max(candidate.source.origin_rank, candidate.target.origin_rank),
        PREDICATE_PRIORITY.get(candidate.predicate, 0),
        candidate.source.attribute_name,
        candidate.target.attribute_name,
    )


def _direction_display(candidate: SpatialCandidate) -> str:
    if candidate.directed:
        return f"{candidate.source.display_ref} -> {candidate.target.display_ref}"
    return f"{candidate.source.display_ref} <-> {candidate.target.display_ref}"


def _candidate_line(candidate: SpatialCandidate) -> str:
    arrow = "->" if candidate.directed else "<->"
    return (
        f"{candidate.predicate_display}: "
        f"{candidate.source.description} {arrow} {candidate.target.description}"
    )


def _edge_label(candidate: SpatialCandidate) -> str:
    arrow = "->" if candidate.directed else "<->"
    return f"{candidate.source_attribute} {arrow} {candidate.target_attribute}"


def _serialize_edge(
    primary: SpatialCandidate,
    candidates: list[SpatialCandidate],
) -> dict[str, object]:
    supported_predicates = []
    for predicate in [candidate.predicate for candidate in candidates]:
        if predicate not in supported_predicates:
            supported_predicates.append(predicate)

    detail_lines = [
        f"Predicate: {primary.predicate_display}",
        (
            f"Direction: {primary.source_table} -> {primary.target_table}"
            if primary.directed
            else "Direction: undirected"
        ),
        f"Source geometry: {primary.source.description}",
        f"Target geometry: {primary.target.description}",
    ]

    if len(supported_predicates) > 1:
        display_values = ", ".join(
            PREDICATE_DISPLAY.get(predicate, predicate.title())
            for predicate in supported_predicates
        )
        detail_lines.append(f"Supported predicates: {display_values}")

    if len(candidates) > 1:
        detail_lines.append("Compatible geometry pairs:")
        detail_lines.extend(_candidate_line(candidate) for candidate in candidates)

    return {
        "source": primary.source_table,
        "target": primary.target_table,
        "label": _edge_label(primary),
        "type": "spatial",
        "predicate": primary.predicate,
        "predicateDisplay": primary.predicate_display,
        "supportedPredicates": supported_predicates,
        "directed": primary.directed,
        "sourceAttribute": primary.source_attribute,
        "targetAttribute": primary.target_attribute,
        "sourceGeometryType": primary.source.geometry_type,
        "targetGeometryType": primary.target.geometry_type,
        "sourceOrigin": primary.source.origin,
        "targetOrigin": primary.target.origin,
        "popupTitle": "Spatial join",
        "popupLines": [
            primary.source.display_ref,
            primary.target.display_ref,
        ],
        "detailLines": detail_lines,
        "directionDisplay": _direction_display(primary),
    }


def discover_spatial_join_edges(
    chosen_lake: str | None,
    *,
    in_memory_tables: Mapping[str, pd.DataFrame] | None = None,
    manual_geo_details: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
    incident_tables: set[str] | None = None,
) -> list[dict[str, object]]:
    geometry_map = discover_table_geometries(
        chosen_lake=chosen_lake,
        in_memory_tables=in_memory_tables,
        manual_geo_details=manual_geo_details,
    )

    edges: list[dict[str, object]] = []
    for table_a, table_b in combinations(sorted(geometry_map.keys()), 2):
        if incident_tables and table_a not in incident_tables and table_b not in incident_tables:
            continue

        candidates: list[SpatialCandidate] = []
        for left_attr in geometry_map.get(table_a, []):
            for right_attr in geometry_map.get(table_b, []):
                candidate = _candidate_for_pair(left_attr, right_attr)
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            continue

        candidates.sort(
            key=lambda candidate: (
                -_candidate_sort_key(candidate)[0],
                -_candidate_sort_key(candidate)[1],
                -_candidate_sort_key(candidate)[2],
                _candidate_sort_key(candidate)[3],
                _candidate_sort_key(candidate)[4],
            )
        )
        primary = candidates[0]
        edges.append(_serialize_edge(primary, candidates))

    edges.sort(key=lambda edge: (str(edge["source"]), str(edge["target"]), str(edge["label"])))
    return edges


def refresh_incident_spatial_join_edges(
    existing_edges: list[dict[str, object]] | None,
    chosen_lake: str | None,
    *,
    affected_tables: set[str],
    in_memory_tables: Mapping[str, pd.DataFrame] | None = None,
    manual_geo_details: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> list[dict[str, object]]:
    if not affected_tables:
        return list(existing_edges or [])

    refreshed = discover_spatial_join_edges(
        chosen_lake=chosen_lake,
        in_memory_tables=in_memory_tables,
        manual_geo_details=manual_geo_details,
        incident_tables=affected_tables,
    )
    existing = [
        edge
        for edge in (existing_edges or [])
        if edge.get("source") not in affected_tables and edge.get("target") not in affected_tables
    ]
    merged = existing + refreshed
    merged.sort(key=lambda edge: (str(edge["source"]), str(edge["target"]), str(edge["label"])))
    return merged
