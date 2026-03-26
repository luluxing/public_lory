from __future__ import annotations

import json
import random
import re
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

import compute_associativity
import ddg_spatial_joins
import geo_augmentation
import geocoding
import graph_joins
import pandas_helper
from sections.join_path_renderer import build_table_metadata, render_join_path_card

DDG_COMPONENT = components.declare_component(
    "ddg_graph_component",
    path=str(Path(__file__).resolve().parent.parent / "custom_components" / "ddg_graph_component"),
)

# Node fills (distinct from edge greens/oranges/blues)
NODE_FILL_WITH_GEO = "#0d47a1"
NODE_FILL_WITHOUT_GEO = "#ffffff"
NODE_FILL_AUGMENTED_GEO = "#90caf9"
NODE_STROKE_DEFAULT = "#ffffff"
NODE_STROKE_LIGHT = "#94a3b8"


def _table_node_fill(
    table_name: str,
    df: pd.DataFrame,
    native_geo_tables: set[str],
    augmented_tables: set[str],
    geo_augmentation_enabled: bool,
) -> str:
    # Session-manual geo augmentation stays light blue until the lake resets.
    if geo_augmentation_enabled and table_name in augmented_tables:
        return NODE_FILL_AUGMENTED_GEO
    if table_name in native_geo_tables:
        return NODE_FILL_WITH_GEO
    return NODE_FILL_WITHOUT_GEO


def _table_node_stroke(node_fill: str) -> str:
    if node_fill == NODE_FILL_WITHOUT_GEO:
        return NODE_STROKE_LIGHT
    return NODE_STROKE_DEFAULT


def _table_label_fill(node_fill: str) -> str:
    return "#111111"


def _geo_augmentation_state_for_lake(
    data_lake: dict[str, pd.DataFrame],
    *,
    manual_geo_details: dict[str, dict[str, dict[str, object]]] | None = None,
):
    if manual_geo_details is None:
        manual_geo_details = _manual_geo_augmentation_details()
    geometry_map = ddg_spatial_joins.discover_table_geometries(
        chosen_lake=st.session_state.get("chosen_lake"),
        in_memory_tables=data_lake,
        manual_geo_details=manual_geo_details,
    )

    native_geo_tables = {
        table_name
        for table_name, attrs in geometry_map.items()
        if any(not attr.is_augmented for attr in attrs)
    }
    augmented_tables = {
        table_name
        for table_name, attr_map in manual_geo_details.items()
        if attr_map
    }
    geo_columns_map: dict[str, set[str]] = {
        table_name: {attr.attribute_name for attr in attrs}
        for table_name, attrs in geometry_map.items()
    }
    for table_name, attr_map in manual_geo_details.items():
        geo_columns_map.setdefault(table_name, set()).update(
            str(attribute_name)
            for attribute_name in attr_map
        )
    for table_name in data_lake:
        geo_columns_map.setdefault(table_name, set())

    return native_geo_tables, augmented_tables, geo_columns_map, geometry_map


def _ddg_geo_augmentation_info(
    data_lake: dict[str, pd.DataFrame],
    geo_augmentation_enabled: bool | None = None,
):
    """
    DDG geo information uses authoritative table copies (temp_tables first, then the
    selected lake) plus any in-session manual geo augmentation metadata.
    """
    del geo_augmentation_enabled
    manual_geo_details = _manual_geo_augmentation_details()
    native_geo_tables, augmented_tables, virtual_geo_columns_map, _ = (
        _geo_augmentation_state_for_lake(
            data_lake,
            manual_geo_details=manual_geo_details,
        )
    )
    return native_geo_tables, augmented_tables, virtual_geo_columns_map


def _lake_table_names_with_geo(data_lake: dict) -> list:
    native_geo_tables, augmented_tables, _ = _ddg_geo_augmentation_info(data_lake)
    return sorted(native_geo_tables.union(augmented_tables))


def _lake_table_names_without_geo(data_lake: dict) -> list:
    native_geo_tables, augmented_tables, _ = _ddg_geo_augmentation_info(data_lake)
    with_geo = native_geo_tables.union(augmented_tables)
    return sorted(name for name in data_lake if name not in with_geo)


def _ddg_visible_table_names(data_lake: dict) -> set:
    show_geo = st.session_state.get("ddg_show_geo_tables", False)
    show_nogeo = st.session_state.get("ddg_show_nogeo_tables", False)
    if not show_geo and not show_nogeo:
        return set()
    names = set()
    if show_geo:
        names.update(_lake_table_names_with_geo(data_lake))
    if show_nogeo:
        names.update(_lake_table_names_without_geo(data_lake))
    return names


def _persist_geo_augmented_csv(table_name: str, df: pd.DataFrame) -> Path:
    out_path = pandas_helper.temp_table_path(table_name)
    df.to_csv(out_path, index=False)
    return out_path


def _apply_physical_geo_augmentation_from_discovery() -> None:
    """
    For tables that are only geo-connected via relational paths, left-join along
    the shortest path to a native geo table, append spatial columns, save CSVs under
    temp_tables/, and refresh session data_lake entries.
    """
    lake = st.session_state.data_lake
    chosen = st.session_state.get("chosen_lake")
    if not chosen or chosen == "<no datalakes found>":
        st.warning("Select a data lake first.")
        return

    native_geo_tables = {
        n for n, df in lake.items() if graph_joins._find_spatial_columns(df)
    }
    _, augmented_tables, _ = _ddg_geo_augmentation_info(
        lake, geo_augmentation_enabled=True
    )
    to_process = sorted(augmented_tables)
    if not to_process:
        st.info(
            "No tables are reachable from geo-enabled tables via shared column names; "
            "nothing to augment."
        )
        return

    working = dict(lake)
    written: list[tuple[str, str]] = []

    for t in to_process:
        path = graph_joins.bfs_shortest_path_to_geo_table(
            t, working, native_geo_tables
        )
        if not path or len(path) < 2:
            continue
        aug_df = graph_joins.physical_geo_augment_along_path(working, path)
        if aug_df is None or not graph_joins._find_spatial_columns(aug_df):
            continue
        out_path = _persist_geo_augmented_csv(t, aug_df)
        working[t] = aug_df
        st.session_state.data_lake[t] = aug_df
        written.append((t, str(out_path)))

    if written:
        st.session_state.discovery_geo_augmentation = True
        st.success(
            "Geo-augmented "
            + ", ".join(name for name, _ in written)
            + ". Saved under temp_tables/; the app now uses these versions."
        )
    else:
        st.warning(
            "Geo-augmentation did not produce new spatial columns for any table. "
            "Check relational links to geo tables."
        )


def _full_ddg_spatial_edges(
    lake: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    return ddg_spatial_joins.discover_spatial_join_edges(
        chosen_lake=st.session_state.get("chosen_lake"),
        in_memory_tables=lake,
        manual_geo_details=_manual_geo_augmentation_details(),
    )


def _refresh_ddg_spatial_edges_for_tables(
    lake: dict[str, pd.DataFrame],
    affected_tables: set[str],
) -> None:
    st.session_state.ddg_spatial_edges = ddg_spatial_joins.refresh_incident_spatial_join_edges(
        st.session_state.get("ddg_spatial_edges", []),
        chosen_lake=st.session_state.get("chosen_lake"),
        affected_tables=affected_tables,
        in_memory_tables=lake,
        manual_geo_details=_manual_geo_augmentation_details(),
    )


def _data_discovery_legend_html() -> str:
    return f"""
<div style="display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;margin:0.15rem 0 0.85rem 0;font-size:13px;color:#333;">
  <span style="font-weight:600;color:#111;">Tables</span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <span style="display:inline-block;width:14px;height:14px;background:{NODE_FILL_WITH_GEO};border-radius:2px;border:1px solid {NODE_STROKE_LIGHT};"></span>
    Tables with Geo
  </span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <span style="display:inline-block;width:14px;height:14px;background:{NODE_FILL_WITHOUT_GEO};border-radius:2px;border:1px solid {NODE_STROKE_LIGHT};"></span>
    Tables without Geo
  </span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <span style="display:inline-block;width:14px;height:14px;background:{NODE_FILL_AUGMENTED_GEO};border-radius:2px;border:1px solid {NODE_STROKE_LIGHT};"></span>
    Tables augmented with Geo
  </span>
  <span style="font-weight:600;color:#111;margin-left:8px;">Joins</span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <span style="display:inline-block;width:28px;height:0;border-bottom:3px solid #4caf50;vertical-align:middle;"></span>
    Equi-join
  </span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <span style="display:inline-block;width:28px;height:0;border-bottom:3px solid #ff9800;vertical-align:middle;"></span>
    Semantic join
  </span>
  <span style="display:inline-flex;align-items:center;gap:6px;">
    <svg width="32" height="8" style="vertical-align:middle" aria-hidden="true">
      <line x1="0" y1="4" x2="32" y2="4" stroke="#1e88e5" stroke-width="3" stroke-dasharray="6 4"/>
    </svg>
    Spatial join
  </span>
</div>
"""


def _build_graphviz(nodes, rel_edges, spatial_edges) -> str:
    lines = ["graph G {", "  node [shape=box];", "  layout=neato;"]
    for node in sorted(nodes):
        lines.append(f'  "{node}";')

    for left, right, label in rel_edges:
        label_txt = f' [label="{label}"]' if label else ""
        lines.append(f'  "{left}" -- "{right}"{label_txt};')

    for left, right, label in spatial_edges:
        label_txt = (
            f' [label="{label}", color="#1e88e5", fontcolor="#1e88e5", style="dashed"]'
        )
        lines.append(f'  "{left}" -- "{right}"{label_txt};')

    lines.append("}")
    return "\n".join(lines)


def _pair_join_attributes(label: str, table_left: str, table_right: str) -> list[tuple[str, str]]:
    if not label:
        return [("", "")]

    label_clean = label.replace("…", "").strip()
    parts = [part.strip() for part in label_clean.split(",") if part.strip()]
    left_attrs = []
    right_attrs = []
    for part in parts:
        if "≈" in part:
            left_part, right_part = [piece.strip() for piece in part.split("≈", 1)]
            left_attrs.append(left_part)
            right_attrs.append(right_part)
        elif part.startswith(f"{table_left}."):
            left_attrs.append(part.split(".", 1)[1])
        elif part.startswith(f"{table_right}."):
            right_attrs.append(part.split(".", 1)[1])
        else:
            left_attrs.append(part)
            right_attrs.append(part)

    max_len = max(len(left_attrs), len(right_attrs), 1)
    pairs = []
    for i in range(max_len):
        left_attr = left_attrs[i] if i < len(left_attrs) else ""
        right_attr = right_attrs[i] if i < len(right_attrs) else ""
        pairs.append((left_attr, right_attr))
    return pairs


def _edge_popup_lines(label: str, table_left: str, table_right: str) -> list[str]:
    lines = []
    for left_attr, right_attr in _pair_join_attributes(label, table_left, table_right):
        if left_attr and right_attr:
            lines.append(f"{table_left}.{left_attr}, {table_right}.{right_attr}")
        elif left_attr:
            lines.append(f"{table_left}.{left_attr}")
        elif right_attr:
            lines.append(f"{table_right}.{right_attr}")
    return lines or ["No join columns available."]


def _node_geo_associativity_lines(
    df: pd.DataFrame,
    geo_columns: set[str] | list[str],
    geo_association_details: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    return [
        f"{geo_col} (geo): Top associativity score ({attribute}, {score:.3f})"
        for geo_col, attribute, score in compute_associativity.top_geo_associativity(
            df,
            geo_columns,
            geo_association_details=geo_association_details,
        )
    ]


def _node_attribute_lines(
    df: pd.DataFrame,
    geo_columns: set[str] | list[str],
) -> list[str]:
    geo_columns_set = set(geo_columns)
    attributes = [
        f"{column} (geo)" if str(column) in geo_columns_set else str(column)
        for column in df.columns
    ]
    known_columns = {str(column) for column in df.columns}
    attributes.extend(
        f"{column} (geo)"
        for column in sorted(geo_columns_set - known_columns)
    )
    return attributes


def _manual_geo_augmentation_map() -> dict[str, set[str]]:
    return {
        str(table_name): {str(column) for column in columns}
        for table_name, columns in st.session_state.get(
            "ddg_manual_geo_augmentation_map",
            {},
        ).items()
    }


def _store_manual_geo_augmentation_map(
    manual_map: dict[str, set[str]],
) -> None:
    st.session_state.ddg_manual_geo_augmentation_map = {
        table_name: sorted(columns)
        for table_name, columns in manual_map.items()
        if columns
    }


def _manual_geo_augmentation_details() -> dict[str, dict[str, dict[str, object]]]:
    normalized: dict[str, dict[str, dict[str, object]]] = {}
    for table_name, attr_map in st.session_state.get(
        "ddg_manual_geo_augmentation_details",
        {},
    ).items():
        table_key = str(table_name)
        table_details = geocoding.normalize_geo_association_details(dict(attr_map))
        if table_details:
            normalized[table_key] = table_details
    return normalized


def _store_manual_geo_augmentation_details(
    details_map: dict[str, dict[str, dict[str, object]]],
) -> None:
    normalized: dict[str, dict[str, dict[str, object]]] = {}
    for table_name, attr_map in details_map.items():
        table_details = geocoding.normalize_geo_association_details(dict(attr_map))
        if table_details:
            normalized[str(table_name)] = table_details
    st.session_state.ddg_manual_geo_augmentation_details = normalized


def _merge_geo_association_detail(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    merged = dict(base)
    attribute_scores = dict(base.get("attribute_scores") or {})
    attribute_scores.update(dict(override.get("attribute_scores") or {}))
    if attribute_scores:
        merged["attribute_scores"] = attribute_scores

    origin_attributes = list(
        dict.fromkeys(
            [
                *list(base.get("origin_attributes") or []),
                *list(override.get("origin_attributes") or []),
            ]
        )
    )
    if origin_attributes:
        merged["origin_attributes"] = origin_attributes

    for key in (
        "geometry_type",
        "source_table",
        "source_origin",
        "source_attribute",
        "target_attribute",
    ):
        value = override.get(key)
        if value:
            merged[key] = value
        elif key not in merged and base.get(key):
            merged[key] = base.get(key)

    return geocoding.normalize_geo_association_details({"_": merged}).get("_", {})


def _table_geo_association_details(
    table_name: str,
    manual_geo_details: dict[str, dict[str, dict[str, object]]] | None = None,
) -> dict[str, dict[str, object]]:
    merged = geocoding.read_geo_association_details(table_name)
    manual_details = (manual_geo_details or {}).get(table_name, {})
    for geo_attr, detail in manual_details.items():
        merged[geo_attr] = _merge_geo_association_detail(
            merged.get(geo_attr, {}),
            detail,
        )
    return geocoding.normalize_geo_association_details(merged)


def _geo_association_details_map_for_lake(
    data_lake: dict[str, pd.DataFrame],
    manual_geo_details: dict[str, dict[str, dict[str, object]]] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    if manual_geo_details is None:
        manual_geo_details = _manual_geo_augmentation_details()
    return {
        table_name: _table_geo_association_details(
            table_name,
            manual_geo_details=manual_geo_details,
        )
        for table_name in data_lake
    }


def _directed_relation_geo_augmentation_candidates(
    *,
    source_table: str,
    target_table: str,
    left: str,
    right: str,
    label: str,
    lake: dict[str, pd.DataFrame],
    geo_columns_map: dict[str, set[str]],
    geo_association_details_map: dict[str, dict[str, dict[str, object]]],
) -> list[geo_augmentation.GeoAugmentationCandidate]:
    source_geo_columns = set(geo_columns_map.get(source_table, set()))
    if not source_geo_columns:
        return []

    source_df = lake.get(source_table, pd.DataFrame())
    target_df = lake.get(target_table, pd.DataFrame())
    if compute_associativity.infer_primary_key(target_df) is None:
        return []

    source_geo_association_details = geo_association_details_map.get(source_table, {})
    candidates: list[geo_augmentation.GeoAugmentationCandidate] = []
    for left_attr, right_attr in _pair_join_attributes(label, left, right):
        if source_table == left:
            source_attr, target_attr = left_attr, right_attr
        else:
            source_attr, target_attr = right_attr, left_attr
        if (
            not source_attr
            or not target_attr
            or source_attr in source_geo_columns
        ):
            continue
        candidates.extend(
            geo_augmentation.evaluate_geo_augmentation(
                source_table=source_table,
                source_df=source_df,
                source_attr=source_attr,
                target_table=target_table,
                target_df=target_df,
                target_attr=target_attr,
                geo_columns=source_geo_columns,
                source_geo_association_details=source_geo_association_details,
            )
        )
    return candidates


def _random_best_geo_candidate(
    candidates: list[geo_augmentation.GeoAugmentationCandidate],
) -> geo_augmentation.GeoAugmentationCandidate | None:
    if not candidates:
        return None
    best_score = max(candidate.target_score for candidate in candidates)
    best_candidates = [
        candidate
        for candidate in candidates
        if candidate.target_score == best_score
    ]
    return random.choice(best_candidates)


def _select_bulk_geo_candidates_for_target(
    target_df: pd.DataFrame,
    candidates: list[geo_augmentation.GeoAugmentationCandidate],
) -> list[geo_augmentation.GeoAugmentationCandidate]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible and candidate.geo_attr not in target_df.columns
    ]
    if not eligible:
        return []

    selected_by_join_attr: list[geo_augmentation.GeoAugmentationCandidate] = []
    join_groups: dict[str, list[geo_augmentation.GeoAugmentationCandidate]] = {}
    for candidate in eligible:
        join_groups.setdefault(candidate.target_attr, []).append(candidate)
    for target_attr in sorted(join_groups):
        winner = _random_best_geo_candidate(join_groups[target_attr])
        if winner is not None:
            selected_by_join_attr.append(winner)

    deduped_by_geo_attr: list[geo_augmentation.GeoAugmentationCandidate] = []
    geo_groups: dict[str, list[geo_augmentation.GeoAugmentationCandidate]] = {}
    for candidate in selected_by_join_attr:
        geo_groups.setdefault(candidate.geo_attr, []).append(candidate)
    for geo_attr in sorted(geo_groups):
        winner = _random_best_geo_candidate(geo_groups[geo_attr])
        if winner is not None:
            deduped_by_geo_attr.append(winner)

    return sorted(
        deduped_by_geo_attr,
        key=lambda candidate: (
            candidate.target_attr,
            candidate.geo_attr,
            candidate.source_table,
        ),
    )


def _merge_geo_attribute_into_dataframe(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    target_attr: str,
    source_attr: str,
    geo_attr: str,
) -> tuple[pd.DataFrame, bool]:
    if (
        not target_attr
        or not source_attr
        or not geo_attr
        or target_attr not in target_df.columns
        or source_attr not in source_df.columns
        or geo_attr not in source_df.columns
        or geo_attr in target_df.columns
    ):
        return target_df, False

    merge_slice = source_df[[source_attr, geo_attr]].copy()
    merge_slice["__ddg_geo_non_null__"] = merge_slice[geo_attr].notna().astype(int)
    merge_slice = (
        merge_slice.sort_values("__ddg_geo_non_null__", ascending=False)
        .drop_duplicates(subset=[source_attr])
        .drop(columns=["__ddg_geo_non_null__"])
    )
    if source_attr == target_attr:
        merged_df = target_df.merge(merge_slice, on=target_attr, how="left")
        return merged_df, geo_attr in merged_df.columns

    temp_source_attr = "__ddg_geo_source_join_key__"
    merge_slice = merge_slice.rename(columns={source_attr: temp_source_attr})
    merged_df = target_df.merge(
        merge_slice,
        left_on=target_attr,
        right_on=temp_source_attr,
        how="left",
    ).drop(columns=[temp_source_attr])
    return merged_df, geo_attr in merged_df.columns


def _apply_geo_candidates_to_table(
    *,
    target_table: str,
    target_df: pd.DataFrame,
    candidates: list[geo_augmentation.GeoAugmentationCandidate],
    lake: dict[str, pd.DataFrame],
    geometry_map: dict[str, list[ddg_spatial_joins.GeometryAttribute]],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], list[str]]:
    updated_df = target_df.copy()
    target_primary_key = compute_associativity.infer_primary_key(target_df) or ""
    manual_detail_updates: dict[str, dict[str, object]] = {}
    new_geo_columns: list[str] = []

    for candidate in candidates:
        updated_df, added = _merge_geo_attribute_into_dataframe(
            updated_df,
            lake.get(candidate.source_table, pd.DataFrame()),
            target_attr=candidate.target_attr,
            source_attr=candidate.source_attr,
            geo_attr=candidate.geo_attr,
        )
        if not added:
            continue

        source_attr_map = {
            attr.attribute_name: attr
            for attr in geometry_map.get(candidate.source_table, [])
        }
        source_geo = source_attr_map.get(candidate.geo_attr)
        attribute_scores = {
            candidate.target_attr: round(candidate.target_join_score, 3),
        }
        if target_primary_key:
            attribute_scores[target_primary_key] = round(candidate.target_score, 3)
        manual_detail_updates[candidate.geo_attr] = {
            "geometry_type": (
                source_geo.geometry_type
                if source_geo is not None
                else ddg_spatial_joins.infer_geometry_type_from_name(candidate.geo_attr)
            ),
            "source_table": candidate.source_table,
            "source_origin": source_geo.origin if source_geo is not None else "",
            "source_attribute": candidate.source_attr,
            "target_attribute": candidate.target_attr,
            "origin_attributes": [],
            "attribute_scores": attribute_scores,
        }
        new_geo_columns.append(candidate.geo_attr)

    return updated_df, manual_detail_updates, new_geo_columns


def _apply_bulk_geo_augmentation_from_discovery() -> None:
    lake = dict(st.session_state.data_lake)
    if not lake:
        st.warning("Load a data lake first.")
        return

    manual_map = _manual_geo_augmentation_map()
    manual_geo_details = _manual_geo_augmentation_details()
    written: list[tuple[str, str]] = []
    affected_tables: set[str] = set()

    while True:
        _, _, geo_columns_map, geometry_map = _geo_augmentation_state_for_lake(
            lake,
            manual_geo_details=manual_geo_details,
        )
        geo_association_details_map = _geo_association_details_map_for_lake(
            lake,
            manual_geo_details=manual_geo_details,
        )

        candidates_by_target: dict[
            str,
            list[geo_augmentation.GeoAugmentationCandidate],
        ] = {}
        for left, right, label in graph_joins.find_relational_joins(lake):
            left_geo_columns = set(geo_columns_map.get(left, set()))
            right_geo_columns = set(geo_columns_map.get(right, set()))
            if bool(left_geo_columns) == bool(right_geo_columns):
                continue

            if left_geo_columns and not right_geo_columns:
                edge_candidates = _directed_relation_geo_augmentation_candidates(
                    source_table=left,
                    target_table=right,
                    left=left,
                    right=right,
                    label=label,
                    lake=lake,
                    geo_columns_map=geo_columns_map,
                    geo_association_details_map=geo_association_details_map,
                )
            else:
                edge_candidates = _directed_relation_geo_augmentation_candidates(
                    source_table=right,
                    target_table=left,
                    left=left,
                    right=right,
                    label=label,
                    lake=lake,
                    geo_columns_map=geo_columns_map,
                    geo_association_details_map=geo_association_details_map,
                )

            for candidate in edge_candidates:
                candidates_by_target.setdefault(candidate.target_table, []).append(
                    candidate
                )

        planned_updates: list[tuple[str, list[geo_augmentation.GeoAugmentationCandidate]]] = []
        for target_table in sorted(candidates_by_target):
            selected_candidates = _select_bulk_geo_candidates_for_target(
                lake.get(target_table, pd.DataFrame()),
                candidates_by_target[target_table],
            )
            if selected_candidates:
                planned_updates.append((target_table, selected_candidates))

        if not planned_updates:
            break

        for target_table, selected_candidates in planned_updates:
            updated_df, detail_updates, new_geo_columns = _apply_geo_candidates_to_table(
                target_table=target_table,
                target_df=lake.get(target_table, pd.DataFrame()),
                candidates=selected_candidates,
                lake=lake,
                geometry_map=geometry_map,
            )
            if not new_geo_columns:
                continue

            lake[target_table] = updated_df
            st.session_state.data_lake[target_table] = updated_df
            if target_table in st.session_state.get("uploaded_csv_files", {}):
                st.session_state.uploaded_csv_files[target_table]["dataframe"] = updated_df

            manual_map.setdefault(target_table, set()).update(new_geo_columns)
            target_details = manual_geo_details.setdefault(target_table, {})
            persisted_details = geocoding.read_geo_association_details(target_table)
            for geo_attr, detail in detail_updates.items():
                target_details[geo_attr] = _merge_geo_association_detail(
                    target_details.get(geo_attr, {}),
                    detail,
                )
                persisted_details[geo_attr] = _merge_geo_association_detail(
                    persisted_details.get(geo_attr, {}),
                    detail,
                )
            geocoding.write_geo_association_details(target_table, persisted_details)

            out_path = _persist_geo_augmented_csv(target_table, updated_df)
            written.append((target_table, str(out_path)))
            affected_tables.add(target_table)

    _store_manual_geo_augmentation_map(manual_map)
    _store_manual_geo_augmentation_details(manual_geo_details)

    if not written:
        st.info("No additional tables are geo-augmentable right now.")
        return

    st.session_state.discovery_geo_augmentation = True
    st.session_state.ddg_show_geo_tables = True
    st.session_state.ddg_spatial_refresh_tables = sorted(
        set(st.session_state.get("ddg_spatial_refresh_tables", []))
        | affected_tables
    )
    # st.success(
    #     "Applied geo-augmentation to "
    #     + ", ".join(name for name, _ in written)
    #     + ". Saved under temp_tables/; the app now uses these versions."
    # )


def _process_ddg_geo_augmentation_request(
    event_payload: object,
    lake: dict[str, pd.DataFrame],
) -> bool:
    if not isinstance(event_payload, dict):
        return False

    if str(event_payload.get("type") or "") != "geo_augmentation_request":
        return False

    payload = event_payload.get("payload")
    if not isinstance(payload, dict):
        st.session_state.ddg_geo_aug_message = {
            "kind": "warning",
            "text": "Ignoring an invalid geo augmentation request.",
        }
        return True

    nonce = str(payload.get("nonce") or "")
    if nonce and st.session_state.get("ddg_last_geo_aug_nonce") == nonce:
        return False
    if nonce:
        st.session_state.ddg_last_geo_aug_nonce = nonce

    source_table = str(payload.get("sourceTable") or "")
    target_table = str(payload.get("targetTable") or "")
    source_attr = str(payload.get("sourceAttr") or "")
    target_attr = str(payload.get("targetAttr") or "")
    target_primary_key = str(payload.get("targetPrimaryKey") or "")
    geo_columns = [str(column) for column in payload.get("geoColumns") or []]
    threshold = float(
        payload.get("threshold", geo_augmentation.GEO_AUGMENTATION_THRESHOLD)
    )
    target_score = float(payload.get("targetScore", 0.0))
    target_join_score = float(payload.get("targetJoinScore", 0.0))

    if (
        source_table not in lake
        or target_table not in lake
        or not source_attr
        or not target_attr
        or not geo_columns
    ):
        st.session_state.ddg_geo_aug_message = {
            "kind": "warning",
            "text": "Ignoring an incomplete geo augmentation request.",
        }
        return True

    if target_score < threshold:
        st.session_state.ddg_geo_aug_message = {
            "kind": "warning",
            "text": (
                f"Geo augmentation blocked for {target_table}: "
                f"y={target_score:.3f} is below {threshold:.3f}."
            ),
        }
        return True

    manual_map = _manual_geo_augmentation_map()
    manual_map.setdefault(target_table, set()).update(geo_columns)
    _store_manual_geo_augmentation_map(manual_map)

    geometry_map = ddg_spatial_joins.discover_table_geometries(
        chosen_lake=st.session_state.get("chosen_lake"),
        in_memory_tables=lake,
        manual_geo_details=_manual_geo_augmentation_details(),
    )
    source_attr_map = {
        attr.attribute_name: attr
        for attr in geometry_map.get(source_table, [])
    }
    details_map = _manual_geo_augmentation_details()
    target_details = details_map.setdefault(target_table, {})
    for geo_column in geo_columns:
        source_geo = source_attr_map.get(geo_column)
        geometry_type = (
            source_geo.geometry_type
            if source_geo is not None
            else ddg_spatial_joins.infer_geometry_type_from_name(geo_column)
        )
        if geometry_type not in {"point", "polygon"}:
            continue
        attribute_scores: dict[str, float] = {
            target_attr: round(target_join_score, 3)
        }
        if target_primary_key:
            attribute_scores[target_primary_key] = round(target_score, 3)
        target_details[geo_column] = {
            "geometry_type": geometry_type,
            "source_table": source_table,
            "source_origin": source_geo.origin if source_geo is not None else "",
            "source_attribute": source_attr,
            "target_attribute": target_attr,
            "origin_attributes": [],
            "attribute_scores": attribute_scores,
        }
    _store_manual_geo_augmentation_details(details_map)
    st.session_state.ddg_spatial_refresh_tables = sorted(
        set(st.session_state.get("ddg_spatial_refresh_tables", [])) | {target_table}
    )
    st.session_state.ddg_geo_aug_message = {
        "kind": "success",
        "text": (
            f"Geo augmented {target_table} from {source_table} via "
            f"{source_table}.{source_attr} = {target_table}.{target_attr} "
            f"(y={target_score:.3f})."
        ),
    }
    return True


def _relation_geo_augmentation_data(
    left: str,
    right: str,
    label: str,
    lake: dict[str, pd.DataFrame],
    geo_columns_map: dict[str, set[str]],
    geo_association_details_map: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object] | None:
    left_geo_columns = set(geo_columns_map.get(left, set()))
    right_geo_columns = set(geo_columns_map.get(right, set()))

    if bool(left_geo_columns) == bool(right_geo_columns):
        if left_geo_columns:
            return {
                "status": "Both tables already have geo, so augmentation is unnecessary.",
                "canAugment": False,
            }
        return {
            "status": "Neither table has geo, so augmentation is unavailable.",
            "canAugment": False,
        }

    if left_geo_columns:
        source_table, target_table = left, right
        source_geo_columns = left_geo_columns
    else:
        source_table, target_table = right, left
        source_geo_columns = right_geo_columns

    source_df = lake.get(source_table, pd.DataFrame())
    target_df = lake.get(target_table, pd.DataFrame())
    target_primary_key = compute_associativity.infer_primary_key(target_df)
    if not target_primary_key:
        return {
            "status": (
                f"Geo augmentation is unavailable because {target_table} "
                "has no valid primary key."
            ),
            "canAugment": False,
        }

    source_geo_association_details = geo_association_details_map.get(source_table, {})
    candidates = []
    for left_attr, right_attr in _pair_join_attributes(label, left, right):
        if source_table == left:
            source_attr, target_attr = left_attr, right_attr
        else:
            source_attr, target_attr = right_attr, left_attr
        if (
            not source_attr
            or not target_attr
            or source_attr in source_geo_columns
        ):
            continue
        candidates.extend(
            geo_augmentation.evaluate_geo_augmentation(
                source_table=source_table,
                source_df=source_df,
                source_attr=source_attr,
                target_table=target_table,
                target_df=target_df,
                target_attr=target_attr,
                geo_columns=source_geo_columns,
                source_geo_association_details=source_geo_association_details,
            )
        )

    if not candidates:
        return {
            "status": (
                "Geo augmentation is unavailable because the equi-join does not "
                "connect non-geo attributes."
            ),
            "canAugment": False,
        }

    best_candidate = candidates[0]
    source_geo_label = f"{best_candidate.source_table}.{best_candidate.geo_attr}"
    source_primary_key = compute_associativity.infer_primary_key(source_df)
    source_primary_key_score = (
        compute_associativity.geo_associativity_score(
            source_df,
            source_primary_key,
            best_candidate.geo_attr,
            geo_association_details=source_geo_association_details,
        )
        if source_primary_key
        else None
    )
    assoc_lines = [
        line
        for line in [
            (
                f"{best_candidate.source_table}.{source_primary_key} "
                f"with {source_geo_label} (geo) = {source_primary_key_score:.3f}"
                if source_primary_key and source_primary_key_score is not None
                else None
            ),
            (
                f"{best_candidate.source_table}.{best_candidate.source_attr} "
                f"with {source_geo_label} (geo) = {best_candidate.source_score:.3f}"
                if best_candidate.source_attr != source_primary_key
                else None
            ),
            (
                f"{best_candidate.target_table}.{best_candidate.target_attr} "
                f"with {source_geo_label} (geo) = "
                f"{best_candidate.target_join_score:.3f}"
            ),
        ]
        if line
    ]
    if (
        target_primary_key
        and target_primary_key != best_candidate.target_attr
        and target_primary_key not in source_geo_columns
    ):
        assoc_lines.append(
            (
                f"{best_candidate.target_table}.{target_primary_key} "
                f"with {source_geo_label} (geo) = {best_candidate.target_score:.3f}"
            )
        )
    if best_candidate.dropped_rows > 0:
        assoc_lines.append(
            (
                "Dropped "
                f"{best_candidate.dropped_rows} row(s) with null values before "
                f"computing H({best_candidate.target_table}.{best_candidate.target_attr}) / "
                f"H({best_candidate.target_table}.{target_primary_key})."
            )
        )
    status = (
        f"Geo can be propagated to {target_table}."
        if best_candidate.eligible
        else (
            f"Geo cannot be propagated to {target_table}: "
            f"y={best_candidate.target_score:.3f} is below "
            f"{best_candidate.threshold:.3f}."
        )
    )
    return {
        "assocLines": assoc_lines,
        "geoColumnsLine": (
            "Geo columns to propagate: "
            + ", ".join(
                f"{best_candidate.source_table}.{column}"
                for column in sorted(source_geo_columns)
            )
        ),
        "status": status,
        "canAugment": best_candidate.eligible,
        "buttonLabel": "Apply",
        "requestPayload": {
            "sourceTable": best_candidate.source_table,
            "sourceAttr": best_candidate.source_attr,
            "targetTable": best_candidate.target_table,
            "targetAttr": best_candidate.target_attr,
            "targetPrimaryKey": target_primary_key,
            "geoAttr": best_candidate.geo_attr,
            "geoColumns": sorted(source_geo_columns),
            "sourceScore": best_candidate.source_score,
            "targetJoinScore": best_candidate.target_join_score,
            "targetScore": best_candidate.target_score,
            "threshold": best_candidate.threshold,
            "eligible": best_candidate.eligible,
        },
    }


def _d3_discovery_graph_html(d3_data: dict) -> str:
    """Force-directed join graph for embedding via components.html (one iframe per call)."""
    return f"""
    <div id="graph" style="position:relative;width:100%;height:520px;border:1px solid #e0e0e0;border-radius:8px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script>
    const data = {json.dumps(d3_data)};
    const width = document.getElementById('graph').clientWidth;
    const height = 500;

    const padding = 40;
    data.nodes.forEach((d) => {{
      d.x = padding + Math.random() * (width - 2 * padding);
      d.y = padding + Math.random() * (height - 2 * padding);
      d.wasDragged = false;
    }});

    const svg = d3.select('#graph')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const color = d3.scaleOrdinal()
      .domain(['relation', 'semantic', 'spatial'])
      .range(['#4caf50', '#ff9800', '#1e88e5']);

    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'spatial-arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 9)
      .attr('refY', 0)
      .attr('markerWidth', 7)
      .attr('markerHeight', 7)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#1e88e5');

    const typeOrder = {{ relation: 0, semantic: 1, spatial: 2 }};
    const pairGroups = new Map();
    data.links.forEach((linkData) => {{
      const sourceId = typeof linkData.source === 'object' ? linkData.source.id : linkData.source;
      const targetId = typeof linkData.target === 'object' ? linkData.target.id : linkData.target;
      const pairKey = [sourceId, targetId].sort().join('||');
      if (!pairGroups.has(pairKey)) {{
        pairGroups.set(pairKey, []);
      }}
      pairGroups.get(pairKey).push(linkData);
    }});
    pairGroups.forEach((group) => {{
      group.sort((left, right) => (typeOrder[left.type] ?? 99) - (typeOrder[right.type] ?? 99));
      if (group.length === 1) {{
        group[0].curveOffset = 0;
        return;
      }}
      if (group.length === 2) {{
        const spatialEdge = group.find(edge => edge.type === 'spatial');
        if (spatialEdge) {{
          group.forEach(edge => {{
            edge.curveOffset = edge === spatialEdge ? 28 : 0;
          }});
          return;
        }}
        group[0].curveOffset = -18;
        group[1].curveOffset = 18;
        return;
      }}
      const offsets = [-24, 0, 24];
      group.forEach((edge, index) => {{
        edge.curveOffset = offsets[index] ?? ((index - ((group.length - 1) / 2)) * 24);
      }});
    }});

    const linkHitArea = svg.append('g')
      .attr('stroke', '#000')
      .attr('stroke-opacity', 0)
      .attr('stroke-width', 16)
      .attr('fill', 'none')
      .selectAll('path')
      .data(data.links)
      .join('path')
      .style('pointer-events', 'stroke')
      .style('cursor', 'pointer');

    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.7)
      .attr('fill', 'none')
      .selectAll('path')
      .data(data.links)
      .join('path')
      .attr('stroke-width', 2)
      .attr('stroke', d => color(d.type))
      .attr('stroke-dasharray', d => d.type === 'spatial' ? '6,4' : '0')
      .attr('marker-end', d => d.type === 'spatial' && d.directed ? 'url(#spatial-arrowhead)' : null)
      .style('pointer-events', 'none');

    const hasLinks = data.links.length > 0;
    const linkDistance = hasLinks ? 200 : 120;
    const chargeStrength = hasLinks ? -300 : -140;

    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .on('tick', ticked);

    const node = svg.append('g')
      .attr('stroke-width', 1.5)
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', 26)
      .attr('fill', d => d.nodeFill || '#9e9e9e')
      .attr('stroke', d => d.nodeStroke || '#fff')
      .style('cursor', 'pointer')
      .call(drag(simulation));

    const label = svg.append('g')
      .style('font', '16px sans-serif')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.labelFill || '#111')
      .style('pointer-events', 'none')
      .text(d => d.id);

    const edgePopup = d3.select('#graph').append('div')
      .style('position', 'absolute')
      .style('display', 'none')
      .style('min-width', '220px')
      .style('max-width', '320px')
      .style('padding', '12px 14px 10px 14px')
      .style('background', '#fff')
      .style('color', '#111')
      .style('border', '1px solid #cbd5e1')
      .style('border-radius', '4px')
      .style('box-shadow', '0 10px 30px rgba(15, 23, 42, 0.18)')
      .style('font', '12px sans-serif')
      .style('z-index', 20)
      .style('pointer-events', 'auto')
      .on('click', event => event.stopPropagation());

    const edgePopupClose = edgePopup.append('button')
      .attr('type', 'button')
      .attr('aria-label', 'Close edge details')
      .style('position', 'absolute')
      .style('top', '6px')
      .style('right', '8px')
      .style('border', 'none')
      .style('background', 'transparent')
      .style('color', '#475569')
      .style('cursor', 'pointer')
      .style('font-size', '16px')
      .style('line-height', '1')
      .text('×');

    const edgePopupBody = edgePopup.append('div')
      .style('padding-right', '16px')
      .style('max-height', '280px')
      .style('overflow-y', 'auto');

    let selectedEdgeKey = null;

    function edgeKey(d) {{
      const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
      const targetId = typeof d.target === 'object' ? d.target.id : d.target;
      return [sourceId, targetId].sort().join('||') + '||' + (d.type || '') + '||' + (d.label || '');
    }}

    function updateLinkStyles() {{
      link
        .attr('stroke-width', d => edgeKey(d) === selectedEdgeKey ? 4 : 2)
        .attr('stroke-opacity', d => edgeKey(d) === selectedEdgeKey ? 1 : 0.7);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function hideEdgePopup() {{
      selectedEdgeKey = null;
      edgePopup.style('display', 'none');
      updateLinkStyles();
    }}

    function getNodeId(nodeOrId) {{
      return typeof nodeOrId === 'object' && nodeOrId !== null
        ? nodeOrId.id
        : nodeOrId;
    }}

    function getNodeData(nodeOrId) {{
      const nodeId = getNodeId(nodeOrId);
      return data.nodes.find(nodeData => nodeData.id === nodeId) || null;
    }}

    function linkPath(d) {{
      const sx = d.source.x || 0;
      const sy = d.source.y || 0;
      const tx = d.target.x || 0;
      const ty = d.target.y || 0;
      const dx = tx - sx;
      const dy = ty - sy;
      const length = Math.max(Math.sqrt((dx * dx) + (dy * dy)), 1);
      const ux = dx / length;
      const uy = dy / length;
      const startPadding = 26;
      const endPadding = d.type === 'spatial' && d.directed ? 30 : 26;
      const startX = sx + (ux * startPadding);
      const startY = sy + (uy * startPadding);
      const endX = tx - (ux * endPadding);
      const endY = ty - (uy * endPadding);
      const curveOffset = Number(d.curveOffset || 0);
      if (!curveOffset) {{
        return `M${{startX}},${{startY}}L${{endX}},${{endY}}`;
      }}
      const nx = -uy;
      const ny = ux;
      const controlX = ((startX + endX) / 2) + (nx * curveOffset);
      const controlY = ((startY + endY) / 2) + (ny * curveOffset);
      return `M${{startX}},${{startY}}Q${{controlX}},${{controlY}} ${{endX}},${{endY}}`;
    }}

    function sortUniqueStrings(values) {{
      return [...new Set((values || []).map(value => String(value)))]
        .sort((left, right) => left.localeCompare(right));
    }}

    function round3(value) {{
      return Math.round(Number(value || 0) * 1000) / 1000;
    }}

    function pythonStringRepr(value) {{
      return `'${{String(value)
        .replace(/\\\\/g, '\\\\\\\\')
        .replace(/'/g, "\\\\'")}}'`;
    }}

    function pythonTupleString(values) {{
      const parts = (values || []).map(value => pythonStringRepr(value));
      if (!parts.length) {{
        return '()';
      }}
      if (parts.length === 1) {{
        return `(${{parts[0]}},)`;
      }}
      return `(${{parts.join(', ')}})`;
    }}

    function serializeStablePart(part) {{
      return Array.isArray(part) ? pythonTupleString(part) : String(part ?? '');
    }}

    function fallbackStableRatioFromKey(key) {{
      let hash = 2166136261;
      for (let index = 0; index < key.length; index += 1) {{
        hash ^= key.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }}
      return (hash >>> 0) / 4294967295;
    }}

    async function stableRatio(...parts) {{
      const key = parts.map(part => serializeStablePart(part)).join('||');
      if (!(window.crypto && window.crypto.subtle && window.TextEncoder)) {{
        return fallbackStableRatioFromKey(key);
      }}

      const encoded = new TextEncoder().encode(key);
      const digest = await window.crypto.subtle.digest('SHA-256', encoded);
      const hex = Array.from(new Uint8Array(digest))
        .map(value => value.toString(16).padStart(2, '0'))
        .join('');
      const numerator = Number.parseInt(hex.slice(0, 12), 16);
      return numerator / (Math.pow(16, 12) - 1);
    }}

    async function stableFloat(low, high, ...parts) {{
      if (high <= low) {{
        return round3(low);
      }}
      const ratio = await stableRatio(...parts);
      return round3(low + ((high - low) * ratio));
    }}

    async function geoAssociativityScore(nodeData, attribute, geoAttribute) {{
      if (!nodeData || !attribute || !geoAttribute) {{
        return 0;
      }}

      const primaryKey = nodeData.primaryKey ? String(nodeData.primaryKey) : '';
      if (primaryKey && String(attribute) === primaryKey) {{
        return 1.0;
      }}

      const ratio = await stableRatio(
        nodeData.rawColumns || [],
        String(attribute),
        String(geoAttribute),
      );
      return round3(0.3 + (0.4 * ratio));
    }}

    function pairJoinAttributes(label, tableLeft, tableRight) {{
      if (!label) {{
        return [['', '']];
      }}

      const labelClean = String(label).replace(/…/g, '').trim();
      const parts = labelClean
        .split(',')
        .map(part => part.trim())
        .filter(Boolean);
      const leftAttrs = [];
      const rightAttrs = [];

      parts.forEach((part) => {{
        if (part.includes('≈')) {{
          const pieces = part.split('≈', 2).map(piece => piece.trim());
          leftAttrs.push(pieces[0] || '');
          rightAttrs.push(pieces[1] || '');
        }} else if (part.startsWith(`${{tableLeft}}.`)) {{
          leftAttrs.push(part.split('.', 2)[1] || '');
        }} else if (part.startsWith(`${{tableRight}}.`)) {{
          rightAttrs.push(part.split('.', 2)[1] || '');
        }} else {{
          leftAttrs.push(part);
          rightAttrs.push(part);
        }}
      }});

      const maxLen = Math.max(leftAttrs.length, rightAttrs.length, 1);
      return Array.from({{ length: maxLen }}, (_, index) => [
        leftAttrs[index] || '',
        rightAttrs[index] || '',
      ]);
    }}

    async function buildRelationGeoAugmentation(linkData) {{
      if (!linkData || linkData.type !== 'relation') {{
        return null;
      }}
      return linkData.geoAugmentation || null;
    }}

    async function submitGeoAugmentationRequest(payload) {{
      if (!payload) {{
        return;
      }}
      await applyGeoAugmentationVisuals(payload);
      hideEdgePopup();
      if (!window.parent || typeof window.parent.submitGeoAugmentationRequest !== 'function') {{
        return;
      }}
      window.parent.submitGeoAugmentationRequest({{
        type: 'geo_augmentation_request',
        payload: {{
          ...payload,
          nonce: Date.now(),
        }},
      }});
    }}

    async function applyGeoAugmentationVisuals(payload) {{
      const targetTable = payload && payload.targetTable;
      if (!targetTable) {{
        return;
      }}

      const targetNode = data.nodes.find(nodeData => nodeData.id === targetTable);
      if (!targetNode) {{
        return;
      }}

      targetNode.nodeFill = '{NODE_FILL_AUGMENTED_GEO}';
      targetNode.nodeStroke = '{NODE_STROKE_DEFAULT}';
      targetNode.labelFill = '#111111';

      const geoColumns = sortUniqueStrings(payload.geoColumns || []);
      targetNode.geoColumns = sortUniqueStrings([
        ...(targetNode.geoColumns || []),
        ...geoColumns,
      ]);
      const existingAttributes = new Set((targetNode.attributes || []).map(value => String(value)));
      targetNode.attributes = [...(targetNode.attributes || [])];
      geoColumns.forEach((geoColumn) => {{
        const annotated = `${{geoColumn}} (geo)`;
        if (!existingAttributes.has(geoColumn) && !existingAttributes.has(annotated)) {{
          targetNode.attributes.push(annotated);
          existingAttributes.add(annotated);
        }}
      }});

      const targetJoinScore = Number(payload.targetJoinScore || 0);
      const targetScore = Number(payload.targetScore || 0);
      const topAttr = targetJoinScore >= targetScore
        ? (payload.targetAttr || payload.targetPrimaryKey)
        : (payload.targetPrimaryKey || payload.targetAttr);
      const topScore = targetJoinScore >= targetScore ? targetJoinScore : targetScore;
      if (topAttr) {{
        targetNode.geoAssociativityLines = geoColumns.map(
          geoColumn =>
            `${{geoColumn}} (geo): Top associativity score (${{topAttr}}, ${{topScore.toFixed(3)}})`
        );
      }}

      node
        .filter(nodeData => nodeData.id === targetTable)
        .attr('fill', targetNode.nodeFill)
        .attr('stroke', targetNode.nodeStroke);

    }}

    function showPopupHtml(event, html) {{
      edgePopupBody.html(html);
      edgePopup.style('display', 'block');

      const graphRect = document.getElementById('graph').getBoundingClientRect();
      const popupNode = edgePopup.node();
      let left = event.clientX - graphRect.left + 12;
      let top = event.clientY - graphRect.top + 12;
      const maxLeft = Math.max(12, graphRect.width - popupNode.offsetWidth - 12);
      const maxTop = Math.max(12, graphRect.height - popupNode.offsetHeight - 12);
      left = Math.max(12, Math.min(left, maxLeft));
      top = Math.max(12, Math.min(top, maxTop));
      edgePopup
        .style('left', `${{left}}px`)
        .style('top', `${{top}}px`);
    }}

    async function showEdgePopup(event, d) {{
      const currentEdgeKey = edgeKey(d);
      selectedEdgeKey = currentEdgeKey;
      updateLinkStyles();
      const popupEvent = {{
        clientX: event.clientX,
        clientY: event.clientY,
      }};
      if (d.type === 'relation') {{
        d.geoAugmentation = await buildRelationGeoAugmentation(d);
      }}
      if (selectedEdgeKey !== currentEdgeKey) {{
        return;
      }}
      const popupTitle = d.popupTitle || 'Join columns';
      const popupLines = (d.popupLines || []).map(line =>
        `<div style="margin-top:4px;">${{escapeHtml(line)}}</div>`
      ).join('');
      const detailLines = (d.detailLines || []).map(line =>
        `<div style="margin-top:4px;">${{escapeHtml(line)}}</div>`
      ).join('');
      let popupHtml =
        `<div style="font-weight:600;color:#0f172a;margin-bottom:6px;">${{escapeHtml(popupTitle)}}</div>${{popupLines}}`;
      if (detailLines) {{
        popupHtml +=
          `<div style="font-weight:600;color:#0f172a;margin-top:10px;">Details</div>${{detailLines}}`;
      }}
      if (d.geoAugmentation) {{
        const assocLines = (d.geoAugmentation.assocLines || []).map(line =>
          `<div style="margin-top:4px;">${{escapeHtml(line)}}</div>`
        ).join('');
        const statusColor = d.geoAugmentation.canAugment ? '#166534' : '#991b1b';
        popupHtml +=
          `<div style="font-weight:600;color:#0f172a;margin-top:10px;">Geo augmentation</div>`;
        if (assocLines) {{
          popupHtml +=
            `<div style="font-weight:600;color:#0f172a;margin-top:6px;">Assoc. Score:</div>` +
            assocLines;
        }}
        if (d.geoAugmentation.geoColumnsLine) {{
          popupHtml +=
            `<div style="margin-top:8px;">${{escapeHtml(d.geoAugmentation.geoColumnsLine)}}</div>`;
        }}
        popupHtml +=
          `<div style="margin-top:8px;color:${{statusColor}};">${{escapeHtml(d.geoAugmentation.status || '')}}</div>`;
        if (d.type === 'relation' && d.geoAugmentation.canAugment) {{
          popupHtml +=
            `<button type="button" id="ddg-geo-augment-btn" ` +
            `style="margin-top:10px;padding:6px 10px;border:1px solid #0d47a1;border-radius:6px;background:#0d47a1;color:#fff;cursor:pointer;">` +
            `${{escapeHtml(d.geoAugmentation.buttonLabel || 'Apply')}}` +
            `</button>`;
        }}
      }}
      showPopupHtml(popupEvent, popupHtml);
      const augmentButton = edgePopupBody.select('#ddg-geo-augment-btn');
      if (!augmentButton.empty()) {{
        augmentButton.on('click', async function(clickEvent) {{
          clickEvent.stopPropagation();
          await submitGeoAugmentationRequest(d.geoAugmentation.requestPayload);
        }});
      }}
    }}

    function showNodePopup(event, d) {{
      selectedEdgeKey = null;
      updateLinkStyles();
      const attributeLines = (d.attributes || []).map(attribute =>
        `<div style="margin-top:4px;">${{escapeHtml(attribute)}}</div>`
      ).join('');
      const geoAssociativityLines = (d.geoAssociativityLines || []).map(line =>
        `<div style="margin-top:4px;">${{escapeHtml(line)}}</div>`
      ).join('');
      let popupHtml =
        `<div style="font-weight:600;color:#0f172a;margin-bottom:6px;">${{escapeHtml(d.id)}}</div>` +
        `<div style="font-weight:600;color:#0f172a;margin-top:2px;">Attributes</div>${{attributeLines}}`;
      if (geoAssociativityLines) {{
        popupHtml +=
          `<div style="font-weight:600;color:#0f172a;margin-top:10px;">Geo associativity</div>${{geoAssociativityLines}}`;
      }}
      showPopupHtml(event, popupHtml);
    }}

    edgePopupClose.on('click', function(event) {{
      event.stopPropagation();
      hideEdgePopup();
    }});

    node.on('click', function(event, d) {{
        event.stopPropagation();
        if (d.wasDragged) {{
          d.wasDragged = false;
          return;
        }}
        showNodePopup(event, d);
      }});

    linkHitArea.on('mouseover', function(event, d) {{
      if (edgeKey(d) === selectedEdgeKey) {{
        return;
      }}
      link.filter(linkData => edgeKey(linkData) === edgeKey(d))
        .attr('stroke-width', 4)
        .attr('stroke-opacity', 1);
    }}).on('mouseout', function(event, d) {{
      if (edgeKey(d) === selectedEdgeKey) {{
        return;
      }}
      updateLinkStyles();
    }}).on('click', function(event, d) {{
      event.stopPropagation();
      showEdgePopup(event, d);
    }});

    svg.on('click', () => hideEdgePopup());

    function ticked() {{
      link.attr('d', d => linkPath(d));

      linkHitArea.attr('d', d => linkPath(d));

      node
        .attr('cx', d => d.x = Math.max(26, Math.min(width - 26, d.x)))
        .attr('cy', d => d.y = Math.max(26, Math.min(height - 26, d.y)));

      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    }}

    function drag(simulation) {{
      function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d.wasDragged = false;
      }}

      function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
        d.wasDragged = true;
      }}

      function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}

      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }}
    updateLinkStyles();
    </script>
    """


def render_data_discovery_graph_section(
    west_lafayette_bbox, lafayette_default_bbox
):
    # Inject CSS once at the top (no spacing)
    st.markdown(
        '<style>div[data-testid="stHtml"] iframe[src*="folium"]{height:1400px!important;min-height:1400px!important;max-height:1400px!important;}</style>',
        unsafe_allow_html=True,
    )
    st.markdown('<div id="data-discovery-graph"></div>', unsafe_allow_html=True)
    st.subheader("Data Discovery Graph")
    lake = st.session_state.data_lake
    st.markdown(_data_discovery_legend_html(), unsafe_allow_html=True)
    row1_c_geo_with, row1_c_geo_without = st.columns(2)

    with row1_c_geo_with:
        show_geo = st.session_state.get("ddg_show_geo_tables", False)
        if st.button(
            "Show/Hide Tables with Geo",
            key="ddg_toggle_tables_geo",
            width="stretch",
        ):
            st.session_state.ddg_show_geo_tables = not show_geo

    with row1_c_geo_without:
        show_nogeo = st.session_state.get("ddg_show_nogeo_tables", False)
        if st.button(
            "Show/Hide Tables without Geo",
            key="ddg_toggle_tables_nogeo",
            width="stretch",
        ):
            st.session_state.ddg_show_nogeo_tables = not show_nogeo

    row2_c_rel, row2_c_sem, row2_c_spatial = st.columns(3)

    with row2_c_rel:
        rel_on = st.session_state.get("ddg_show_rel_joins", False)
        if st.button(
            "Show/Hide equi-joins",
            key="ddg_toggle_rel_joins",
            width="stretch",
        ):
            rel_on = not rel_on
            st.session_state.ddg_show_rel_joins = rel_on
            if rel_on:
                st.session_state.graph_rel_edges = (
                    graph_joins.find_relational_joins(st.session_state.data_lake)
                )
            else:
                st.session_state.graph_rel_edges = []

    with row2_c_sem:
        sem_on = st.session_state.get("ddg_show_semantic_joins", False)
        joinability_threshold = st.session_state.get("graph_equi_joinability_measure", 0.0)
        if st.button(
            "Show/Hide semantic joins",
            key="ddg_toggle_semantic_joins",
            width="stretch",
        ):
            sem_on = not sem_on
            st.session_state.ddg_show_semantic_joins = sem_on
            if sem_on:
                st.session_state.graph_semantic_edges = (
                    graph_joins.find_semantic_joins(
                        st.session_state.data_lake,
                        min_overlap=joinability_threshold,
                    )
                )
            else:
                st.session_state.graph_semantic_edges = []

    with row2_c_spatial:
        spatial_on = st.session_state.get("ddg_show_spatial_joins", False)
        if st.button(
            "Show/Hide spatial joins",
            key="ddg_toggle_spatial_joins",
            width="stretch",
        ):
            spatial_on = not spatial_on
            st.session_state.ddg_show_spatial_joins = spatial_on
            if spatial_on:
                st.session_state.ddg_spatial_edges = _full_ddg_spatial_edges(
                    st.session_state.data_lake
                )
            else:
                st.session_state.ddg_spatial_edges = []
            st.session_state.ddg_spatial_refresh_tables = []

    row3_c_bulk_geo, = st.columns(1)
    with row3_c_bulk_geo:
        if st.button(
            "Apply Geo-Augmentation to All",
            key="ddg_apply_geo_augmentation_all",
            width="stretch",
        ):
            _apply_bulk_geo_augmentation_from_discovery()

    geo_aug_message = st.session_state.get("ddg_geo_aug_message")
    if geo_aug_message:
        if geo_aug_message.get("kind") == "success":
            st.success(geo_aug_message.get("text", ""))
        else:
            st.warning(geo_aug_message.get("text", ""))
        st.session_state.ddg_geo_aug_message = None

    # if st.session_state.get("discovery_geo_augmentation", False):
    #     st.caption(
    #         "Geo-augmentation has been applied in this session: extra spatial columns were "
    #         "joined in from geo tables and saved under temp_tables/; "
    #         "the data lake in memory uses those versions."
    #     )

    joinability_threshold = st.session_state.get("graph_equi_joinability_measure", 0.0)
    if st.session_state.get("ddg_show_semantic_joins", False):
        st.session_state.graph_semantic_edges = graph_joins.find_semantic_joins(
            lake,
            min_overlap=joinability_threshold,
        )
    if st.session_state.get("ddg_show_spatial_joins", False):
        refresh_tables = set(st.session_state.get("ddg_spatial_refresh_tables", []))
        if refresh_tables and st.session_state.get("ddg_spatial_edges"):
            _refresh_ddg_spatial_edges_for_tables(lake, refresh_tables)
            st.session_state.ddg_spatial_refresh_tables = []
        elif refresh_tables or not st.session_state.get("ddg_spatial_edges"):
            st.session_state.ddg_spatial_edges = _full_ddg_spatial_edges(lake)
            st.session_state.ddg_spatial_refresh_tables = []
    visible_names = _ddg_visible_table_names(lake)
    st.session_state.graph_nodes = sorted(visible_names)

    native_geo_tables, augmented_tables, augmented_geo_columns_map = (
        _ddg_geo_augmentation_info(lake)
    )
    manual_geo_details = _manual_geo_augmentation_details()
    geo_association_details_map = {
        name: _table_geo_association_details(
            name,
            manual_geo_details=manual_geo_details,
        )
        for name in lake
    }
    geo_aug_enabled = bool(augmented_tables)
    nodes_data = [
        {
            "id": name,
            "rows": len(df),
            "columns": len(df.columns),
            "nodeFill": node_fill,
            "nodeStroke": _table_node_stroke(node_fill),
            "labelFill": _table_label_fill(node_fill),
            "rawColumns": [str(column) for column in df.columns],
            "geoColumns": sorted(augmented_geo_columns_map.get(name, set())),
            "primaryKey": compute_associativity.infer_primary_key(df),
            "attributes": _node_attribute_lines(
                df,
                augmented_geo_columns_map.get(name, set()),
            ),
            "geoAssociativityLines": _node_geo_associativity_lines(
                df,
                augmented_geo_columns_map.get(name, set()),
                geo_association_details_map.get(name, {}),
            ),
        }
        for name, df in lake.items()
        if name in visible_names
        for node_fill in [
            _table_node_fill(
                name,
                df,
                native_geo_tables,
                augmented_tables,
                geo_aug_enabled,
            )
        ]
    ]

    edges_main = []
    if st.session_state.get("ddg_show_rel_joins", False):
        for left, right, label in st.session_state.graph_rel_edges:
            edges_main.append(
                {
                    "source": left,
                    "target": right,
                    "label": label,
                    "type": "relation",
                    "popupLines": _edge_popup_lines(label, left, right),
                    "geoAugmentation": _relation_geo_augmentation_data(
                        left,
                        right,
                        label,
                        lake,
                        augmented_geo_columns_map,
                        geo_association_details_map,
                    ),
                }
            )

    if st.session_state.get("ddg_show_semantic_joins", False):
        for left, right, label in st.session_state.get("graph_semantic_edges", []):
            edges_main.append(
                {
                    "source": left,
                    "target": right,
                    "label": label,
                    "type": "semantic",
                    "popupLines": _edge_popup_lines(label, left, right),
                }
            )

    spatial_edges = (
        st.session_state.get("ddg_spatial_edges", [])
        if st.session_state.get("ddg_show_spatial_joins", False)
        else []
    )
    for edge in spatial_edges:
        edges_main.append(dict(edge))

    edges_main = [
        e
        for e in edges_main
        if e["source"] in visible_names and e["target"] in visible_names
    ]

    edges_data = edges_main

    component_event = DDG_COMPONENT(
        graph_html=_d3_discovery_graph_html({"nodes": nodes_data, "links": edges_main}),
        height=540,
        default=None,
        key="ddg_graph_component",
    )
    if _process_ddg_geo_augmentation_request(component_event, lake):
        st.rerun()

    # Add button to show join details
    if st.button("Show join details"):
        st.session_state.show_join_details = True
    
    # Display join details table if requested
    if st.session_state.get("show_join_details", False):
        join_details_data = []

        for edge in edges_data:
            base_table_1 = edge["source"]
            base_table_2 = edge["target"]
            attributes = edge.get("label", "")
            if edge["type"] == "relation":
                join_type = "relational"
                attribute_pairs = _pair_join_attributes(attributes, base_table_1, base_table_2)
            elif edge["type"] == "semantic":
                join_type = "semantic"
                attribute_pairs = _pair_join_attributes(attributes, base_table_1, base_table_2)
            elif edge["type"] == "spatial":
                predicate = edge.get("predicateDisplay") or edge.get("predicate", "unknown")
                join_type = f"spatial ({predicate})"
                attribute_pairs = [
                    (
                        str(edge.get("sourceAttribute") or ""),
                        str(edge.get("targetAttribute") or ""),
                    )
                ]
            else:
                join_type = edge["type"]
                attribute_pairs = _pair_join_attributes(attributes, base_table_1, base_table_2)
            
            for attr_left, attr_right in attribute_pairs:
                join_details_data.append({
                    "Base Table #1": base_table_1,
                    "Attribute (Table 1)": attr_left,
                    "Base Table #2": base_table_2,
                    "Attribute (Table 2)": attr_right,
                    "Type of join": join_type
                })
        
        if join_details_data:
            join_details_df = pd.DataFrame(join_details_data)
            join_details_df.index = range(1, len(join_details_df) + 1)
            # Render as HTML so font sizing reliably applies to the table.
            join_details_html = (
                join_details_df.style
                .set_properties(**{"font-size": "100%"})
                .set_table_styles([{"selector": "th", "props": [("font-size", "100%")]}])
                .to_html()
            )
            # Wrap entire table in scrollable container showing ~5 rows at a time
            scrollable_html = f'<div style="max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 4px;">{join_details_html}</div>'
            st.markdown(scrollable_html, unsafe_allow_html=True)
        else:
            st.info("No joins are currently displayed in the graph.")

    # Show Join Paths section when join details are shown
    if st.session_state.get("show_join_details", False):
        _render_interesting_paths(west_lafayette_bbox, lafayette_default_bbox)


def _extract_point_coords(point_str):
    """Extract lat, lon from POINT(lon lat) or similar formats."""
    if not isinstance(point_str, str):
        return None
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', point_str, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lat, lon)
    return None


def _extract_points_from_path(path_tables, data_lake, max_points_per_table=200):
    """
    Extract all point coordinates from all tables in a path.
    Returns: list of (lat, lon) tuples
    """
    all_coords = []
    
    for table_name in path_tables:
        if table_name not in data_lake:
            continue
        df = data_lake[table_name]
        
        # Try to find lat/lon columns
        lat_col = None
        lon_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        sample_vals = pd.to_numeric(sample.head(10), errors='coerce').dropna()
                        if len(sample_vals) > 0 and all(-90 <= v <= 90 for v in sample_vals):
                            lat_col = col
                    except Exception:
                        pass
            if ('lon' in col_lower or 'lng' in col_lower) and lon_col is None:
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        sample_vals = pd.to_numeric(sample.head(10), errors='coerce').dropna()
                        if len(sample_vals) > 0 and all(-180 <= v <= 180 for v in sample_vals):
                            lon_col = col
                    except Exception:
                        pass
        
        # Extract coordinates from lat/lon columns
        if lat_col and lon_col:
            try:
                coords_df = df[[lat_col, lon_col]].dropna()
                for _, row in coords_df.head(max_points_per_table).iterrows():
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            all_coords.append((lat, lon, table_name))
                    except Exception:
                        continue
            except Exception:
                pass
        
        # Extract coordinates from spatial columns (POINT format)
        spatial_cols = graph_joins._find_spatial_columns(df)
        for col in spatial_cols:
            sample = df[col].dropna().astype(str).head(max_points_per_table)
            for val in sample:
                coords = _extract_point_coords(val)
                if coords:
                    all_coords.append((coords[0], coords[1], table_name))
    
    return all_coords


def _compute_bbox_from_path(path_tables, data_lake):
    """
    Extract spatial coordinates from all tables in a path and compute bounding box.
    Returns: [south, west, north, east] or None if no coordinates found
    """
    all_coords = _extract_points_from_path(path_tables, data_lake)
    
    # Compute bounding box from all coordinates
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]
        return bbox
    
    return None


def _render_interesting_paths(west_lafayette_bbox, lafayette_default_bbox):
    st.markdown('<div id="interesting-paths"></div>', unsafe_allow_html=True)
    st.subheader("Join paths")

    XA = 40.0  # south (minimum latitude)
    XB = -87.0  # west (minimum longitude)
    XC = 41.0  # north (maximum latitude)
    XD = -86.0  # east (maximum longitude)
    # Get parameters from session state (set in sidebar)
    path_min_len = st.session_state.get("path_min_len", "")
    path_max_len = st.session_state.get("path_max_len", "")
    path_num_relational_joins = st.session_state.get("path_num_relational_joins", 0)
    path_num_spatial_joins = st.session_state.get("path_num_spatial_joins", 0)
    path_spatial_join_type = st.session_state.get("path_spatial_join_type", "Containment")

    if st.button("Generate path"):
        # Build graph from edges_data (joinability information)
        edges_data = []
        predicate_map = {
            "Containment": "containment",
            "Overlapping": "overlap",
            "Distance": "distance",
        }
        selected_path_predicate = predicate_map.get(path_spatial_join_type, "containment")

        for left, right, label in st.session_state.graph_rel_edges:
            edges_data.append({"source": left, "target": right, "type": "relation"})

        for left, right, label in st.session_state.get("graph_semantic_edges", []):
            edges_data.append({"source": left, "target": right, "type": "semantic"})

        for edge in st.session_state.get("ddg_spatial_edges", []):
            predicates = edge.get("supportedPredicates") or [edge.get("predicate")]
            edges_data.append(
                {
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "type": "spatial",
                    "predicates": [predicate for predicate in predicates if predicate],
                }
            )

        # Build adjacency list from edges
        graph = {}
        edge_types = {}
        edge_spatial_predicates = {}
        for edge in edges_data:
            source = edge["source"]
            target = edge["target"]
            if source not in graph:
                graph[source] = []
            if target not in graph:
                graph[target] = []
            graph[source].append(target)
            graph[target].append(source)
            edge_key = frozenset((source, target))
            edge_types.setdefault(edge_key, set()).add(edge["type"])
            if edge["type"] == "spatial":
                edge_spatial_predicates.setdefault(edge_key, set()).update(
                    edge.get("predicates") or []
                )
        
        # Find all valid paths through the graph
        def find_paths(start, max_length, visited=None, current_path=None):
            if visited is None:
                visited = set()
            if current_path is None:
                current_path = []
            
            # Create copies to avoid mutation issues
            new_visited = visited.copy()
            new_path = current_path.copy()
            
            # Add current node to path
            new_path.append(start)
            new_visited.add(start)
            
            paths = []
            # If path has at least 2 nodes, add it as a valid path
            if len(new_path) >= 2:
                paths.append(tuple(new_path))
            
            # Continue exploring neighbors if we haven't reached max length
            if len(new_path) < max_length:
                if start in graph:
                    for neighbor in graph[start]:
                        if neighbor not in new_visited:
                            paths.extend(find_paths(neighbor, max_length, new_visited, new_path))
            
            return paths
        
        # Generate paths from all nodes
        candidate_paths = []
        max_path_length = 10
        if path_max_len.strip():
            try:
                max_path_length = max(2, int(path_max_len))
            except ValueError:
                max_path_length = 10
        all_nodes = sorted(st.session_state.data_lake.keys())
        
        for start_node in all_nodes:
            paths_from_node = find_paths(start_node, max_path_length)
            candidate_paths.extend(paths_from_node)
        
        # Remove duplicates
        candidate_paths = list(set(candidate_paths))
        
        # Filter by minimum path length
        if path_min_len.strip():
            try:
                min_path_length = max(2, int(path_min_len))
                candidate_paths = [path for path in candidate_paths if len(path) >= min_path_length]
            except ValueError:
                pass

        # Filter by maximum path length
        if path_max_len.strip():
            try:
                max_path_length_val = max(2, int(path_max_len))
                candidate_paths = [path for path in candidate_paths if len(path) <= max_path_length_val]
            except ValueError:
                pass

        # Filter by number of relational joins
        if path_num_relational_joins > 0:
            def _count_joins_by_type(path, join_type):
                count = 0
                for i in range(len(path) - 1):
                    edge_key = frozenset((path[i], path[i + 1]))
                    if join_type in edge_types.get(edge_key, set()):
                        count += 1
                return count
            
            candidate_paths = [
                path for path in candidate_paths 
                if _count_joins_by_type(path, "relation") >= path_num_relational_joins
            ]

        # Filter by number of spatial joins for the requested predicate type.
        if path_num_spatial_joins > 0:
            def _count_spatial_joins(path):
                count = 0
                for i in range(len(path) - 1):
                    edge_key = frozenset((path[i], path[i + 1]))
                    if (
                        "spatial" in edge_types.get(edge_key, set())
                        and selected_path_predicate in edge_spatial_predicates.get(edge_key, set())
                    ):
                        count += 1
                return count
            
            candidate_paths = [
                path for path in candidate_paths 
                if _count_spatial_joins(path) >= path_num_spatial_joins
            ]

        
        # Sort by length descending (longer paths first), then alphabetically
        candidate_paths = sorted(candidate_paths, key=lambda x: (-len(x), x))
        
        # Filter to only keep paths of length 5 (or longest available if none are length 5)
        filtered_paths = []
        if candidate_paths:
            max_path_len = max(len(p) for p in candidate_paths)
            # Prioritize paths of length 5, or use the maximum available length
            target_length = min(5, max_path_len)
            long_paths = [p for p in candidate_paths if len(p) == target_length]
            if long_paths:
                filtered_paths = long_paths
            else:
                # If no paths of target length, take longest available
                filtered_paths = [p for p in candidate_paths if len(p) == max_path_len]

        st.session_state["generated_path_count"] = len(filtered_paths)
        st.session_state["all_join_paths"] = filtered_paths
        st.session_state["interesting_paths"] = filtered_paths
        st.session_state["paths_page"] = 0
        st.session_state["selected_path_for_map"] = None

    if "path_map_selections" not in st.session_state:
        st.session_state["path_map_selections"] = {}

    if "selected_path_for_map" not in st.session_state:
        st.session_state["selected_path_for_map"] = None

    if "paths_page" not in st.session_state:
        st.session_state["paths_page"] = 0

    paths = st.session_state.get("interesting_paths", [])
    if "generated_path_count" in st.session_state:
        st.markdown(f"{st.session_state['generated_path_count']} join paths generated")

    if paths:
        page_size = 5
        total_paths = len(paths)
        page_count = (total_paths + page_size - 1) // page_size
        current_page = min(st.session_state["paths_page"], max(page_count - 1, 0))
        st.session_state["paths_page"] = current_page
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_paths)

        st.markdown("**Join Paths**")

        def get_join_info_discovery(table1, table2):
            """Get join information between two tables for data discovery graph."""
            for left, right, label in st.session_state.get("graph_semantic_edges", []):
                if (left == table1 and right == table2) or (left == table2 and right == table1):
                    return {"type": "semantic", "attributes": label}
            for left, right, label in st.session_state.graph_rel_edges:
                if (left == table1 and right == table2) or (left == table2 and right == table1):
                    return {"type": "relation", "attributes": label}
            for edge in st.session_state.get("ddg_spatial_edges", []):
                left = edge.get("source")
                right = edge.get("target")
                if (left == table1 and right == table2) or (left == table2 and right == table1):
                    return {"type": "spatial", "attributes": edge.get("label", "")}
            return {"type": "relation", "attributes": ""}

        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button(
                "Previous join paths",
                key="paths_prev_page",
                disabled=current_page <= 0,
                width="stretch",
            ):
                st.session_state["paths_page"] = current_page - 1
                st.session_state["selected_path_for_map"] = None
                st.rerun()
        with nav_cols[1]:
            if st.button(
                "Next join paths",
                key="paths_next_page",
                disabled=current_page >= page_count - 1,
                width="stretch",
            ):
                st.session_state["paths_page"] = current_page + 1
                st.session_state["selected_path_for_map"] = None
                st.rerun()

        table_metadata = build_table_metadata(st.session_state.data_lake)
        selected_path_idx = st.session_state.get("selected_path_for_map")
        for idx, path in enumerate(paths[start_idx:end_idx], start=start_idx + 1):
            is_selected = selected_path_idx == idx

            path_data = {
                "tables": list(path),
                "joins": []
            }

            for i in range(len(path) - 1):
                join_info = get_join_info_discovery(path[i], path[i+1])
                path_data["joins"].append({
                    "from": path[i],
                    "to": path[i+1],
                    "attributes": join_info["attributes"],
                    "type": join_info["type"]
                })

            render_join_path_card(
                component_id=f"discovery_path_{idx}_card",
                title=f"Join Path {idx:03d}",
                path_data=path_data,
                table_metadata=table_metadata,
                stats={
                    "len": len(path),
                    "hops": max(len(path) - 1, 0),
                },
                stat_items=[("len", "LEN"), ("hops", "HOPS")],
                selected=is_selected,
            )

            if st.button(
                "Show on the map",
                key=f"show_path_{idx}",
                width="stretch",
            ):
                st.session_state["selected_path_for_map"] = idx
                computed_bbox = _compute_bbox_from_path(path, st.session_state.data_lake)
                if computed_bbox:
                    st.session_state.path_map_selections[f"path_{idx}"] = {
                        "bbox": computed_bbox,
                        "region_name": "Computed from path tables",
                        "method": "spatial_coordinates"
                    }
                else:
                    st.session_state.path_map_selections[f"path_{idx}"] = {
                        "bbox": [XA, XB, XC, XD],
                        "region_name": "Path Region",
                        "method": "default_coordinates"
                    }
                st.rerun()

        st.markdown("**Map View**")

        selected_path_idx = st.session_state.get("selected_path_for_map")

        path_region_info = None
        path_map_bbox = None

        if selected_path_idx:
            stored_info = st.session_state.path_map_selections.get(f"path_{selected_path_idx}")
            if stored_info:
                if isinstance(stored_info, dict):
                    path_region_info = stored_info
                    path_map_bbox = stored_info.get("bbox")
                else:
                    path_map_bbox = stored_info
            else:
                path_map_bbox = None
            if path_map_bbox is None and paths:
                path = paths[selected_path_idx - 1]
                path_map_bbox = _compute_bbox_from_path(path, st.session_state.data_lake)
            if path_map_bbox is None:
                path_map_bbox = west_lafayette_bbox
        else:
            path_map_bbox = None

        if path_map_bbox:
            center_lat = (path_map_bbox[0] + path_map_bbox[2]) / 2
            center_lon = (path_map_bbox[1] + path_map_bbox[3]) / 2
            lat_range = path_map_bbox[2] - path_map_bbox[0]
            lon_range = path_map_bbox[3] - path_map_bbox[1]
            max_range = max(lat_range, lon_range)
            if max_range < 0.1:
                zoom_level = 13
            elif max_range < 0.5:
                zoom_level = 11
            elif max_range < 1.0:
                zoom_level = 9
            elif max_range < 5.0:
                zoom_level = 7
            else:
                zoom_level = 5
        else:
            center_lat = (lafayette_default_bbox[0] + lafayette_default_bbox[2]) / 2
            center_lon = (lafayette_default_bbox[1] + lafayette_default_bbox[3]) / 2
            zoom_level = 12

        m_path = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )

        if path_map_bbox is not None:
            folium.Rectangle(
                bounds=[[path_map_bbox[0], path_map_bbox[1]], [path_map_bbox[2], path_map_bbox[3]]],
                color="#0066cc",
                fillColor="#0066cc",
                fillOpacity=0.3,
                weight=2,
                opacity=0.8,
                popup=f"Path {selected_path_idx}",
                tooltip=f"Path {selected_path_idx} bounding box"
            ).add_to(m_path)
            
            if selected_path_idx and paths:
                path = paths[selected_path_idx - 1]
                points = _extract_points_from_path(path, st.session_state.data_lake, max_points_per_table=200)
                
                table_colors = {}
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                         'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 
                         'lightgreen', 'gray', 'black', 'lightgray']
                for idx, table_name in enumerate(path):
                    table_colors[table_name] = colors[idx % len(colors)]
                
                for lat, lon, table_name in points:
                    margin = 0.01
                    if (path_map_bbox[0] - margin <= lat <= path_map_bbox[2] + margin and 
                        path_map_bbox[1] - margin <= lon <= path_map_bbox[3] + margin):
                        color = table_colors.get(table_name, 'blue')
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            popup=f"{table_name}<br>({lat:.4f}, {lon:.4f})",
                            tooltip=table_name,
                            color=color,
                            fillColor=color,
                            fillOpacity=0.7,
                            weight=1
                        ).add_to(m_path)

        if HAS_ST_FOLIUM:
            st_folium(m_path, width=None, height=1400, returned_objects=[])
        else:
            map_html = m_path._repr_html_()
            import re
            map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*height:\s*)\d+px', r'\11400px', map_html)
            if 'id="map' in map_html:
                if 'style=' in map_html:
                    map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*)"', r'\1 height: 1400px !important;"', map_html)
                else:
                    map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*")', r'\1 style="height: 1400px !important;"', map_html)
            
            components.html(map_html, height=1400)
        st.markdown('<script>(function(){{function forceResize(){{var iframes=document.querySelectorAll("iframe");iframes.forEach(function(iframe){{iframe.style.setProperty("height","1400px","important");iframe.style.setProperty("min-height","1400px","important");iframe.style.setProperty("max-height","1400px","important");iframe.setAttribute("height","1400");}});}}forceResize();setTimeout(forceResize,100);setTimeout(forceResize,500);setTimeout(forceResize,1000);setTimeout(forceResize,2000);var observer=new MutationObserver(forceResize);observer.observe(document.body,{{childList:true,subtree:true}});}})();</script>', unsafe_allow_html=True)

        if selected_path_idx and path_region_info:
            region_name = path_region_info.get("region_name", "Unknown") if isinstance(path_region_info, dict) else "Unknown"
            method = path_region_info.get("method", "unknown") if isinstance(path_region_info, dict) else "unknown"
            st.success(f"**Path {selected_path_idx} Region:** {region_name} (detected via {method})")
        elif selected_path_idx and path_map_bbox:
            bbox_label_path = (
                f"{path_map_bbox[0]:.4f}, {path_map_bbox[1]:.4f}, "
                f"{path_map_bbox[2]:.4f}, {path_map_bbox[3]:.4f}"
            )
            st.info(f"**Path {selected_path_idx}:** Region highlighted on map (bbox: {bbox_label_path})")
