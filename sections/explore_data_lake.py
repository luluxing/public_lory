import leafmap.foliumap as leafmap
import streamlit as st
from collections import defaultdict

import graph_joins
from sections.user_query import _bbox_from_geojson_feature, _extract_point_coords


def _coerce_coords(val):
    """Best-effort parse of a value into (lat, lon)."""
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            lat, lon = float(val[0]), float(val[1])
            return (lat, lon)
        if isinstance(val, str):
            txt = val.strip()
            if "," in txt and "POINT" not in txt.upper():
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                if len(parts) >= 2:
                    lat, lon = float(parts[0]), float(parts[1])
                    return (lat, lon)
            point = _extract_point_coords(txt)
            if point:
                return point
    except Exception:
        return None
    return None


def _point_in_bbox(coords, bbox):
    """Return True if (lat, lon) is inside bbox [south, west, north, east]."""
    if not coords or not bbox or len(bbox) != 4:
        return False
    lat, lon = coords
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def _tables_overlapping_bbox(bbox):
    """Find tables with spatial columns that intersect the bbox."""
    hits = []
    data_lake = st.session_state.get("data_lake", {})
    for table_name, df in data_lake.items():
        spatial_cols = graph_joins._find_spatial_columns(df)
        if not spatial_cols:
            continue

        matched_cols = []
        for col in spatial_cols:
            try:
                # Sample rows to keep things light.
                sample_vals = df[col].dropna().head(500)
            except Exception:
                continue
            found = False
            for val in sample_vals:
                coords = _coerce_coords(val)
                if coords and _point_in_bbox(coords, bbox):
                    found = True
                    break
            if found:
                matched_cols.append(col)

        if matched_cols:
            hits.append({"table": table_name, "columns": matched_cols})
    return hits


def _bbox_equal(a, b, tol=1e-6):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _parse_label_columns(label: str):
    if not label:
        return []
    cols = label.replace("…", "").split(",")
    return [c.strip() for c in cols if c.strip()]


def render_explore_section():
    st.markdown('<div id="explore-data-lake"></div>', unsafe_allow_html=True)
    st.subheader("Explore Data Lake")
    st.markdown("**Bounding box (draw on the map)**")

    existing = st.session_state.get("bbox_selection", [39.9, -86.3, 40.1, -86.1])
    center_lat = (existing[0] + existing[2]) / 2
    center_lon = (existing[1] + existing[3]) / 2

    m = leafmap.Map(
        center=[center_lat, center_lon],
        zoom=6,
        draw_control=True,
        measure_control=False,
        fullscreen_control=False,
        locate_control=False,
    )
    m.add_basemap("OpenStreetMap")

    st_map = m.to_streamlit(height=700, bidirectional=True)
    new_bbox = None
    if st_map:
        last_draw = m.st_last_draw(st_map)
        new_bbox = _bbox_from_geojson_feature(last_draw)
        if not new_bbox:
            drawings = m.st_draw_features(st_map) or []
            for feat in reversed(drawings):
                new_bbox = _bbox_from_geojson_feature(feat)
                if new_bbox:
                    break

    if new_bbox:
        st.session_state["bbox_selection"] = new_bbox
        bbox_label = (
            f"{new_bbox[0]:.4f}, {new_bbox[1]:.4f}, "
            f"{new_bbox[2]:.4f}, {new_bbox[3]:.4f}"
        )
        st.success(f"Selected rectangle: {bbox_label}")
    else:
        bbox_label = (
            f"{existing[0]:.4f}, {existing[1]:.4f}, "
            f"{existing[2]:.4f}, {existing[3]:.4f}"
        )
        st.caption(f"Current bbox: {bbox_label} (draw a rectangle to update)")

    st.subheader("Tables within the selected region")
    active_bbox = st.session_state.get("bbox_selection")
    if active_bbox:
        overlaps = _tables_overlapping_bbox(active_bbox)
        overlap_tables = [hit["table"] for hit in overlaps]

        # Reset paths if bbox changed or no paths exist yet.
        paths_state = st.session_state.setdefault("explore_paths", [])
        bbox_state = st.session_state.setdefault("explore_paths_bbox", None)
        if not _bbox_equal(active_bbox, bbox_state):
            st.session_state.explore_paths_bbox = list(active_bbox)
            st.session_state.explore_paths = [[t] for t in overlap_tables]
            paths_state = st.session_state.explore_paths

        if overlaps:
            # Build adjacency among overlapping tables using join heuristics.
            lake_subset = {
                name: st.session_state.data_lake[name]
                for name in overlap_tables
                if name in st.session_state.data_lake
            }
            rel_edges = graph_joins.find_relational_joins(lake_subset)
            spatial_edges = graph_joins.find_spatial_joins(lake_subset)
            adj = defaultdict(set)
            edge_info = {}
            for left, right, _ in rel_edges + spatial_edges:
                adj[left].add(right)
                adj[right].add(left)
            for left, right, label in rel_edges:
                edge_info[(left, right)] = ("rel", label)
                edge_info[(right, left)] = ("rel", label)
            for left, right, label in spatial_edges:
                edge_info[(left, right)] = ("spatial", label)
                edge_info[(right, left)] = ("spatial", label)

            joinable_per_idx = []
            clicked_idx = None
            for idx, path in enumerate(paths_state):
                last_table = path[-1]
                joinable = sorted(
                    t for t in overlap_tables if t != last_table and t not in path and t in adj[last_table]
                )
                joinable_per_idx.append(joinable)

                cols = ", ".join(next((hit["columns"] for hit in overlaps if hit["table"] == last_table), []))
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.markdown(f"**Path {idx+1}:** {' → '.join(path)}")
                    # Show involved columns per table in the path.
                    detail_lines = []
                    for pos, tbl in enumerate(path):
                        if pos == 0:
                            spatial_cols = next((hit["columns"] for hit in overlaps if hit["table"] == tbl), [])
                            if spatial_cols:
                                detail_lines.append(f"{', '.join(f'{tbl}.{c}' for c in spatial_cols)}")
                            else:
                                detail_lines.append(f"{tbl}: (spatial columns not detected)")
                        else:
                            prev_tbl = path[pos - 1]
                            jtype, jlabel = edge_info.get((prev_tbl, tbl), (None, None))
                            candidates = _parse_label_columns(jlabel)
                            df_curr = st.session_state.data_lake.get(tbl)
                            cols_for_tbl = []
                            if df_curr is not None:
                                for cand in candidates:
                                    if cand in df_curr.columns:
                                        cols_for_tbl.append(f"{tbl}.{cand}")
                            if not cols_for_tbl and candidates:
                                cols_for_tbl.append(f"{tbl}.{candidates[0]}")
                            if not cols_for_tbl:
                                cols_for_tbl.append(f"{tbl} (join columns unknown)")
                            join_kind = "relational" if jtype == "rel" else "spatial" if jtype == "spatial" else "join"
                            detail_lines.append(f"{', '.join(cols_for_tbl)} [{join_kind}]")

                    if detail_lines:
                        st.caption("  • " + " | ".join(detail_lines))
                with col2:
                    if st.button("Add", key=f"explore_add_{idx}", help="Append joinable tables to this path"):
                        clicked_idx = idx

            # Apply the click after rendering buttons so only the clicked path expands.
            # Keep existing paths in order; append any new paths at the end.
            new_paths = list(paths_state)
            added_paths = False
            if clicked_idx is not None:
                joinable = joinable_per_idx[clicked_idx]
                if joinable:
                    for tbl in joinable:
                        new_paths.append(paths_state[clicked_idx] + [tbl])
                    added_paths = True

            # Deduplicate identical paths to avoid runaway growth.
            dedup = []
            seen = set()
            for p in new_paths:
                tup = tuple(p)
                if tup not in seen:
                    seen.add(tup)
                    dedup.append(p)
            st.session_state.explore_paths = dedup
            if added_paths:
                st.rerun()

        else:
            st.info("No tables with spatial columns inside/overlapping the selected box yet.")
    else:
        st.info("Draw a rectangle to see tables that intersect the selected box.")
