import json
import math
import re

import pandas as pd
import folium
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

import graph_joins
import semantic_joins
from geocoding import COUNTRY_ALIASES, COUNTRY_BBOXES
from sections.geo_ui import (
    DEFAULT_BBOX_SELECTION,
    bbox_from_geojson_feature,
    extract_point_coords,
)
from sections.join_path_renderer import (
    build_table_metadata,
    render_join_path_card,
    render_join_path_legend,
)
from sections.generate_target_schema import (
    find_min_join_path,
    fuzzy_match_attribute,
    generate_joined_tuples,
    infer_spatial_preferences,
    join_edges_for_path_ordered,
)


def _wkt_polygon_aabb_intersects_bbox(wkt, bbox):
    """
    True if the axis-aligned bounding box of a POLYGON outer ring overlaps
    bbox [south, west, north, east].
    """
    if not isinstance(wkt, str) or "POLYGON" not in wkt.upper():
        return False
    inner = re.search(r"POLYGON\s*\(\s*\(([^)]+)\)", wkt, re.IGNORECASE)
    if not inner:
        return False
    pairs = re.findall(r"([-+\d.]+)\s+([-+\d.]+)", inner.group(1))
    if not pairs:
        return False
    lons = [float(a) for a, _ in pairs]
    lats = [float(b) for _, b in pairs]
    poly_s, poly_n = min(lats), max(lats)
    poly_w, poly_e = min(lons), max(lons)
    south, west, north, east = bbox
    return not (poly_n < south or poly_s > north or poly_e < west or poly_w > east)


def _extract_wkt_polygon_bbox(wkt):
    if not isinstance(wkt, str) or "POLYGON" not in wkt.upper():
        return None
    inner = re.search(r"POLYGON\s*\(\s*\(([^)]+)\)", wkt, re.IGNORECASE)
    if not inner:
        return None
    pairs = re.findall(r"([-+\d.]+)\s+([-+\d.]+)", inner.group(1))
    if not pairs:
        return None
    lons = [float(a) for a, _ in pairs]
    lats = [float(b) for _, b in pairs]
    return [min(lats), min(lons), max(lats), max(lons)]


def _find_region_from_tables(path_tables, data_lake):
    """
    Prefer the finest available geographic granularity for a path:
    points, then polygon geometry, then state, then country.
    """
    all_coords = []
    polygon_bboxes = []
    state_names = []
    country_names = []

    for table_name in path_tables:
        if table_name not in data_lake:
            continue
        df = data_lake[table_name]

        for col in df.columns:
            col_lower = col.lower()
            if "state" in col_lower:
                unique_vals = df[col].dropna().unique()[:10]
                for val in unique_vals:
                    val_str = str(val).lower().strip()
                    state_name = US_STATE_ALIASES.get(val_str)
                    if state_name:
                        state_names.append(state_name.title())
            elif "country" in col_lower:
                unique_vals = df[col].dropna().unique()[:10]
                for val in unique_vals:
                    val_str = str(val).lower().strip()
                    country_key = COUNTRY_ALIASES.get(val_str)
                    if country_key in COUNTRY_BBOXES:
                        country_names.append(COUNTRY_DISPLAY_NAMES.get(country_key, country_key.upper()))

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
            if 'lon' in col_lower or ('lng' in col_lower):
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        sample_vals = pd.to_numeric(sample.head(10), errors='coerce').dropna()
                        if len(sample_vals) > 0 and all(-180 <= v <= 180 for v in sample_vals):
                            lon_col = col
                    except Exception:
                        pass

        if lat_col and lon_col:
            try:
                coords_df = df[[lat_col, lon_col]].dropna()
                for _, row in coords_df.head(100).iterrows():
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            all_coords.append((lat, lon))
                    except Exception:
                        continue
            except Exception:
                pass

        spatial_cols = graph_joins._find_spatial_columns(df)
        for col in spatial_cols:
            sample = df[col].dropna().astype(str).head(100)
            for val in sample:
                poly_bbox = _extract_wkt_polygon_bbox(val)
                if poly_bbox:
                    polygon_bboxes.append(poly_bbox)
                    continue
                coords = extract_point_coords(val)
                if coords:
                    all_coords.append(coords)

    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]
        return {
            "region_name": "Computed from coordinates",
            "bbox": bbox,
            "method": "coordinates"
        }

    if polygon_bboxes:
        return {
            "region_name": "Computed from polygon geometry",
            "bbox": _merge_bboxes(*polygon_bboxes),
            "method": "polygons",
        }

    if state_names:
        from collections import Counter
        most_common_region = Counter(state_names).most_common(1)[0][0]
        region_lower = most_common_region.lower()
        if region_lower in US_STATE_BBOXES:
            return {
                "region_name": most_common_region,
                "bbox": US_STATE_BBOXES[region_lower],
                "method": "state_column"
            }

    if country_names:
        from collections import Counter
        most_common_country = Counter(country_names).most_common(1)[0][0]
        country_key = COUNTRY_DISPLAY_KEYS.get(most_common_country)
        if country_key in COUNTRY_BBOXES:
            return {
                "region_name": most_common_country,
                "bbox": list(COUNTRY_BBOXES[country_key]),
                "method": "country_column",
            }

    return None


def _find_lat_lon_columns(df):
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_lower = col.lower()
        if lat_col is None and "lat" in col_lower:
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-90, 90).all():
                lat_col = col
        if lon_col is None and ("lon" in col_lower or "lng" in col_lower):
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-180, 180).all():
                lon_col = col
    return lat_col, lon_col


def _filter_table_by_bbox(df, bbox):
    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        mask = lats.between(bbox[0], bbox[2]) & lons.between(bbox[1], bbox[3])
        return df[mask], True

    spatial_cols = graph_joins._find_spatial_columns(df)
    for col in spatial_cols:
        s = df[col].astype(str)
        # WKT POLYGON: str.extract(POINT) still returns a row per line but all NaN —
        # must not treat that as a successful parse (would drop every row).
        if s.str.contains("POLYGON", case=False, regex=False).any():
            mask = s.map(lambda v: _wkt_polygon_aabb_intersects_bbox(v, bbox))
            return df[mask], True
        coords = s.str.extract(
            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
            flags=re.IGNORECASE,
        )
        if coords.empty or coords[0].isna().all():
            continue
        lons = pd.to_numeric(coords[0], errors="coerce")
        lats = pd.to_numeric(coords[1], errors="coerce")
        mask = lats.between(bbox[0], bbox[2]) & lons.between(bbox[1], bbox[3])
        return df[mask], True

    return df, False


def _get_lat_lon_series(df):
    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        return lats, lons

    spatial_cols = graph_joins._find_spatial_columns(df)
    for col in spatial_cols:
        s = df[col].astype(str)
        if s.str.contains("POLYGON", case=False, regex=False).any():
            lats_list = []
            lons_list = []
            for wkt in s:
                inner = re.search(r"POLYGON\s*\(\s*\(([^)]+)\)", str(wkt), re.IGNORECASE)
                if not inner:
                    lats_list.append(float("nan"))
                    lons_list.append(float("nan"))
                    continue
                m = re.search(r"([-+\d.]+)\s+([-+\d.]+)", inner.group(1))
                if not m:
                    lats_list.append(float("nan"))
                    lons_list.append(float("nan"))
                    continue
                lon, lat = float(m.group(1)), float(m.group(2))
                lons_list.append(lon)
                lats_list.append(lat)
            return pd.Series(lats_list, index=df.index), pd.Series(
                lons_list, index=df.index
            )
        coords = s.str.extract(
            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
            flags=re.IGNORECASE,
        )
        if coords.empty or coords[0].isna().all():
            continue
        lons = pd.to_numeric(coords[0], errors="coerce")
        lats = pd.to_numeric(coords[1], errors="coerce")
        return lats, lons

    return None, None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def _collect_join_paths(
    input_tables,
    rel_edges,
    semantic_edges,
    spatial_edges,
    allow_spatial,
    require_spatial,
    max_paths=5,
    max_len=6,
):
    if not input_tables:
        return []

    adjacency = {}

    def _add_edge(left, right, edge_type):
        adjacency.setdefault(left, []).append((right, edge_type))
        adjacency.setdefault(right, []).append((left, edge_type))

    for left, right, _ in rel_edges or []:
        _add_edge(left, right, "relation")
    for left, right, _ in semantic_edges or []:
        _add_edge(left, right, "semantic")
    if allow_spatial:
        for left, right, _ in spatial_edges or []:
            _add_edge(left, right, "spatial")

    input_tables = set(input_tables)
    max_len = max(max_len, len(input_tables))
    paths = set()

    def _dfs(current, path, has_spatial):
        if len(paths) >= max_paths:
            return
        if input_tables.issubset(path) and (not require_spatial or has_spatial):
            paths.add(tuple(path))
        if len(path) >= max_len:
            return
        for neighbor, edge_type in adjacency.get(current, []):
            if neighbor in path:
                continue
            _dfs(neighbor, path + [neighbor], has_spatial or edge_type == "spatial")
            if len(paths) >= max_paths:
                return

    for start in sorted(input_tables):
        _dfs(start, [start], False)
        if len(paths) >= max_paths:
            break

    return sorted(paths, key=lambda p: (len(p), p))


# Vision demo: Path 1 is always police_reports → accidents → hospitals
# (reports↔accidents on accident_id; accidents↔hospitals by closest-point spatial join).
HARDCODED_FIRST_SCENARIO1_PATH = ("police_reports", "accidents", "hospitals")


def hardcoded_first_scenario1_path_edges():
    return [
        {
            "from": "police_reports",
            "to": "accidents",
            "type": "relation",
            "attributes": "accident_id",
        },
        {
            "from": "accidents",
            "to": "hospitals",
            "type": "spatial",
            "attributes": "closest point",
        },
    ]


def _prepend_hardcoded_first_scenario1_path(candidate_paths, data_lake):
    demo = list(HARDCODED_FIRST_SCENARIO1_PATH)
    if not all(t in data_lake for t in demo):
        return list(candidate_paths)
    normalized = [list(p) for p in candidate_paths]
    rest = [p for p in normalized if p != demo]
    return [demo] + rest


def _resolve_scenario1_path_edges(path, rel_e, sem_e, spa_e, allow_spatial):
    if tuple(path) == HARDCODED_FIRST_SCENARIO1_PATH:
        return hardcoded_first_scenario1_path_edges()
    return join_edges_for_path_ordered(
        list(path), rel_e, sem_e, spa_e, allow_spatial=allow_spatial
    )


def _spatial_nearest_join(left_df, right_df, right_cols, spatial_distance_km=None):
    left_lats, left_lons = _get_lat_lon_series(left_df)
    right_lats, right_lons = _get_lat_lon_series(right_df)
    if left_lats is None or right_lats is None:
        return None

    right_points = []
    for idx in right_df.index:
        lat = right_lats.loc[idx]
        lon = right_lons.loc[idx]
        if pd.isna(lat) or pd.isna(lon):
            continue
        right_points.append((idx, float(lat), float(lon)))

    if not right_points:
        return None

    rows = []
    for idx in left_df.index:
        lat = left_lats.loc[idx]
        lon = left_lons.loc[idx]
        if pd.isna(lat) or pd.isna(lon):
            rows.append([None] * len(right_cols))
            continue
        best_idx = None
        best_dist = None
        for r_idx, r_lat, r_lon in right_points:
            dist = _haversine_km(lat, lon, r_lat, r_lon)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = r_idx
        if best_idx is None or (spatial_distance_km is not None and best_dist > spatial_distance_km):
            rows.append([None] * len(right_cols))
        else:
            rows.append(right_df.loc[best_idx, right_cols].tolist())

    right_joined = pd.DataFrame(rows, columns=right_cols, index=left_df.index)
    return pd.concat([left_df, right_joined], axis=1)


def _data_lake_has_spatial(data_lake):
    for _, df in data_lake.items():
        lat_col, lon_col = _find_lat_lon_columns(df)
        if lat_col and lon_col:
            return True
        if graph_joins._find_spatial_columns(df):
            return True
    return False


def _table_has_spatial(df):
    """Check if a table has spatial attributes (lat/lon columns or spatial columns)."""
    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        return True
    if graph_joins._find_spatial_columns(df):
        return True
    return False


def _collect_map_points(df, bbox=None, limit=200):
    points = []
    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        if bbox:
            mask = lats.between(bbox[0], bbox[2]) & lons.between(bbox[1], bbox[3])
            lats = lats[mask]
            lons = lons[mask]
        for lat, lon in zip(lats.head(limit), lons.head(limit)):
            if pd.notna(lat) and pd.notna(lon):
                points.append((float(lat), float(lon)))
        return points

    spatial_cols = graph_joins._find_spatial_columns(df)
    for col in spatial_cols:
        sample = df[col].dropna().astype(str).head(limit * 2)
        for value in sample:
            coords = extract_point_coords(value)
            if coords:
                lat, lon = coords
                if bbox and not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
                    continue
                points.append((lat, lon))
                if len(points) >= limit:
                    return points
    return points


def _extract_points_from_path(path_tables, data_lake, max_points_per_table=200):
    """
    Extract all point coordinates from all tables in a path.
    Returns: list of (lat, lon, table_name) tuples
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
                coords = extract_point_coords(val)
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


def _filter_preview_table_by_keyword(df, keyword):
    query = (keyword or "").strip()
    if not query or df.empty:
        return df

    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(
            query, case=False, na=False, regex=False
        )
    return df[mask]


USER_BBOX_COLOR = "#2e7d32"
USER_BBOX_FILL = "#66bb6a"
PATH_OVERLAY_COLOR = "#1565c0"
PATH_OVERLAY_FILL = "#42a5f5"
SCENARIO1_MAP_HEIGHT = 560
# Match the map block so the action button sits on the same baseline.
SCENARIO1_RIGHT_PANEL_HEIGHT = SCENARIO1_MAP_HEIGHT + 72


def _merge_bboxes(*bboxes):
    valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
    if not valid:
        return None
    south = min(bbox[0] for bbox in valid)
    west = min(bbox[1] for bbox in valid)
    north = max(bbox[2] for bbox in valid)
    east = max(bbox[3] for bbox in valid)
    return [south, west, north, east]


def _pad_bbox(bbox, pad_ratio=0.08, min_pad=0.01):
    if not bbox or len(bbox) != 4:
        return bbox
    south, west, north, east = bbox
    lat_span = max(north - south, min_pad)
    lon_span = max(east - west, min_pad)
    lat_pad = max(lat_span * pad_ratio, min_pad)
    lon_pad = max(lon_span * pad_ratio, min_pad)
    return [
        south - lat_pad,
        west - lon_pad,
        north + lat_pad,
        east + lon_pad,
    ]


def _get_join_info_user_query(table1, table2):
    for edge in st.session_state.get("generated_join_path_edges", []):
        if edge["from"] == table1 and edge["to"] == table2:
            return {"type": edge["type"], "attributes": edge["attributes"]}
        if edge["from"] == table2 and edge["to"] == table1:
            return {"type": edge["type"], "attributes": edge["attributes"]}
    for left, right, label in st.session_state.get("graph_semantic_edges", []):
        if (left == table1 and right == table2) or (left == table2 and right == table1):
            return {"type": "semantic", "attributes": label}
    for left, right, label in st.session_state.get("graph_rel_edges", []):
        if (left == table1 and right == table2) or (left == table2 and right == table1):
            return {"type": "relation", "attributes": label}
    for left, right, label in st.session_state.get("graph_spatial_edges", []):
        if (left == table1 and right == table2) or (left == table2 and right == table1):
            return {"type": "spatial", "attributes": label}
    return {"type": "relation", "attributes": ""}


def _build_path_map_selection(path_tables, data_lake, fallback_bbox):
    region_info = _find_region_from_tables(path_tables, data_lake)
    if region_info and region_info.get("bbox"):
        return region_info

    computed_bbox = _compute_bbox_from_path(path_tables, data_lake)
    if computed_bbox:
        return {
            "bbox": computed_bbox,
            "region_name": "Computed from path tables",
            "method": "spatial_coordinates",
        }

    if fallback_bbox:
        return {
            "bbox": fallback_bbox,
            "region_name": "Path Region",
            "method": "default_coordinates",
        }

    return {}


def _sync_user_query_path_state():
    path_tables = st.session_state.get("generated_join_path_tables", [])
    path_tables_list = st.session_state.get("generated_join_paths_tables", [])

    if path_tables_list:
        st.session_state["user_query_paths"] = [
            " → ".join(path) for path in path_tables_list
        ]
        st.session_state["user_query_parsed_paths"] = [list(path) for path in path_tables_list]
    elif path_tables:
        path_str = " → ".join(path_tables)
        st.session_state["user_query_paths"] = [path_str]
        st.session_state["user_query_parsed_paths"] = [path_tables]
    else:
        st.session_state["user_query_paths"] = []
        st.session_state["user_query_parsed_paths"] = []

    if "user_query_path_map_selections" not in st.session_state:
        st.session_state["user_query_path_map_selections"] = {}
    if "selected_user_query_path_for_map" not in st.session_state:
        st.session_state["selected_user_query_path_for_map"] = None
    if "current_user_query_path_index" not in st.session_state:
        st.session_state["current_user_query_path_index"] = 1

    parsed_paths = st.session_state.get("user_query_parsed_paths", [])
    active_data_lake = st.session_state.get(
        "user_query_active_data_lake",
        st.session_state.data_lake,
    )
    return parsed_paths, active_data_lake


def _store_selected_user_query_path_for_map(
    path_idx,
    parsed_paths,
    active_data_lake,
    fallback_bbox,
):
    path_sel = parsed_paths[path_idx - 1] if 0 < path_idx <= len(parsed_paths) else []
    info = _build_path_map_selection(path_sel, active_data_lake, fallback_bbox)
    st.session_state["user_query_path_map_selections"][
        f"user_query_path_{path_idx}"
    ] = info
    st.session_state["selected_user_query_path_for_map"] = path_idx
    return info


def _render_scenario1_map_panel(parsed_paths, active_data_lake):
    import leafmap.foliumap as leafmap

    st.markdown("**Step 1: Please select your area of interest**")
    st.caption(
        "Draw a bounding box on the map. Green appears after you draw your area; blue shows the active join path."
    )

    existing = st.session_state.get("bbox_selection")
    selected_path_idx = st.session_state.get("selected_user_query_path_for_map")
    path_region_info = None
    path_map_bbox = None

    if selected_path_idx and parsed_paths:
        stored_info = _store_selected_user_query_path_for_map(
            selected_path_idx,
            parsed_paths,
            active_data_lake,
            existing,
        )
        if stored_info:
            path_region_info = stored_info
            path_map_bbox = stored_info.get("bbox")
    elif selected_path_idx:
        stored_info = st.session_state.get("user_query_path_map_selections", {}).get(
            f"user_query_path_{selected_path_idx}"
        )
        if stored_info:
            path_region_info = stored_info
            path_map_bbox = stored_info.get("bbox")

    viewport_bbox = _merge_bboxes(existing, path_map_bbox) or path_map_bbox or DEFAULT_BBOX_SELECTION
    padded_bbox = _pad_bbox(viewport_bbox)
    center_lat = (padded_bbox[0] + padded_bbox[2]) / 2
    center_lon = (padded_bbox[1] + padded_bbox[3]) / 2

    m = leafmap.Map(
        center=[center_lat, center_lon],
        zoom=6,
        draw_control=True,
        measure_control=False,
        fullscreen_control=False,
        locate_control=False,
    )
    m.add_basemap("OpenStreetMap")
    if padded_bbox:
        m.fit_bounds([[padded_bbox[0], padded_bbox[1]], [padded_bbox[2], padded_bbox[3]]])

    if existing:
        folium.Rectangle(
            bounds=[[existing[0], existing[1]], [existing[2], existing[3]]],
            color=USER_BBOX_COLOR,
            fillColor=USER_BBOX_FILL,
            fillOpacity=0.18,
            weight=2,
            opacity=0.85,
            tooltip="Selected area of interest",
        ).add_to(m)

    if path_map_bbox is not None:
        folium.Rectangle(
            bounds=[[path_map_bbox[0], path_map_bbox[1]], [path_map_bbox[2], path_map_bbox[3]]],
            color=PATH_OVERLAY_COLOR,
            fillColor=PATH_OVERLAY_FILL,
            fillOpacity=0.14,
            weight=2,
            opacity=0.85,
            tooltip=f"Join Path {selected_path_idx}",
        ).add_to(m)

        path = parsed_paths[selected_path_idx - 1] if 0 < selected_path_idx <= len(parsed_paths) else []
        points = _extract_points_from_path(path, active_data_lake, max_points_per_table=200)
        if existing:
            margin = 0.01
            points = [
                (lat, lon, table_name)
                for lat, lon, table_name in points
                if existing[0] - margin <= lat <= existing[2] + margin
                and existing[1] - margin <= lon <= existing[3] + margin
            ]
        for lat, lon, table_name in points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                popup=f"{table_name}<br>({lat:.4f}, {lon:.4f})",
                tooltip=table_name,
                color=PATH_OVERLAY_COLOR,
                fillColor=PATH_OVERLAY_FILL,
                fillOpacity=0.75,
                weight=1,
            ).add_to(m)

    st_map = m.to_streamlit(height=SCENARIO1_MAP_HEIGHT, bidirectional=True)
    new_bbox = None
    if st_map:
        last_draw = m.st_last_draw(st_map)
        new_bbox = bbox_from_geojson_feature(last_draw)
        if not new_bbox:
            drawings = m.st_draw_features(st_map) or []
            for feat in reversed(drawings):
                new_bbox = bbox_from_geojson_feature(feat)
                if new_bbox:
                    break

    if new_bbox:
        st.session_state["bbox_selection"] = new_bbox
        bbox_label = (
            f"{new_bbox[0]:.4f}, {new_bbox[1]:.4f}, "
            f"{new_bbox[2]:.4f}, {new_bbox[3]:.4f}"
        )
        st.success(f"Selected rectangle: {bbox_label}")
    elif existing:
        bbox_label = (
            f"{existing[0]:.4f}, {existing[1]:.4f}, "
            f"{existing[2]:.4f}, {existing[3]:.4f}"
        )
        st.caption(f"Current bbox: {bbox_label} (draw a rectangle to update)")

    if selected_path_idx and path_region_info and path_map_bbox is not None:
        region_name = path_region_info.get("region_name", "Unknown")
        method = path_region_info.get("method", "unknown")
        st.caption(
            f"Showing Join Path {selected_path_idx} in blue. Region: {region_name} ({method})."
        )
    # else:
    #     st.caption("Click **Show on the map** for a join path to overlay its region and points here.")


def _render_scenario1_target_table(current_path_idx):
    preview = st.session_state.get("scenario1_path_tables_data", {}).get(current_path_idx)
    if not preview:
        return

    st.markdown("**Target table**")
    if preview.get("note"):
        st.caption(preview["note"])

    tuples_list = preview.get("tuples") or []
    preview_cols = preview.get("col_order") or []
    if tuples_list and preview_cols:
        prev_df = pd.DataFrame(tuples_list, columns=preview_cols)
        search_query = st.text_input(
            "Search target table",
            key=f"target_table_search_{current_path_idx}",
            placeholder="Type a keyword to filter rows",
        )
        filtered_prev_df = _filter_preview_table_by_keyword(prev_df, search_query)
        if search_query.strip():
            st.caption(
                f"{len(filtered_prev_df)} matching row(s) for "
                f'"{search_query.strip()}".'
            )
        pt_html = filtered_prev_df.to_html(
            index=False, table_id=f"target_table_path_{current_path_idx}"
        )
        scroll = '<div style="max-height: 280px; overflow-y: auto;">' f"{pt_html}</div>"
        st.markdown(scroll, unsafe_allow_html=True)
        if search_query.strip() and filtered_prev_df.empty:
            st.caption("No rows match the current keyword search.")
    elif preview_cols:
        st.dataframe(
            pd.DataFrame(columns=preview_cols),
            width="stretch",
        )
        st.caption("No rows after joining along this path.")
    else:
        st.caption("No columns to display.")


def _render_scenario1_join_path_panel(parsed_paths, active_data_lake, fallback_bbox):
    if not parsed_paths:
        return None

    table_metadata = build_table_metadata(active_data_lake)
    highlighted_tables = {
        table_name
        for matches in st.session_state.get("generated_target_schema", {}).values()
        for table_name, _ in matches
    }

    st.markdown("**Generated Join Paths**")
    render_join_path_legend(
        rows_hint="ROWS and COLS populate after you click Show target table for a path."
    )

    current_path_idx = st.session_state.get("current_user_query_path_index", 1)
    current_path_idx = max(1, min(current_path_idx, len(parsed_paths)))

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button(
            "Previous join path",
            key="prev_user_query_join_path",
            disabled=len(parsed_paths) <= 1,
            width="stretch",
        ):
            current_path_idx = ((current_path_idx - 2) % len(parsed_paths)) + 1
            st.session_state["selected_user_query_path_for_map"] = None
    with nav_cols[1]:
        if st.button(
            "Next join path",
            key="next_user_query_join_path",
            disabled=len(parsed_paths) <= 1,
            width="stretch",
        ):
            current_path_idx = (current_path_idx % len(parsed_paths)) + 1
            st.session_state["selected_user_query_path_for_map"] = None

    st.session_state["current_user_query_path_index"] = current_path_idx

    idx = current_path_idx
    path = parsed_paths[idx - 1]
    path_data = {
        "tables": path,
        "joins": [],
    }
    for i in range(len(path) - 1):
        join_info = _get_join_info_user_query(path[i], path[i + 1])
        path_data["joins"].append(
            {
                "from": path[i],
                "to": path[i + 1],
                "attributes": join_info["attributes"],
                "type": join_info["type"],
            }
        )

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(
            "Show on the map",
            key=f"show_user_query_path_{idx}",
            width="stretch",
        ):
            _store_selected_user_query_path_for_map(
                idx,
                parsed_paths,
                active_data_lake,
                fallback_bbox,
            )

    with action_cols[1]:
        if st.button(
            "Show target table",
            key=f"show_target_table_path_{idx}",
            width="stretch",
        ):
            matched = st.session_state.get("generated_target_schema")
            if not matched:
                st.warning("Click **Show the join paths** first.")
            else:
                filtered = st.session_state.get("scenario1_filtered_data_lake")
                if filtered is None:
                    filtered = st.session_state.data_lake
                allow_sp = st.session_state.get("generated_spatial_mode") is not None
                if tuple(path) == HARDCODED_FIRST_SCENARIO1_PATH:
                    allow_sp = True
                rel_e = st.session_state.get("graph_rel_edges", [])
                sem_e = st.session_state.get("graph_semantic_edges", [])
                spa_e = st.session_state.get("graph_spatial_edges", [])
                path_edges = _resolve_scenario1_path_edges(
                    list(path),
                    rel_e,
                    sem_e,
                    spa_e,
                    allow_spatial=allow_sp,
                )
                if path_edges is None:
                    st.error(
                        "Could not resolve join edges for consecutive tables on this path."
                    )
                else:
                    if tuple(path) == HARDCODED_FIRST_SCENARIO1_PATH:
                        spatial_mode_run = "distance"
                        spatial_dist_run = None
                    else:
                        spatial_mode_run = st.session_state.get("generated_spatial_mode")
                        spatial_dist_run = st.session_state.get(
                            "generated_spatial_distance_km"
                        )
                    tuples, col_order = generate_joined_tuples(
                        matched,
                        filtered,
                        join_path=list(path),
                        join_edges=path_edges,
                        spatial_mode=spatial_mode_run,
                        spatial_distance_km=spatial_dist_run,
                    )
                    preview_note = None
                    if (
                        not tuples
                        and st.session_state.get("scenario1_filtered_data_lake") is not None
                    ):
                        tuples, col_order = generate_joined_tuples(
                            matched,
                            st.session_state.data_lake,
                            join_path=list(path),
                            join_edges=path_edges,
                            spatial_mode=spatial_mode_run,
                            spatial_distance_km=spatial_dist_run,
                        )
                        if tuples:
                            preview_note = (
                                "Your **bounding box** did not overlap any coordinates in the "
                                "filtered tables, so every table was empty after filtering. "
                                "Results below use the **full data lake** (no bbox filter)."
                            )
                    st.session_state.setdefault("scenario1_path_tables_data", {})
                    st.session_state["scenario1_path_tables_data"][idx] = {
                        "tuples": tuples,
                        "col_order": col_order,
                        "note": preview_note,
                    }
                    if not tuples:
                        st.info(
                            "**No rows** after inner-joining along this path. Check keys, "
                            "spatial columns, and your bounding box."
                        )

    preview = st.session_state.get("scenario1_path_tables_data", {}).get(idx)
    preview_rows = None
    preview_cols = None
    if preview is not None:
        preview_rows = len(preview.get("tuples") or [])
        col_order = preview.get("col_order") or []
        preview_cols = len(col_order) if col_order else None

    render_join_path_card(
        component_id=f"user_query_path_{idx}_card",
        title=f"Join Path {idx:03d}",
        path_data=path_data,
        table_metadata=table_metadata,
        highlighted_tables=highlighted_tables,
        stats={
            "rows": preview_rows,
            "cols": preview_cols,
            "len": len(path),
            "hops": max(len(path) - 1, 0),
        },
        selected=True,
    )

    return idx


# US State bounding boxes [south, west, north, east]
US_STATE_BBOXES = {
    "indiana": [37.7717, -88.0976, 41.7606, -84.7846],
    "illinois": [36.9703, -91.5131, 42.5083, -87.4948],
    "ohio": [38.4020, -84.8203, 41.9775, -80.5187],
    "kentucky": [36.4970, -89.5715, 39.1477, -81.9644],
    "michigan": [41.6961, -90.4184, 48.2388, -82.1229],
    "wisconsin": [42.4919, -92.8881, 47.0806, -86.2495],
    "california": [32.5121, -124.4096, 42.0126, -114.1312],
    "texas": [25.8371, -106.6456, 36.5007, -93.5083],
    "new york": [40.4774, -79.7624, 45.0159, -71.8562],
    "florida": [24.5210, -87.6349, 31.0009, -79.9743],
}

US_STATE_ALIASES = {
    "in": "indiana",
    "indiana": "indiana",
    "il": "illinois",
    "illinois": "illinois",
    "oh": "ohio",
    "ohio": "ohio",
    "ky": "kentucky",
    "kentucky": "kentucky",
    "mi": "michigan",
    "michigan": "michigan",
    "wi": "wisconsin",
    "wisconsin": "wisconsin",
    "ca": "california",
    "california": "california",
    "tx": "texas",
    "texas": "texas",
    "ny": "new york",
    "new york": "new york",
    "fl": "florida",
    "florida": "florida",
}

COUNTRY_DISPLAY_NAMES = {
    "us": "United States",
}

COUNTRY_DISPLAY_KEYS = {
    display_name: country_key
    for country_key, display_name in COUNTRY_DISPLAY_NAMES.items()
}


def render_user_query_section(west_lafayette_bbox, lafayette_default_bbox):
    import leafmap.foliumap as leafmap

    st.markdown('<div id="user-query"></div>', unsafe_allow_html=True)
    st.subheader("Location-aware Discovery & Augmentation")
    if "graph_rel_edges" not in st.session_state:
        st.session_state["graph_rel_edges"] = []
    if "graph_spatial_edges" not in st.session_state:
        st.session_state["graph_spatial_edges"] = []
    if "target_schema_raw" not in st.session_state:
        st.session_state["target_schema_raw"] = []
    
    scenario_discovery_key = "scenario_1_discovery_integration"
    scenario_augmentation_key = "scenario_2_augmentation"

    selected_scenario_key = st.session_state.get("user_query_scenario_key")
    if selected_scenario_key not in {scenario_discovery_key, scenario_augmentation_key}:
        legacy_scenario = st.session_state.get("user_query_scenario", "")
        if isinstance(legacy_scenario, str) and "Augmentation" in legacy_scenario:
            selected_scenario_key = scenario_augmentation_key
        else:
            selected_scenario_key = scenario_discovery_key
        st.session_state.user_query_scenario_key = selected_scenario_key

    story_col1, story_col2 = st.columns(2)
    with story_col1:
        st.markdown(
            "**Scenario 1.** Alice is investigating hit-and-run incidents in a selected area of Chicago and wants to discover and combine the right data lake tables into a single dataset containing `accident_id`, `date`, `victim_type`, `report_id`, `status`, and the nearest hospital name for each case."
        )
    with story_col2:
        st.markdown(
            "**Scenario 2.** Alice starts with a housing listings table and wants to enrich each listing with nearby attributes from other data lake tables, using a spatial range to determine which records should be appended."
        )
    st.divider()

    # Scenario 2: Location-aware Data Augmentation
    if selected_scenario_key == scenario_augmentation_key:
        if st.session_state.get("user_uploaded_table_df") is None:
            st.info("Upload a CSV table from the sidebar to run Location-aware Data Augmentation.")
            return

        uploaded_df = st.session_state.user_uploaded_table_df
        uploaded_table_name = st.session_state.get("user_uploaded_table_name", "uploaded_table")
        
        st.markdown("**Uploaded Table:**")
        st.dataframe(uploaded_df, width="stretch", height=300)
        
        st.markdown("**Step 1: Please select your area of interest (draw a bounding box on the map)**")
        existing = st.session_state.get("bbox_selection", DEFAULT_BBOX_SELECTION)
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
            new_bbox = bbox_from_geojson_feature(last_draw)
            if not new_bbox:
                drawings = m.st_draw_features(st_map) or []
                for feat in reversed(drawings):
                    new_bbox = bbox_from_geojson_feature(feat)
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
        
        st.markdown("**Attributes to append:**")
        if st.session_state.target_schema:
            attrs = st.session_state.target_schema.copy()
            to_remove = []
            for attr in attrs:
                row = st.columns([1, 20])
                with row[0]:
                    if st.button("✕", key=f"scenario2_remove_attr_{attr}", help="Remove", width="stretch"):
                        to_remove.append(attr)
                with row[1]:
                    st.markdown(
                        f"""
                        <div style="padding:6px 10px;border:1px solid #d0d7de;border-radius:6px;font-size:0.95rem;background:#f7f9fc;display:inline-flex;align-items:center;white-space:nowrap;height:32px;line-height:1;">
                          {attr}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            if to_remove:
                remaining = []
                remaining_raw = []
                for idx, value in enumerate(st.session_state.target_schema):
                    if value in to_remove:
                        continue
                    remaining.append(value)
                    if idx < len(st.session_state.target_schema_raw):
                        remaining_raw.append(st.session_state.target_schema_raw[idx])
                st.session_state.target_schema = remaining
                st.session_state.target_schema_raw = remaining_raw
                st.rerun()
        else:
            st.write("*No attributes selected yet. Please add attributes in the sidebar.*")
        
        # Initialize session state for scenario 2 results
        if "scenario2_augmented_tables" not in st.session_state:
            st.session_state["scenario2_augmented_tables"] = []
        if "scenario2_join_paths" not in st.session_state:
            st.session_state["scenario2_join_paths"] = []
        
        # Generate augmented table button
        if st.button("Generate Augmented Table", key="scenario2_generate_button"):
            if not st.session_state.target_schema:
                st.warning("Please add at least one target attribute first.")
            else:
                # Add uploaded table to data lake temporarily for join path finding
                temp_data_lake = st.session_state.data_lake.copy()
                temp_data_lake[uploaded_table_name] = uploaded_df
                
                # Match attributes
                matched_attributes = {}
                unmatched = []
                for user_input in st.session_state.target_schema:
                    matches = fuzzy_match_attribute(user_input, temp_data_lake)
                    if matches:
                        matched_attributes[user_input] = matches
                    else:
                        unmatched.append(user_input)
                
                if unmatched:
                    st.warning(f"Could not match the following attributes: {', '.join(unmatched)}")
                
                if matched_attributes:
                    # Get spatial preferences
                    spatial_valid = True
                    if st.session_state.get("spatial_choice_mode", "inferred") == "inferred":
                        inferred_inputs = st.session_state.target_schema_raw or st.session_state.target_schema
                        inferred = infer_spatial_preferences(inferred_inputs)
                        spatial_mode = inferred["mode"]
                        spatial_distance_km = inferred["distance_km"]
                        spatial_source = "inferred"
                    else:
                        manual_mode = st.session_state.get("spatial_manual_mode", "exclude")
                        if manual_mode == "exclude":
                            spatial_mode = None
                            spatial_distance_km = None
                            spatial_source = "manual"
                        elif manual_mode == "distance":
                            distance_km = st.session_state.get("spatial_manual_distance_km")
                            if not distance_km or distance_km <= 0:
                                st.warning("Please enter a positive distance (km) for the spatial predicate.")
                                spatial_valid = False
                            spatial_mode = "distance"
                            spatial_distance_km = distance_km
                            spatial_source = "manual"
                        else:
                            spatial_mode = manual_mode
                            spatial_distance_km = None
                            spatial_source = "manual"
                    
                    if not spatial_valid:
                        return
                    
                    # Get edges
                    joinability_threshold = st.session_state.get("graph_equi_joinability_measure", 0.0)
                    rel_edges = graph_joins.find_relational_joins(temp_data_lake)
                    semantic_edges = graph_joins.find_semantic_joins(
                        temp_data_lake,
                        min_overlap=joinability_threshold,
                    )
                    spatial_edges = graph_joins.find_spatial_joins(temp_data_lake)
                    
                    allow_spatial = spatial_mode is not None
                    require_spatial = spatial_mode is not None

                    uploaded_tokens = set()
                    for col in uploaded_df.columns:
                        parts = re.split(r"[^a-zA-Z0-9]+", str(col).lower())
                        for part in parts:
                            if len(part) >= 3:
                                uploaded_tokens.add(part)
                    table_priorities = {}
                    for table_name, df in temp_data_lake.items():
                        if table_name == uploaded_table_name:
                            table_priorities[table_name] = 0
                            continue
                        name_lower = table_name.lower()
                        if any(token in name_lower for token in uploaded_tokens):
                            table_priorities[table_name] = 1
                            continue
                        shared_cols = set(uploaded_df.columns) & set(df.columns)
                        if shared_cols:
                            table_priorities[table_name] = 2
                        else:
                            table_priorities[table_name] = 5
                    
                    # Find join paths
                    path_info = find_min_join_path(
                        matched_attributes,
                        rel_edges,
                        semantic_edges,
                        spatial_edges,
                        allow_spatial=allow_spatial,
                        require_spatial=require_spatial,
                        max_tables=len(temp_data_lake),
                        required_tables=[uploaded_table_name],
                        table_priorities=table_priorities,
                    )
                    
                    if path_info:
                        # Ensure the path starts with the uploaded table
                        path_tables = path_info["tables"].copy()
                        
                        # Generate augmented table
                        bbox = st.session_state.get("bbox_selection")
                        filtered_data_lake = temp_data_lake
                        
                        if bbox:
                            # Filter by bbox if spatial data exists
                            filtered_data_lake = {}
                            for name, df in temp_data_lake.items():
                                filtered_df, _ = _filter_table_by_bbox(df, bbox)
                                filtered_data_lake[name] = filtered_df
                        
                        # Generate augmented table by joining along the path
                        # Start with uploaded table (preserve all original columns)
                        result_df = filtered_data_lake.get(uploaded_table_name, uploaded_df).copy()
                        
                        # Get columns to add from matched attributes (only from tables in path, not uploaded table)
                        # Map user_input -> (table_name, col_name) for renaming later
                        user_input_to_column = {}  # Maps user_input -> actual (table_name, col_name)
                        columns_to_add = {}  # Maps table_name -> list of col_names to add
                        for user_input, matches in matched_attributes.items():
                            # Use the first match for each user input
                            for table_name, col_name in matches:
                                if table_name in path_tables and table_name != uploaded_table_name:
                                    user_input_to_column[user_input] = (table_name, col_name)
                                    if table_name not in columns_to_add:
                                        columns_to_add[table_name] = []
                                    if col_name not in columns_to_add[table_name]:
                                        columns_to_add[table_name].append(col_name)
                                    break  # Use first match
                        
                        # Store original uploaded columns to ensure they're preserved
                        original_uploaded_columns = list(result_df.columns)

                        def _get_edge_between(path_info, left, right):
                            for edge in path_info.get("edges", []):
                                if (edge.get("from") == left and edge.get("to") == right) or \
                                   (edge.get("from") == right and edge.get("to") == left):
                                    return edge
                            return None

                        link_cols_by_table = {table: set() for table in path_tables}
                        for i in range(1, len(path_tables)):
                            left = path_tables[i - 1]
                            right = path_tables[i]
                            edge = _get_edge_between(path_info, left, right)
                            edge_type = edge.get("type") if edge else None
                            if edge_type == "spatial":
                                continue
                            left_df = filtered_data_lake.get(left)
                            right_df = filtered_data_lake.get(right)
                            if left_df is None or right_df is None:
                                continue
                            if edge_type == "semantic":
                                semantic_pairs = semantic_joins.resolve_semantic_pairs(
                                    edge.get("attributes", ""),
                                    left_df,
                                    right_df,
                                )
                                if not semantic_pairs:
                                    semantic_pairs = [
                                        (match.left_col, match.right_col)
                                        for match in semantic_joins.find_semantic_column_matches(
                                            left_df,
                                            right_df,
                                        )
                                    ]
                                if semantic_pairs:
                                    link_cols_by_table[left].update(left_col for left_col, _ in semantic_pairs)
                                    link_cols_by_table[right].update(right_col for _, right_col in semantic_pairs)
                                    continue
                            attrs = []
                            if edge and edge.get("attributes"):
                                attrs = [
                                    attr.strip()
                                    for attr in str(edge["attributes"]).replace("…", "").split(",")
                                    if attr.strip() and "≈" not in attr
                                ]
                            shared_attrs = [
                                attr for attr in attrs
                                if attr in left_df.columns and attr in right_df.columns
                            ]
                            if shared_attrs:
                                link_cols_by_table[left].update(shared_attrs)
                                link_cols_by_table[right].update(shared_attrs)
                                continue
                            common = set(left_df.columns) & set(right_df.columns)
                            link_cols_by_table[left].update(common)
                            link_cols_by_table[right].update(common)

                        def _apply_join(result_df, prev_table, table_name):
                            if table_name not in filtered_data_lake:
                                return result_df
                            next_df = filtered_data_lake[table_name]
                            edge = _get_edge_between(path_info, prev_table, table_name)
                            edge_type = edge.get("type") if edge else None
                            requested_cols = columns_to_add.get(table_name, [])
                            link_cols = sorted(link_cols_by_table.get(table_name, set()))
                            requested_cols = [
                                col for col in (requested_cols + link_cols)
                                if col in next_df.columns and col not in result_df.columns
                            ]

                            if edge_type == "spatial":
                                if requested_cols:
                                    spatial_distance = None
                                    if spatial_mode == "distance":
                                        spatial_distance = spatial_distance_km
                                    spatial_joined = _spatial_nearest_join(
                                        result_df,
                                        next_df,
                                        requested_cols,
                                        spatial_distance_km=spatial_distance,
                                    )
                                    if spatial_joined is not None:
                                        return spatial_joined
                                return result_df

                            if edge_type == "semantic":
                                semantic_pairs = semantic_joins.resolve_semantic_pairs(
                                    edge.get("attributes", ""),
                                    result_df,
                                    next_df,
                                )
                                if not semantic_pairs:
                                    semantic_pairs = [
                                        (match.left_col, match.right_col)
                                        for match in semantic_joins.find_semantic_column_matches(
                                            result_df,
                                            next_df,
                                        )
                                    ]
                                if semantic_pairs:
                                    cols_to_include = []
                                    for _, right_col in semantic_pairs:
                                        if right_col not in result_df.columns:
                                            cols_to_include.append(right_col)
                                    for col in requested_cols:
                                        if col not in cols_to_include:
                                            cols_to_include.append(col)
                                    semantic_joined = semantic_joins.semantic_merge(
                                        result_df,
                                        next_df,
                                        semantic_pairs,
                                        how='left',
                                        right_columns=cols_to_include,
                                        suffixes=('', '_dup'),
                                    )
                                    if semantic_joined is not None:
                                        return semantic_joined

                            # Relational join: try edge attributes then common columns.
                            relation_candidates = []
                            if edge and edge.get("attributes"):
                                relation_candidates = [
                                    attr.strip()
                                    for attr in str(edge["attributes"]).replace("…", "").split(",")
                                    if attr.strip() and "≈" not in attr
                                ]
                                relation_candidates = [
                                    attr for attr in relation_candidates
                                    if attr in result_df.columns and attr in next_df.columns
                                ]

                            if not relation_candidates:
                                relation_candidates = sorted(set(result_df.columns) & set(next_df.columns))

                            join_col = relation_candidates[0] if relation_candidates else None
                            if join_col:
                                left_spatial_cols = set(graph_joins._find_spatial_columns(result_df))
                                right_spatial_cols = set(graph_joins._find_spatial_columns(next_df))
                                if join_col in left_spatial_cols and join_col in right_spatial_cols:
                                    def _normalize_point_series(series):
                                        coords = series.astype(str).str.extract(
                                            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
                                            flags=re.IGNORECASE,
                                        )
                                        if coords.empty:
                                            return None
                                        lons = pd.to_numeric(coords[0], errors="coerce")
                                        lats = pd.to_numeric(coords[1], errors="coerce")
                                        return lons.round(4).astype(str) + "," + lats.round(4).astype(str)

                                    left_norm = _normalize_point_series(result_df[join_col])
                                    right_norm = _normalize_point_series(next_df[join_col])
                                    if left_norm is not None and right_norm is not None:
                                        left_key = "__join_location_norm__"
                                        right_key = "__join_location_norm__"
                                        left_tmp = result_df.copy()
                                        right_tmp = next_df.copy()
                                        left_tmp[left_key] = left_norm
                                        right_tmp[right_key] = right_norm
                                        cols_to_include = [right_key]
                                        for col in requested_cols:
                                            if col not in cols_to_include:
                                                cols_to_include.append(col)
                                        return pd.merge(
                                            left_tmp,
                                            right_tmp[cols_to_include],
                                            left_on=left_key,
                                            right_on=right_key,
                                            how='left',
                                            suffixes=('', '_dup')
                                        ).drop(columns=[left_key, right_key], errors="ignore")

                                same_col_semantic = semantic_joins.find_semantic_column_matches(
                                    result_df[[join_col]],
                                    next_df[[join_col]],
                                )
                                if any(
                                    match.left_col == join_col and match.right_col == join_col
                                    for match in same_col_semantic
                                ):
                                    cols_to_include = [
                                        col for col in requested_cols
                                        if col not in result_df.columns
                                    ]
                                    semantic_joined = semantic_joins.semantic_merge(
                                        result_df,
                                        next_df,
                                        [(join_col, join_col)],
                                        how='left',
                                        right_columns=cols_to_include,
                                        suffixes=('', '_dup'),
                                    )
                                    if semantic_joined is not None:
                                        return semantic_joined

                                cols_to_include = [join_col]
                                for col in requested_cols:
                                    if col not in cols_to_include:
                                        cols_to_include.append(col)
                                cols_to_include = [
                                    col for col in cols_to_include
                                    if col not in result_df.columns or col == join_col
                                ]
                                if len(cols_to_include) > 1:
                                    return pd.merge(
                                        result_df,
                                        next_df[cols_to_include],
                                        on=join_col,
                                        how='left',
                                        suffixes=('', '_dup')
                                    )

                            # Fallback to spatial if no relational join is possible.
                            if requested_cols and _table_has_spatial(result_df) and _table_has_spatial(next_df):
                                spatial_distance = None
                                if spatial_mode == "distance":
                                    spatial_distance = spatial_distance_km
                                spatial_joined = _spatial_nearest_join(
                                    result_df,
                                    next_df,
                                    requested_cols,
                                    spatial_distance_km=spatial_distance,
                                )
                                if spatial_joined is not None:
                                    return spatial_joined

                            return result_df
                            next_df = filtered_data_lake[table_name]
                            edge = _get_edge_between(path_info, prev_table, table_name)
                            edge_type = edge.get("type") if edge else None
                            requested_cols = columns_to_add.get(table_name, [])
                            link_cols = sorted(link_cols_by_table.get(table_name, set()))
                            requested_cols = [
                                col for col in (requested_cols + link_cols)
                                if col in next_df.columns and col not in result_df.columns
                            ]

                            if edge_type == "spatial":
                                if requested_cols:
                                    spatial_distance = None
                                    if spatial_mode == "distance":
                                        spatial_distance = spatial_distance_km
                                    spatial_joined = _spatial_nearest_join(
                                        result_df,
                                        next_df,
                                        requested_cols,
                                        spatial_distance_km=spatial_distance,
                                    )
                                    if spatial_joined is not None:
                                        return spatial_joined
                                return result_df

                            # Relational join: try edge attributes then common columns.
                            join_col = None
                            if edge and edge.get("attributes"):
                                attrs_str = edge["attributes"]
                                if isinstance(attrs_str, str):
                                    attrs = [a.strip() for a in attrs_str.split(",")]
                                else:
                                    attrs = [attrs_str]
                                for attr in attrs:
                                    if attr in result_df.columns and attr in next_df.columns:
                                        join_col = attr
                                        break

                            if not join_col:
                                common_cols = list(set(result_df.columns) & set(next_df.columns))
                                if common_cols:
                                    join_col = common_cols[0]

                            if join_col:
                                left_spatial_cols = set(graph_joins._find_spatial_columns(result_df))
                                right_spatial_cols = set(graph_joins._find_spatial_columns(next_df))
                                if join_col in left_spatial_cols and join_col in right_spatial_cols:
                                    def _normalize_point_series(series):
                                        coords = series.astype(str).str.extract(
                                            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
                                            flags=re.IGNORECASE,
                                        )
                                        if coords.empty:
                                            return None
                                        lons = pd.to_numeric(coords[0], errors="coerce")
                                        lats = pd.to_numeric(coords[1], errors="coerce")
                                        return lons.round(4).astype(str) + "," + lats.round(4).astype(str)

                                    left_norm = _normalize_point_series(result_df[join_col])
                                    right_norm = _normalize_point_series(next_df[join_col])
                                    if left_norm is not None and right_norm is not None:
                                        left_key = "__join_location_norm__"
                                        right_key = "__join_location_norm__"
                                        left_tmp = result_df.copy()
                                        right_tmp = next_df.copy()
                                        left_tmp[left_key] = left_norm
                                        right_tmp[right_key] = right_norm
                                        cols_to_include = [right_key]
                                        for col in requested_cols:
                                            if col not in cols_to_include:
                                                cols_to_include.append(col)
                                        return pd.merge(
                                            left_tmp,
                                            right_tmp[cols_to_include],
                                            left_on=left_key,
                                            right_on=right_key,
                                            how='left',
                                            suffixes=('', '_dup')
                                        ).drop(columns=[left_key, right_key], errors="ignore")

                                cols_to_include = [join_col]
                                for col in requested_cols:
                                    if col not in cols_to_include:
                                        cols_to_include.append(col)
                                cols_to_include = [
                                    col for col in cols_to_include
                                    if col not in result_df.columns or col == join_col
                                ]
                                if len(cols_to_include) > 1:
                                    return pd.merge(
                                        result_df,
                                        next_df[cols_to_include],
                                        on=join_col,
                                        how='left',
                                        suffixes=('', '_dup')
                                    )

                            # Fallback to spatial if no relational join is possible.
                            if requested_cols and _table_has_spatial(result_df) and _table_has_spatial(next_df):
                                spatial_distance = None
                                if spatial_mode == "distance":
                                    spatial_distance = spatial_distance_km
                                spatial_joined = _spatial_nearest_join(
                                    result_df,
                                    next_df,
                                    requested_cols,
                                    spatial_distance_km=spatial_distance,
                                )
                                if spatial_joined is not None:
                                    return spatial_joined

                            return result_df
                        
                        # Join along the path sequentially
                        # Find the position of uploaded_table_name in path_tables
                        if uploaded_table_name in path_tables:
                            upload_idx = path_tables.index(uploaded_table_name)
                            # Join tables before the uploaded table (walk backwards).
                            for i in range(upload_idx - 1, -1, -1):
                                result_df = _apply_join(result_df, path_tables[i + 1], path_tables[i])
                            # Join tables after the uploaded table (walk forwards).
                            for i in range(upload_idx + 1, len(path_tables)):
                                result_df = _apply_join(result_df, path_tables[i - 1], path_tables[i])
                        else:
                            if path_tables:
                                path_tables = [uploaded_table_name] + path_tables
                                for i in range(1, len(path_tables)):
                                    result_df = _apply_join(result_df, path_tables[i - 1], path_tables[i])
                        
                        # Rename appended columns to match user input names
                        rename_dict = {}
                        for user_input, (table_name, col_name) in user_input_to_column.items():
                            if col_name in result_df.columns:
                                # Rename the actual column name to the user's input name
                                rename_dict[col_name] = user_input
                        
                        result_df = result_df.rename(columns=rename_dict)
                        
                        # Select only original uploaded columns + user-requested attribute columns
                        # (in case there are intermediate columns from joins)
                        columns_to_keep = original_uploaded_columns.copy()
                        for user_input in user_input_to_column.keys():
                            if user_input in result_df.columns and user_input not in columns_to_keep:
                                columns_to_keep.append(user_input)
                        
                        # Reorder columns: original uploaded columns first, then appended attributes (in user input order)
                        result_df = result_df[columns_to_keep]
                        
                        # Store results in Scenario 2 format
                        st.session_state["scenario2_augmented_tables"] = [result_df]
                        st.session_state["scenario2_join_paths"] = [path_tables]
                        st.session_state["scenario2_path_info"] = [path_info]
                        appended_cols = [col for col in columns_to_keep if col not in original_uploaded_columns]
                        null_counts = {}
                        if appended_cols:
                            null_counts = result_df[appended_cols].isna().sum().to_dict()
                        st.session_state["scenario2_debug_info"] = {
                            "matched_attributes": matched_attributes,
                            "path_tables": path_tables,
                            "path_edges": path_info.get("edges", []),
                            "columns_to_add": columns_to_add,
                            "user_input_to_column": user_input_to_column,
                            "appended_columns": appended_cols,
                            "null_counts": null_counts,
                            "spatial_mode": spatial_mode,
                            "spatial_distance_km": spatial_distance_km,
                            "bbox_selection": bbox,
                        }
                        
                        # Also store in Scenario 1 format for consistent display
                        st.session_state["generated_join_path"] = path_info
                        st.session_state["generated_join_path_edges"] = path_info.get("edges", [])
                        st.session_state["generated_join_path_tables"] = path_tables
                        st.session_state["generated_tables_used"] = path_tables
                        st.session_state["user_query_active_data_lake"] = filtered_data_lake
                        st.session_state["current_user_query_path_index"] = 1
                        st.session_state["selected_user_query_path_for_map"] = None
                    else:
                        st.info("No join path found for the selected attributes.")
                        st.session_state["scenario2_augmented_tables"] = []
                        st.session_state["scenario2_join_paths"] = []
                        st.session_state["current_user_query_path_index"] = 1
                        st.session_state["selected_user_query_path_for_map"] = None
        
        # Display augmented table (same format as Scenario 1)
        if st.session_state.get("scenario2_augmented_tables"):
            augmented_tables = st.session_state["scenario2_augmented_tables"]
            join_paths = st.session_state["scenario2_join_paths"]
            
            # Show target table (same format as Scenario 1)
            if augmented_tables:
                aug_df = augmented_tables[0]  # Show first augmented table
                path_tables = join_paths[0] if join_paths else []
                
                st.markdown("**Target table:**")
                # Render as HTML with scrollable container (same as Scenario 1)
                tuples_html = aug_df.to_html(index=False, table_id="target_table")
                scrollable_html = f'<div style="max-height: 300px; overflow-y: auto;">{tuples_html}</div>'
                st.markdown(scrollable_html, unsafe_allow_html=True)
                
            _render_join_path_and_map(west_lafayette_bbox, lafayette_default_bbox)
        
        return  # Exit early for scenario 2

    # Scenario 1: Location-aware Data Discovery and Integration
    if not st.session_state.get("user_query_scenario1_explicit", False):
        st.info(
            "Select **Scenario 1: Location-aware Data Discovery and Integration** in the sidebar to begin."
        )
        return

    # Initialize session state for generated schema
    if "generated_target_schema" not in st.session_state:
        st.session_state["generated_target_schema"] = {}
    if "generated_tuples" not in st.session_state:
        st.session_state["generated_tuples"] = []
    if "generated_col_order" not in st.session_state:
        st.session_state["generated_col_order"] = []
    if "generated_tables_used" not in st.session_state:
        st.session_state["generated_tables_used"] = []
    if "generated_missing_inputs" not in st.session_state:
        st.session_state["generated_missing_inputs"] = []

    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with right_col:
        right_panel = st.container(
            border=False,
            height=SCENARIO1_RIGHT_PANEL_HEIGHT,
            vertical_alignment="distribute",
            gap=None,
        )
        with right_panel.container():
            st.markdown("**Step 2: Please enter the Columns of interest on the left panel**")
            if st.session_state.target_schema:
                # st.caption(
                #     "Columns of interest: "
                #     + ", ".join(str(value) for value in st.session_state.target_schema)
                # )
                st.caption(
                    "Add, remove, or rename columns in the **Target Attributes** table in the sidebar."
                )
                show_df = pd.DataFrame(
                    {"Column of interest": list(st.session_state.target_schema)}
                )
                st.dataframe(show_df, hide_index=True, width="stretch", height="content")
            else:
                st.info("No columns of interest selected yet.")

        with right_panel.container():
            if st.button(
                "Show the join paths",
                key="show_join_paths_button",
                disabled=not st.session_state.target_schema,
                width="stretch",
            ):
                matched_attributes = {}
                unmatched = []

                for user_input in st.session_state.target_schema:
                    matches = fuzzy_match_attribute(user_input, st.session_state.data_lake)
                    if matches:
                        matched_attributes[user_input] = matches
                    else:
                        unmatched.append(user_input)

                if unmatched:
                    st.warning(
                        f"Could not match the following attributes: {', '.join(unmatched)}"
                    )

                if matched_attributes:
                    st.session_state["scenario1_path_tables_data"] = {}
                    st.session_state["scenario1_filtered_data_lake"] = None
                    st.session_state["current_user_query_path_index"] = 1
                    st.session_state["selected_user_query_path_for_map"] = None
                    st.session_state["generated_target_schema"] = matched_attributes
                    st.session_state["generated_join_path"] = None
                    st.session_state["generated_join_path_edges"] = []
                    st.session_state["generated_join_path_tables"] = []
                    st.session_state["generated_join_paths_tables"] = []
                    st.session_state["generated_spatial_mode"] = None
                    st.session_state["generated_spatial_distance_km"] = None
                    st.session_state["generated_spatial_source"] = None
                    st.session_state["generated_tables_used"] = sorted(
                        {table for matches in matched_attributes.values() for table, _ in matches}
                    )
                    st.session_state["generated_missing_inputs"] = []

                    spatial_valid = True
                    if st.session_state["spatial_choice_mode"] == "inferred":
                        inferred_inputs = (
                            st.session_state.target_schema_raw or st.session_state.target_schema
                        )
                        inferred = infer_spatial_preferences(inferred_inputs)
                        spatial_mode = inferred["mode"]
                        spatial_distance_km = inferred["distance_km"]
                        spatial_source = "inferred"
                    else:
                        manual_mode = st.session_state["spatial_manual_mode"]
                        if manual_mode == "exclude":
                            spatial_mode = None
                            spatial_distance_km = None
                            spatial_source = "manual"
                        elif manual_mode == "distance":
                            distance_km = st.session_state.get("spatial_manual_distance_km")
                            if not distance_km or distance_km <= 0:
                                st.warning(
                                    "Please enter a positive distance (km) for the spatial predicate."
                                )
                                spatial_valid = False
                            spatial_mode = "distance"
                            spatial_distance_km = distance_km
                            spatial_source = "manual"
                        else:
                            spatial_mode = manual_mode
                            spatial_distance_km = None
                            spatial_source = "manual"

                    if spatial_valid:
                        joinability_threshold = st.session_state.get(
                            "graph_equi_joinability_measure",
                            0.0,
                        )
                        rel_edges = graph_joins.find_relational_joins(
                            st.session_state.data_lake
                        )
                        semantic_edges = graph_joins.find_semantic_joins(
                            st.session_state.data_lake,
                            min_overlap=joinability_threshold,
                        )
                        spatial_edges = graph_joins.find_spatial_joins(
                            st.session_state.data_lake
                        )
                        st.session_state["graph_rel_edges"] = rel_edges
                        st.session_state["graph_semantic_edges"] = semantic_edges
                        st.session_state["graph_spatial_edges"] = spatial_edges

                        if (
                            spatial_mode is None
                            and st.session_state["spatial_choice_mode"] == "inferred"
                        ):
                            input_tables = {
                                table
                                for matches in matched_attributes.values()
                                for table, _ in matches
                            }
                            spatial_hint = any(
                                left in input_tables and right in input_tables
                                for left, right, _ in spatial_edges
                            )
                            if spatial_hint:
                                spatial_mode = "distance"
                                spatial_distance_km = None
                                spatial_source = "inferred-graph"

                        allow_spatial = spatial_mode is not None
                        require_spatial = spatial_mode is not None

                        if not _data_lake_has_spatial(st.session_state.data_lake):
                            st.warning(
                                "No spatial columns found. The bounding box filter cannot be applied."
                            )
                        else:
                            path_info = find_min_join_path(
                                matched_attributes,
                                rel_edges,
                                semantic_edges,
                                spatial_edges,
                                allow_spatial=allow_spatial,
                                require_spatial=require_spatial,
                                max_tables=len(st.session_state.data_lake),
                            )

                            if path_info:
                                missing_inputs = []
                                for user_input, matches in matched_attributes.items():
                                    if not any(
                                        table in path_info["tables"] for table, _ in matches
                                    ):
                                        missing_inputs.append(user_input)
                                if missing_inputs:
                                    path_info = None
                                    st.session_state["generated_missing_inputs"] = missing_inputs

                            if not path_info:
                                spatial_hint = False
                                input_tables = {
                                    table
                                    for matches in matched_attributes.values()
                                    for table, _ in matches
                                }
                                for left, right, _ in spatial_edges:
                                    if left in input_tables and right in input_tables:
                                        spatial_hint = True
                                        break
                                if spatial_mode is None and spatial_hint:
                                    st.info(
                                        "No join path found. Spatial joins are excluded; choose a spatial predicate "
                                        "or include a phrase like 'closest' to enable spatial joins."
                                    )
                                else:
                                    st.info(
                                        "No join path found for the selected attributes and spatial preference."
                                    )
                                st.session_state["generated_tuples"] = []
                                st.session_state["generated_col_order"] = []
                                st.session_state["current_user_query_path_index"] = 1
                                st.session_state["selected_user_query_path_for_map"] = None
                            else:
                                st.session_state["generated_join_path"] = path_info
                                st.session_state["generated_join_path_edges"] = path_info["edges"]
                                st.session_state["generated_join_path_tables"] = path_info["tables"]
                                st.session_state["generated_spatial_mode"] = spatial_mode
                                st.session_state["generated_spatial_distance_km"] = (
                                    spatial_distance_km
                                )
                                st.session_state["generated_spatial_source"] = spatial_source
                                st.session_state["generated_tables_used"] = path_info["tables"]
                                input_tables = {
                                    table
                                    for matches in matched_attributes.values()
                                    for table, _ in matches
                                }
                                candidate_paths = _collect_join_paths(
                                    input_tables,
                                    rel_edges,
                                    semantic_edges,
                                    spatial_edges,
                                    allow_spatial=allow_spatial,
                                    require_spatial=require_spatial,
                                    max_paths=6,
                                    max_len=min(6, len(st.session_state.data_lake)),
                                )
                                primary_path = tuple(path_info["tables"])
                                if primary_path in candidate_paths:
                                    candidate_paths = [primary_path] + [
                                        path
                                        for path in candidate_paths
                                        if path != primary_path
                                    ]
                                else:
                                    candidate_paths = [primary_path] + candidate_paths
                                candidate_paths = _prepend_hardcoded_first_scenario1_path(
                                    candidate_paths,
                                    st.session_state.data_lake,
                                )
                                st.session_state["generated_join_paths_tables"] = candidate_paths
                                if (
                                    candidate_paths
                                    and list(candidate_paths[0])
                                    == list(HARDCODED_FIRST_SCENARIO1_PATH)
                                ):
                                    h_edges = hardcoded_first_scenario1_path_edges()
                                    st.session_state["generated_join_path_tables"] = list(
                                        HARDCODED_FIRST_SCENARIO1_PATH
                                    )
                                    st.session_state["generated_join_path_edges"] = h_edges
                                    st.session_state["generated_join_path"] = {
                                        "tables": list(HARDCODED_FIRST_SCENARIO1_PATH),
                                        "edges": h_edges,
                                        "bridge_tables": [],
                                        "used_spatial": True,
                                    }

                                bbox = st.session_state.get("bbox_selection")
                                candidate_paths = (
                                    st.session_state.get("generated_join_paths_tables") or []
                                )
                                all_tables_on_paths = set()
                                for path in candidate_paths:
                                    all_tables_on_paths.update(path)

                                filtered_data_lake = st.session_state.data_lake
                                st.session_state["scenario1_filtered_data_lake"] = None

                                if bbox and candidate_paths:
                                    tables_with_spatial = []
                                    for table_name in all_tables_on_paths:
                                        if table_name in st.session_state.data_lake:
                                            df = st.session_state.data_lake[table_name]
                                            if _table_has_spatial(df):
                                                tables_with_spatial.append(table_name)

                                    if tables_with_spatial:
                                        filtered_data_lake = {}
                                        for name, df in st.session_state.data_lake.items():
                                            filtered_df, _ = _filter_table_by_bbox(df, bbox)
                                            filtered_data_lake[name] = filtered_df

                                        any_spatial_empty = False
                                        for table_name in tables_with_spatial:
                                            if table_name in filtered_data_lake:
                                                df = filtered_data_lake[table_name]
                                                if len(df) == 0:
                                                    any_spatial_empty = True
                                                    break

                                        if any_spatial_empty:
                                            empty_spatial = [
                                                table_name
                                                for table_name in tables_with_spatial
                                                if len(filtered_data_lake.get(table_name, [])) == 0
                                            ]
                                            st.warning(
                                                "Cannot integrate in this area: at least one spatial table used on "
                                                "these paths has **no records inside your drawn bounding box**: "
                                                f"{', '.join(empty_spatial)}. "
                                                "Widen the box or move it over the missing table’s data."
                                            )
                                            st.session_state["generated_join_path"] = None
                                            st.session_state["generated_join_path_edges"] = []
                                            st.session_state["generated_join_path_tables"] = []
                                            st.session_state["generated_join_paths_tables"] = []
                                            st.session_state["generated_tables_used"] = []
                                            st.session_state["current_user_query_path_index"] = 1
                                            st.session_state["selected_user_query_path_for_map"] = None
                                            st.session_state["user_query_path_map_selections"] = {}
                                            col_order = []
                                            seen_cols = set()
                                            for matches in matched_attributes.values():
                                                for table, column in matches:
                                                    if column not in seen_cols:
                                                        col_order.append(column)
                                                        seen_cols.add(column)
                                            st.session_state["generated_col_order"] = col_order
                                        else:
                                            st.session_state["scenario1_filtered_data_lake"] = (
                                                filtered_data_lake
                                            )
                                    else:
                                        st.session_state["scenario1_filtered_data_lake"] = dict(
                                            st.session_state.data_lake
                                        )
                                elif candidate_paths:
                                    st.session_state["scenario1_filtered_data_lake"] = dict(
                                        st.session_state.data_lake
                                    )

    st.session_state["user_query_active_data_lake"] = st.session_state.data_lake
    parsed_paths, active_data_lake = _sync_user_query_path_state()
    with left_col:
        _render_scenario1_map_panel(parsed_paths, active_data_lake)

    if parsed_paths:
        _render_scenario1_join_path_panel(
            parsed_paths,
            active_data_lake,
            st.session_state.get("bbox_selection"),
        )

    _render_scenario1_target_table(
        st.session_state.get("current_user_query_path_index", 1)
    )


def _render_join_path_and_map(west_lafayette_bbox, lafayette_default_bbox):
    show_per_path_target_table = (
        st.session_state.get("user_query_scenario_key") != "scenario_2_augmentation"
    )
    parsed_paths, active_data_lake = _sync_user_query_path_state()

    if "user_query_parsed_paths" in st.session_state:
        table_metadata = build_table_metadata(active_data_lake)
        highlighted_tables = {
            table_name
            for matches in st.session_state.get("generated_target_schema", {}).values()
            for table_name, _ in matches
        }

        st.markdown("**Generated Join Paths**")
        render_join_path_legend(
            rows_hint=(
                "ROWS and COLS populate after you click Show target table for a path."
                if show_per_path_target_table
                else "ROWS and COLS reflect the materialized target table shown above."
            )
        )

        if not parsed_paths:
            st.info("No paths found. Please check your target attributes and ensure they reference valid tables.")
            return

        current_path_idx = st.session_state.get("current_user_query_path_index", 1)
        current_path_idx = max(1, min(current_path_idx, len(parsed_paths)))

        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button(
                "Previous join path",
                key="prev_user_query_join_path",
                disabled=len(parsed_paths) <= 1,
                width="stretch",
            ):
                current_path_idx = ((current_path_idx - 2) % len(parsed_paths)) + 1
                st.session_state["selected_user_query_path_for_map"] = None
        with nav_cols[1]:
            if st.button(
                "Next join path",
                key="next_user_query_join_path",
                disabled=len(parsed_paths) <= 1,
                width="stretch",
            ):
                current_path_idx = (current_path_idx % len(parsed_paths)) + 1
                st.session_state["selected_user_query_path_for_map"] = None

        st.session_state["current_user_query_path_index"] = current_path_idx

        idx = current_path_idx
        path = parsed_paths[idx - 1]
        path_data = {
            "tables": path,
            "joins": [],
        }
        for i in range(len(path) - 1):
            join_info = _get_join_info_user_query(path[i], path[i + 1])
            path_data["joins"].append({
                "from": path[i],
                "to": path[i + 1],
                "attributes": join_info["attributes"],
                "type": join_info["type"],
            })

        card_container = st.container()

        action_cols = st.columns(2 if show_per_path_target_table else 1)
        with action_cols[0]:
            if st.button(
                "Show on the map",
                key=f"show_user_query_path_{idx}",
                width="stretch",
            ):
                _store_selected_user_query_path_for_map(
                    idx,
                    parsed_paths,
                    active_data_lake,
                    west_lafayette_bbox,
                )

        if show_per_path_target_table:
            with action_cols[1]:
                if st.button(
                    "Show target table",
                    key=f"show_target_table_path_{idx}",
                    width="stretch",
                ):
                    matched = st.session_state.get("generated_target_schema")
                    if not matched:
                        st.warning("Click **Show the join paths** first.")
                    else:
                        filtered = st.session_state.get("scenario1_filtered_data_lake")
                        if filtered is None:
                            filtered = st.session_state.data_lake
                        allow_sp = (
                            st.session_state.get("generated_spatial_mode") is not None
                        )
                        if tuple(path) == HARDCODED_FIRST_SCENARIO1_PATH:
                            allow_sp = True
                        rel_e = st.session_state.get("graph_rel_edges", [])
                        sem_e = st.session_state.get("graph_semantic_edges", [])
                        spa_e = st.session_state.get("graph_spatial_edges", [])
                        path_edges = _resolve_scenario1_path_edges(
                            list(path),
                            rel_e,
                            sem_e,
                            spa_e,
                            allow_spatial=allow_sp,
                        )
                        if path_edges is None:
                            st.error(
                                "Could not resolve join edges for consecutive tables on this path."
                            )
                        else:
                            if tuple(path) == HARDCODED_FIRST_SCENARIO1_PATH:
                                spatial_mode_run = "distance"
                                spatial_dist_run = None
                            else:
                                spatial_mode_run = st.session_state.get(
                                    "generated_spatial_mode"
                                )
                                spatial_dist_run = st.session_state.get(
                                    "generated_spatial_distance_km"
                                )
                            tuples, col_order = generate_joined_tuples(
                                matched,
                                filtered,
                                join_path=list(path),
                                join_edges=path_edges,
                                spatial_mode=spatial_mode_run,
                                spatial_distance_km=spatial_dist_run,
                            )
                            preview_note = None
                            if (
                                not tuples
                                and st.session_state.get("scenario1_filtered_data_lake")
                                is not None
                            ):
                                tuples, col_order = generate_joined_tuples(
                                    matched,
                                    st.session_state.data_lake,
                                    join_path=list(path),
                                    join_edges=path_edges,
                                    spatial_mode=spatial_mode_run,
                                    spatial_distance_km=spatial_dist_run,
                                )
                                if tuples:
                                    preview_note = (
                                        "Your **bounding box** did not overlap "
                                        "any coordinates in the filtered tables, "
                                        "so every table was empty after filtering. "
                                        "Results below use the **full data lake** "
                                        "(no bbox filter)."
                                    )
                            st.session_state.setdefault("scenario1_path_tables_data", {})
                            st.session_state["scenario1_path_tables_data"][idx] = {
                                "tuples": tuples,
                                "col_order": col_order,
                                "note": preview_note,
                            }
                            if not tuples:
                                st.info(
                                    "**No rows** after inner-joining along "
                                    "this path—check keys, spatial columns "
                                    "(geocoded tables), and your bounding box."
                                )

        preview = st.session_state.get("scenario1_path_tables_data", {}).get(idx)
        preview_rows = None
        preview_cols = None
        if preview is not None:
            preview_rows = len(preview.get("tuples") or [])
            col_order = preview.get("col_order") or []
            preview_cols = len(col_order) if col_order else None
        elif not show_per_path_target_table:
            augmented_tables = st.session_state.get("scenario2_augmented_tables", [])
            if idx - 1 < len(augmented_tables):
                aug_df = augmented_tables[idx - 1]
                preview_rows = len(aug_df)
                preview_cols = len(aug_df.columns)

        with card_container:
            render_join_path_card(
                component_id=f"user_query_path_{idx}_card",
                title=f"Join Path {idx:03d}",
                path_data=path_data,
                table_metadata=table_metadata,
                highlighted_tables=highlighted_tables,
                stats={
                    "rows": preview_rows,
                    "cols": preview_cols,
                    "len": len(path),
                    "hops": max(len(path) - 1, 0),
                },
                selected=True,
            )

        preview = st.session_state.get("scenario1_path_tables_data", {}).get(idx)
        if show_per_path_target_table and preview:
            st.markdown("**Target table** (this path)")
            if preview.get("note"):
                st.caption(preview["note"])
            tlist = preview.get("tuples") or []
            pcols = preview.get("col_order") or []
            if tlist and pcols:
                prev_df = pd.DataFrame(tlist, columns=pcols)
                search_query = st.text_input(
                    "Search target table",
                    key=f"target_table_search_{idx}",
                    placeholder="Type a keyword to filter rows",
                )
                filtered_prev_df = _filter_preview_table_by_keyword(
                    prev_df, search_query
                )
                if search_query.strip():
                    st.caption(
                        f"{len(filtered_prev_df)} matching row(s) for "
                        f'"{search_query.strip()}".'
                    )
                pt_html = filtered_prev_df.to_html(
                    index=False, table_id=f"target_table_path_{idx}"
                )
                scroll = (
                    '<div style="max-height: 280px; overflow-y: auto;">'
                    f"{pt_html}</div>"
                )
                st.markdown(scroll, unsafe_allow_html=True)
                if search_query.strip() and filtered_prev_df.empty:
                    st.caption("No rows match the current keyword search.")
            elif pcols:
                st.dataframe(
                    pd.DataFrame(columns=pcols),
                    width="stretch",
                )
                st.caption("No rows after joining along this path.")
            else:
                st.caption("No columns to display.")

        st.markdown("**Map View**")
        map_height = 700
        st.markdown(
            f'<style>div[data-testid="stHtml"] iframe[src*="folium"]{{height:{map_height}px!important;min-height:{map_height}px!important;max-height:{map_height}px!important;}}</style>',
            unsafe_allow_html=True,
        )

        selected_path_idx = st.session_state.get("selected_user_query_path_for_map")
        path_region_info = None
        path_map_bbox = None
        if selected_path_idx:
            stored_info = st.session_state.user_query_path_map_selections.get(
                f"user_query_path_{selected_path_idx}"
            )
            if stored_info:
                path_region_info = stored_info
                path_map_bbox = stored_info.get("bbox")
            else:
                path_region_info = _store_selected_user_query_path_for_map(
                    selected_path_idx,
                    parsed_paths,
                    active_data_lake,
                    west_lafayette_bbox,
                )
                path_map_bbox = path_region_info.get("bbox")

        # if not selected_path_idx or path_map_bbox is None:
        #     st.info("Click **Show on the map** to display the current join path on the map.")
        #     return

        bbox_for_filtering = st.session_state.get("bbox_selection")
            
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
            
            if selected_path_idx and parsed_paths:
                path = parsed_paths[selected_path_idx - 1] if selected_path_idx - 1 < len(parsed_paths) else []
                points = _extract_points_from_path(path, active_data_lake, max_points_per_table=200)
                
                table_colors = {}
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                         'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 
                         'lightgreen', 'gray', 'black', 'lightgray']
                for point_idx, table_name in enumerate(path):
                    table_colors[table_name] = colors[point_idx % len(colors)]
                
                if bbox_for_filtering:
                    margin = 0.01
                    for lat, lon, table_name in points:
                        if (bbox_for_filtering[0] - margin <= lat <= bbox_for_filtering[2] + margin and 
                            bbox_for_filtering[1] - margin <= lon <= bbox_for_filtering[3] + margin):
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
                else:
                    for lat, lon, table_name in points:
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
            st_folium(m_path, width=None, height=map_height, returned_objects=[])
        else:
            map_html = m_path._repr_html_()
            map_html = re.sub(
                r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*height:\s*)\d+px',
                lambda m: m.group(1) + f"{map_height}px",
                map_html,
            )
            if 'id="map' in map_html:
                if 'style=' in map_html:
                    map_html = re.sub(
                        r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*)"',
                        rf'\1 height: {map_height}px !important;"',
                        map_html,
                    )
                else:
                    map_html = re.sub(
                        r'(<div[^>]*id="[^"]*map[^"]*")',
                        rf'\1 style="height: {map_height}px !important;"',
                        map_html,
                    )

            components.html(map_html, height=map_height)
            st.markdown(
                f'<script>(function(){{function forceResize(){{var iframes=document.querySelectorAll(\"iframe\");iframes.forEach(function(iframe){{iframe.style.setProperty(\"height\",\"{map_height}px\",\"important\");iframe.style.setProperty(\"min-height\",\"{map_height}px\",\"important\");iframe.style.setProperty(\"max-height\",\"{map_height}px\",\"important\");iframe.setAttribute(\"height\",\"{map_height}\");}});}}forceResize();setTimeout(forceResize,100);setTimeout(forceResize,500);setTimeout(forceResize,1000);setTimeout(forceResize,2000);var observer=new MutationObserver(forceResize);observer.observe(document.body,{{childList:true,subtree:true}});}})();</script>',
                unsafe_allow_html=True,
            )

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
