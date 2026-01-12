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
import leafmap.foliumap as leafmap
from sections.generate_target_schema import (
    find_min_join_path,
    fuzzy_match_attribute,
    generate_joined_tuples,
    infer_spatial_preferences,
)


def _extract_point_coords(point_str):
    """Extract lat, lon from POINT(lon lat) or similar formats."""
    if not isinstance(point_str, str):
        return None
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', point_str, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lat, lon)
    return None


def _find_region_from_tables(path_tables, data_lake):
    """
    Extract geographic region information from tables in a path.
    Returns: dict with 'region_name', 'bbox', and 'method'
    """
    all_coords = []
    region_names = []

    for table_name in path_tables:
        if table_name not in data_lake:
            continue
        df = data_lake[table_name]

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['state', 'region', 'location', 'area', 'county']):
                unique_vals = df[col].dropna().unique()[:10]
                for val in unique_vals:
                    val_str = str(val).lower().strip()
                    for state_name, bbox in US_STATE_BBOXES.items():
                        if state_name in val_str or val_str in state_name:
                            region_names.append(state_name.title())
                            break

        lat_col = None
        lon_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lon_col is None:
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
                coords = _extract_point_coords(val)
                if coords:
                    all_coords.append(coords)

    if region_names:
        from collections import Counter
        most_common_region = Counter(region_names).most_common(1)[0][0]
        region_lower = most_common_region.lower()
        if region_lower in US_STATE_BBOXES:
            return {
                "region_name": most_common_region,
                "bbox": US_STATE_BBOXES[region_lower],
                "method": "state_column"
            }

    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]
        return {
            "region_name": "Computed from coordinates",
            "bbox": bbox,
            "method": "coordinates"
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
        coords = df[col].astype(str).str.extract(
            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
            flags=re.IGNORECASE,
        )
        if coords.empty:
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
        coords = df[col].astype(str).str.extract(
            r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)",
            flags=re.IGNORECASE,
        )
        if coords.empty:
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
            coords = _extract_point_coords(value)
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


def render_user_query_section(west_lafayette_bbox, lafayette_default_bbox):
    st.markdown('<div id="user-query"></div>', unsafe_allow_html=True)
    st.subheader("User Query")
    if "graph_rel_edges" not in st.session_state:
        st.session_state["graph_rel_edges"] = []
    if "graph_spatial_edges" not in st.session_state:
        st.session_state["graph_spatial_edges"] = []
    if "target_schema_raw" not in st.session_state:
        st.session_state["target_schema_raw"] = []
    
    # Check if CSV table is uploaded (Scenario 2)
    if st.session_state.get("user_uploaded_table_df") is not None:
        # Scenario 2: CSV Upload + Append Attributes
        uploaded_df = st.session_state.user_uploaded_table_df
        uploaded_table_name = st.session_state.get("user_uploaded_table_name", "uploaded_table")
        
        st.markdown("**Uploaded Table:**")
        st.dataframe(uploaded_df, use_container_width=True, height=300)
        
        st.markdown("**Step 1: Please select your area of interest (draw a bounding box on the map)**")
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
        
        st.markdown("**Attributes to append:**")
        if st.session_state.target_schema:
            attrs = st.session_state.target_schema.copy()
            to_remove = []
            for attr in attrs:
                row = st.columns([1, 20])
                with row[0]:
                    if st.button("✕", key=f"scenario2_remove_attr_{attr}", help="Remove", use_container_width=True):
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
                    rel_edges = st.session_state.get("graph_rel_edges")
                    spatial_edges = st.session_state.get("graph_spatial_edges")
                    if not rel_edges:
                        rel_edges = graph_joins.find_relational_joins(temp_data_lake)
                        st.session_state["graph_rel_edges"] = rel_edges
                    if not spatial_edges:
                        spatial_edges = graph_joins.find_spatial_joins(temp_data_lake)
                        st.session_state["graph_spatial_edges"] = spatial_edges
                    
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
                            if edge_type != "relation":
                                continue
                            attrs = []
                            if edge and edge.get("attributes"):
                                attrs_str = edge["attributes"]
                                if isinstance(attrs_str, str):
                                    attrs = [a.strip() for a in attrs_str.split(",")]
                                else:
                                    attrs = [attrs_str]
                            if attrs:
                                link_cols_by_table[left].update(attrs)
                                link_cols_by_table[right].update(attrs)
                                continue
                            left_df = filtered_data_lake.get(left)
                            right_df = filtered_data_lake.get(right)
                            if left_df is not None and right_df is not None:
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
                    else:
                        st.info("No join path found for the selected attributes.")
                        st.session_state["scenario2_augmented_tables"] = []
                        st.session_state["scenario2_join_paths"] = []
        
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

    # Scenario 1: Original implementation (no CSV uploaded)
    st.markdown("**Step 1: Please select your area of interest (draw a bounding box on the map)**")
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

    # Target attributes are now handled in the sidebar

    # Only show the rest if attributes have been submitted
    if not st.session_state.target_schema:
        return

    st.markdown("**User input schema:**")
    if st.session_state.target_schema:
        attrs = st.session_state.target_schema.copy()
        to_remove = []

        st.markdown(
            """
            <style>
            div[data-testid="column"]:has(button[key^="remove_attr_"]) {
                flex: 0 0 auto !important;
                width: fit-content !important;
                min-width: 35px !important;
                max-width: 40px !important;
                display: flex !important;
                align-items: center !important;
            }
            div[data-testid="column"]:has(button[key^="remove_attr_"]) + div[data-testid="column"] {
                flex: 0 1 auto !important;
                width: fit-content !important;
                max-width: fit-content !important;
                display: flex !important;
                align-items: center !important;
            }
            div[data-testid="column"]:has(button[key^="remove_attr_"]) button {
                height: 32px !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for attr in attrs:
            row = st.columns([1, 20])
            with row[0]:
                if st.button("✕", key=f"remove_attr_{attr}", help="Remove", use_container_width=True):
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
        st.write("*None selected yet*")

    # Spatial join preference is now handled in the sidebar

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

    # Add "Generate target schema" button
    if st.button("Generate target schema", key="generate_target_schema_button"):
        if not st.session_state.target_schema:
            st.warning("Please add at least one target attribute first.")
        else:
            matched_attributes = {}
            unmatched = []
            
            for user_input in st.session_state.target_schema:
                matches = fuzzy_match_attribute(user_input, st.session_state.data_lake)
                if matches:
                    matched_attributes[user_input] = matches
                else:
                    unmatched.append(user_input)
            
            if unmatched:
                st.warning(f"Could not match the following attributes: {', '.join(unmatched)}")
            
            if matched_attributes:
                st.session_state["generated_target_schema"] = matched_attributes
                st.session_state["generated_join_path"] = None
                st.session_state["generated_join_path_edges"] = []
                st.session_state["generated_join_path_tables"] = []
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
                            st.warning("Please enter a positive distance (km) for the spatial predicate.")
                            spatial_valid = False
                        spatial_mode = "distance"
                        spatial_distance_km = distance_km
                        spatial_source = "manual"
                    else:
                        spatial_mode = manual_mode
                        spatial_distance_km = None
                        spatial_source = "manual"

                rel_edges = st.session_state.get("graph_rel_edges")
                spatial_edges = st.session_state.get("graph_spatial_edges")
                if not rel_edges:
                    rel_edges = graph_joins.find_relational_joins(st.session_state.data_lake)
                    st.session_state["graph_rel_edges"] = rel_edges
                if not spatial_edges:
                    spatial_edges = graph_joins.find_spatial_joins(st.session_state.data_lake)
                    st.session_state["graph_spatial_edges"] = spatial_edges

                if spatial_mode is None and st.session_state["spatial_choice_mode"] == "inferred":
                    input_tables = {
                        table for matches in matched_attributes.values() for table, _ in matches
                    }
                    spatial_hint = any(
                        left in input_tables and right in input_tables for left, right, _ in spatial_edges
                    )
                    if spatial_hint:
                        spatial_mode = "distance"
                        spatial_distance_km = None
                        spatial_source = "inferred-graph"

                allow_spatial = spatial_mode is not None
                require_spatial = spatial_mode is not None

                if not spatial_valid:
                    return

                if not _data_lake_has_spatial(st.session_state.data_lake):
                    st.warning("No spatial columns found. The bounding box filter cannot be applied.")
                    return

                path_info = find_min_join_path(
                    matched_attributes,
                    rel_edges,
                    spatial_edges,
                    allow_spatial=allow_spatial,
                    require_spatial=require_spatial,
                    max_tables=len(st.session_state.data_lake),
                )

                if path_info:
                    missing_inputs = []
                    for user_input, matches in matched_attributes.items():
                        if not any(table in path_info["tables"] for table, _ in matches):
                            missing_inputs.append(user_input)
                    if missing_inputs:
                        path_info = None
                        st.session_state["generated_missing_inputs"] = missing_inputs

                if not path_info:
                    spatial_hint = False
                    input_tables = {
                        table for matches in matched_attributes.values() for table, _ in matches
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
                        st.info("No join path found for the selected attributes and spatial preference.")
                    st.session_state["generated_tuples"] = []
                    st.session_state["generated_col_order"] = []
                    return
                else:
                    st.session_state["generated_join_path"] = path_info
                    st.session_state["generated_join_path_edges"] = path_info["edges"]
                    st.session_state["generated_join_path_tables"] = path_info["tables"]
                    st.session_state["generated_spatial_mode"] = spatial_mode
                    st.session_state["generated_spatial_distance_km"] = spatial_distance_km
                    st.session_state["generated_spatial_source"] = spatial_source
                    st.session_state["generated_tables_used"] = path_info["tables"]
                    input_tables = {
                        table for matches in matched_attributes.values() for table, _ in matches
                    }
                    candidate_paths = _collect_join_paths(
                        input_tables,
                        rel_edges,
                        spatial_edges,
                        allow_spatial=allow_spatial,
                        require_spatial=require_spatial,
                        max_paths=6,
                        max_len=min(6, len(st.session_state.data_lake)),
                    )
                    primary_path = tuple(path_info["tables"])
                    if primary_path in candidate_paths:
                        candidate_paths = [primary_path] + [
                            path for path in candidate_paths if path != primary_path
                        ]
                    else:
                        candidate_paths = [primary_path] + candidate_paths
                    st.session_state["generated_join_paths_tables"] = candidate_paths

                # Generate joined tuples
                bbox = st.session_state.get("bbox_selection")
                path_tables = st.session_state.get("generated_join_path_tables", [])
                filtered_data_lake = st.session_state.data_lake
                
                if bbox and path_tables:
                    # Check if any tables in the path have spatial attributes
                    tables_with_spatial = []
                    for table_name in path_tables:
                        if table_name in st.session_state.data_lake:
                            df = st.session_state.data_lake[table_name]
                            if _table_has_spatial(df):
                                tables_with_spatial.append(table_name)
                    
                    # If at least one table has spatial attributes, check filtering
                    if tables_with_spatial:
                        # Filter all tables by bbox
                        filtered_data_lake = {}
                        for name, df in st.session_state.data_lake.items():
                            filtered_df, _ = _filter_table_by_bbox(df, bbox)
                            filtered_data_lake[name] = filtered_df
                        
                        # Check if any table with spatial attributes is empty after filtering
                        any_spatial_empty = False
                        for table_name in tables_with_spatial:
                            if table_name in filtered_data_lake:
                                df = filtered_data_lake[table_name]
                                if len(df) == 0:
                                    any_spatial_empty = True
                                    break
                        
                        # If any table with spatial attributes is empty, target table should be empty
                        if any_spatial_empty:
                            st.session_state["generated_join_path"] = None
                            st.session_state["generated_join_path_edges"] = []
                            st.session_state["generated_join_path_tables"] = []
                            st.session_state["generated_tables_used"] = []
                            st.session_state["generated_tuples"] = []
                            # Generate column order for empty table display
                            col_order = []
                            seen_cols = set()
                            for matches in matched_attributes.values():
                                for table, column in matches:
                                    if column not in seen_cols:
                                        col_order.append(column)
                                        seen_cols.add(column)
                            st.session_state["generated_col_order"] = col_order
                            total_matches = sum(len(matches) for matches in matched_attributes.values())
                        else:
                            # All spatial tables have data, proceed with filtering
                            tuples, col_order = generate_joined_tuples(
                                matched_attributes,
                                filtered_data_lake,
                                join_path=st.session_state.get("generated_join_path_tables"),
                                join_edges=st.session_state.get("generated_join_path_edges"),
                                spatial_mode=st.session_state.get("generated_spatial_mode"),
                                spatial_distance_km=st.session_state.get("generated_spatial_distance_km"),
                            )
                            st.session_state["generated_tuples"] = tuples
                            st.session_state["generated_col_order"] = col_order
                            total_matches = sum(len(matches) for matches in matched_attributes.values())
                    else:
                        # No tables have spatial attributes, ignore bbox and use unfiltered data
                        tuples, col_order = generate_joined_tuples(
                            matched_attributes,
                            filtered_data_lake,
                            join_path=st.session_state.get("generated_join_path_tables"),
                            join_edges=st.session_state.get("generated_join_path_edges"),
                            spatial_mode=st.session_state.get("generated_spatial_mode"),
                            spatial_distance_km=st.session_state.get("generated_spatial_distance_km"),
                        )
                        st.session_state["generated_tuples"] = tuples
                        st.session_state["generated_col_order"] = col_order
                        total_matches = sum(len(matches) for matches in matched_attributes.values())
                else:
                    # No bbox or no path, use unfiltered data
                    tuples, col_order = generate_joined_tuples(
                        matched_attributes,
                        filtered_data_lake,
                        join_path=st.session_state.get("generated_join_path_tables"),
                        join_edges=st.session_state.get("generated_join_path_edges"),
                        spatial_mode=st.session_state.get("generated_spatial_mode"),
                        spatial_distance_km=st.session_state.get("generated_spatial_distance_km"),
                    )
                    st.session_state["generated_tuples"] = tuples
                    st.session_state["generated_col_order"] = col_order
                    total_matches = sum(len(matches) for matches in matched_attributes.values())

    # Display results
    if st.session_state["generated_target_schema"]:
        col_names = st.session_state.get("generated_col_order", [])
        if not col_names:
            # Fallback: generate column names from matched schema (only column names)
            col_names = []
            for matches in st.session_state["generated_target_schema"].values():
                for table, column in matches:
                    if column not in col_names:
                        col_names.append(column)
        
        # Always show target table (even if empty)
        st.markdown("**Target table:**")
        if st.session_state["generated_tuples"]:
            tuples_df = pd.DataFrame(
                st.session_state["generated_tuples"],
                columns=col_names
            )
            # Render as HTML with scrollable container showing ~5 rows at a time
            tuples_html = tuples_df.to_html(index=False, table_id="target_table")
            scrollable_html = f'<div style="max-height: 300px; overflow-y: auto;">{tuples_html}</div>'
            st.markdown(scrollable_html, unsafe_allow_html=True)
        else:
            # Show empty table with headers only
            empty_df = pd.DataFrame(columns=col_names)
            st.dataframe(empty_df, use_container_width=True)
        
    st.session_state["user_query_active_data_lake"] = st.session_state.data_lake
    _render_join_path_and_map(west_lafayette_bbox, lafayette_default_bbox)


def _render_join_path_and_map(west_lafayette_bbox, lafayette_default_bbox):
    path_tables = st.session_state.get("generated_join_path_tables", [])
    path_tables_list = st.session_state.get("generated_join_paths_tables", [])
    if path_tables_list:
        st.session_state["user_query_paths"] = [" → ".join(path) for path in path_tables_list]
        st.session_state["user_query_parsed_paths"] = [list(path) for path in path_tables_list]
    elif path_tables:
        path_str = " → ".join(path_tables)
        st.session_state["user_query_paths"] = [path_str]
        st.session_state["user_query_parsed_paths"] = [path_tables]

    if "user_query_path_map_selections" not in st.session_state:
        st.session_state["user_query_path_map_selections"] = {}

    if "selected_user_query_path_for_map" not in st.session_state:
        st.session_state["selected_user_query_path_for_map"] = None

    parsed_paths = st.session_state.get("user_query_parsed_paths", [])
    active_data_lake = st.session_state.get("user_query_active_data_lake", st.session_state.data_lake)

    if "user_query_parsed_paths" in st.session_state:
        def get_join_info_user_query(table1, table2):
            for edge in st.session_state.get("generated_join_path_edges", []):
                if edge["from"] == table1 and edge["to"] == table2:
                    return {"type": edge["type"], "attributes": edge["attributes"]}
                if edge["from"] == table2 and edge["to"] == table1:
                    return {"type": edge["type"], "attributes": edge["attributes"]}
            for left, right, label in st.session_state.graph_rel_edges:
                if (left == table1 and right == table2) or (left == table2 and right == table1):
                    return {"type": "relation", "attributes": label}
            for left, right, label in st.session_state.graph_spatial_edges:
                if (left == table1 and right == table2) or (left == table2 and right == table1):
                    return {"type": "spatial", "attributes": label}
            return {"type": "relation", "attributes": ""}

        table_metadata = {}
        for table_name in active_data_lake.keys():
            df = active_data_lake[table_name]
            table_metadata[table_name] = {
                "columns": list(df.columns),
                "cardinality": len(df),
                "num_columns": len(df.columns)
            }

        col_paths, col_map = st.columns([1, 1.2])

        with col_paths:
            st.markdown("**Generated Join Paths**")

            if not parsed_paths:
                st.info("No paths found. Please check your target attributes and ensure they reference valid tables.")
            else:
                selected_path_idx = st.session_state.get("selected_user_query_path_for_map")
                for idx, path in enumerate(parsed_paths, start=1):
                    is_selected = selected_path_idx == idx
                    st.markdown(f"**Path {idx}:**" + (" ✓ Selected" if is_selected else ""))

                    path_data = {
                        "tables": path,
                        "joins": []
                    }

                    for i in range(len(path) - 1):
                        join_info = get_join_info_user_query(path[i], path[i + 1])
                        path_data["joins"].append({
                            "from": path[i],
                            "to": path[i + 1],
                            "attributes": join_info["attributes"],
                            "type": join_info["type"]
                        })

                    highlight_style = "border: 2px solid #4caf50; background: #f1f8e9;" if is_selected else "border: 1px solid #e0e0e0; background: #f9f9f9;"
                    path_html = f"""
                <div id="user_query_path_{idx}_container" style="margin: 10px 0; padding: 15px; {highlight_style} border-radius: 8px; position: relative; overflow: visible; min-height: 120px;">
                    <div id="user_query_path_{idx}_visualization" style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
                    </div>
                </div>
                <script>
                (function() {{
                    const pathData = {json.dumps(path_data)};
                    const tableMetadata = {json.dumps(table_metadata)};
                    const container = document.getElementById('user_query_path_{idx}_visualization');
                    
                    pathData.tables.forEach((table, i) => {{
                        const tableNode = document.createElement('span');
                        tableNode.className = 'table-node';
                        tableNode.textContent = table;
                        tableNode.style.cssText = 'padding: 8px 12px; background: #fff; border: 2px solid #1e88e5; border-radius: 6px; cursor: pointer; font-weight: 600; transition: all 0.2s;';
                        
                        const tableInfo = tableMetadata[table] || {{columns: [], cardinality: 0, num_columns: 0}};
                        const tooltip = document.createElement('div');
                        tooltip.style.cssText = 'position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 6px; font-size: 12px; z-index: 1000; pointer-events: none; opacity: 0; transition: opacity 0.2s; max-width: 300px;';
                        tooltip.innerHTML = `<strong>${{table}}</strong><br/>Cardinality: ${{tableInfo.cardinality}}<br/>Columns: ${{tableInfo.num_columns}}<br/><small>${{tableInfo.columns.slice(0, 5).join(', ')}}${{tableInfo.columns.length > 5 ? '...' : ''}}</small>`;
                        document.body.appendChild(tooltip);
                        
                        tableNode.addEventListener('mouseenter', (e) => {{
                            tooltip.style.opacity = '1';
                            tooltip.style.left = (e.pageX + 10) + 'px';
                            tooltip.style.top = (e.pageY + 10) + 'px';
                            tableNode.style.background = '#e3f2fd';
                        }});
                        tableNode.addEventListener('mouseleave', () => {{
                            tooltip.style.opacity = '0';
                            tableNode.style.background = '#fff';
                        }});
                        tableNode.addEventListener('mousemove', (e) => {{
                            tooltip.style.left = (e.pageX + 10) + 'px';
                            tooltip.style.top = (e.pageY + 10) + 'px';
                        }});
                        
                        container.appendChild(tableNode);
                        
                        if (i < pathData.tables.length - 1) {{
                            const arrow = document.createElement('span');
                            arrow.textContent = '--';
                            arrow.style.cssText = 'font-size: 20px; color: #666; margin: 0 4px;';
                            
                            const edgeInfo = pathData.joins[i];
                            const edgeTooltip = document.createElement('div');
                            edgeTooltip.style.cssText = 'position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 6px; font-size: 12px; z-index: 1000; pointer-events: none; opacity: 0; transition: opacity 0.2s;';
                            edgeTooltip.innerHTML = `<strong>Join Attributes:</strong><br/>${{edgeInfo.attributes}}<br/><small>Type: ${{edgeInfo.type}}</small>`;
                            document.body.appendChild(edgeTooltip);
                            
                            arrow.addEventListener('mouseenter', (e) => {{
                                edgeTooltip.style.opacity = '1';
                                edgeTooltip.style.left = (e.pageX + 10) + 'px';
                                edgeTooltip.style.top = (e.pageY + 10) + 'px';
                                arrow.style.color = '#1e88e5';
                                arrow.style.fontWeight = 'bold';
                            }});
                            arrow.addEventListener('mouseleave', () => {{
                                edgeTooltip.style.opacity = '0';
                                arrow.style.color = '#666';
                                arrow.style.fontWeight = 'normal';
                            }});
                            arrow.addEventListener('mousemove', (e) => {{
                                edgeTooltip.style.left = (e.pageX + 10) + 'px';
                                edgeTooltip.style.top = (e.pageY + 10) + 'px';
                            }});
                            
                            container.appendChild(arrow);
                        }}
                    }});
                }})();
                </script>
                """
                    components.html(path_html, height=180)

                    if st.button("Show on the map", key=f"show_user_query_path_{idx}"):
                        st.session_state["selected_user_query_path_for_map"] = idx
                        path = parsed_paths[idx - 1] if idx - 1 < len(parsed_paths) else []
                        computed_bbox = _compute_bbox_from_path(path, active_data_lake)
                        if computed_bbox:
                            st.session_state.user_query_path_map_selections[f"user_query_path_{idx}"] = {
                                "bbox": computed_bbox,
                                "region_name": "Computed from path tables",
                                "method": "spatial_coordinates"
                            }
                        else:
                            st.session_state.user_query_path_map_selections[f"user_query_path_{idx}"] = {
                                "bbox": west_lafayette_bbox,
                                "region_name": "Path Region",
                                "method": "default_coordinates"
                            }
                        st.rerun()

        with col_map:
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
                    if isinstance(stored_info, dict):
                        path_region_info = stored_info
                        path_map_bbox = stored_info.get("bbox")
                    else:
                        path_map_bbox = stored_info
                else:
                    path_map_bbox = st.session_state.get("bbox_selection")
            else:
                path_map_bbox = None

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
                    for idx, table_name in enumerate(path):
                        table_colors[table_name] = colors[idx % len(colors)]
                    
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


def _bbox_from_geojson_feature(feature):
    """Extract [south, west, north, east] from a GeoJSON-like feature."""
    if not feature:
        return None
    geom = feature.get("geometry") if isinstance(feature, dict) else None
    if not geom:
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None

    flat = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and all(isinstance(v, (int, float)) for v in node):
                if len(node) >= 2:
                    flat.append((node[0], node[1]))
            else:
                for child in node:
                    walk(child)

    walk(coords)
    if not flat:
        return None
    lons = [p[0] for p in flat]
    lats = [p[1] for p in flat]
    return [min(lats), min(lons), max(lats), max(lons)]
