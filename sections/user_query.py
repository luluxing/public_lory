import json
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

    # Initialize session state for next step
    if "show_target_attributes" not in st.session_state:
        st.session_state["show_target_attributes"] = False

    # Show "Next step" button after bbox selection
    if st.button("Next step", key="next_step_button"):
        st.session_state["show_target_attributes"] = True
        st.rerun()

    # Only show target attributes section after clicking "Next step"
    if not st.session_state["show_target_attributes"]:
        return

    st.markdown("**Step 2: Target attributes**")
    st.markdown(
        """
        <style>
        div[data-testid="InputInstructions"],
        div[data-testid="stTextInputInstructions"] {
            display: none !important;
        }
        div[data-testid="column"]:has(button[key="target_attr_add"]) button {
            height: 38px;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    def _clean_attribute_label(value):
        value = value.strip()
        value = re.sub(r"^(closest|nearest)\s+", "", value, flags=re.IGNORECASE)
        return value

    def _submit_target_attributes():
        raw_items = [
            item.strip()
            for item in st.session_state.get("target_attr_input", "").split(",")
        ]
        for raw in raw_items:
            cleaned = _clean_attribute_label(raw)
            if cleaned and cleaned not in st.session_state.target_schema:
                st.session_state.target_schema.append(cleaned)
                st.session_state.target_schema_raw.append(raw)
        st.session_state["target_attr_input"] = ""

    col_input, col_button = st.columns([4, 1])
    with col_input:
        st.text_area(
            "Target attributes",
            key="target_attr_input",
            placeholder="e.g., accident_id, date, victim_type, hospitals.name",
            label_visibility="collapsed",
            height=90,
        )
    with col_button:
        st.button(
            "Submit",
            use_container_width=True,
            key="target_attr_submit",
            on_click=_submit_target_attributes,
        )
        st.markdown(
            """
            <style>
            div[data-testid="column"]:has(button[key="target_attr_submit"]) button {
                height: 38px;
                margin-top: 4px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

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

    if "spatial_choice_mode" not in st.session_state:
        st.session_state["spatial_choice_mode"] = "inferred"
    if "spatial_manual_mode" not in st.session_state:
        st.session_state["spatial_manual_mode"] = "exclude"
    if "spatial_manual_distance_km" not in st.session_state:
        st.session_state["spatial_manual_distance_km"] = None

    inferred_inputs = st.session_state.target_schema_raw or st.session_state.target_schema
    inferred = infer_spatial_preferences(inferred_inputs)
    st.markdown("**Spatial join preference**")
    if inferred["mode"] is None:
        st.caption("Inferred predicate: none")
    elif inferred["mode"] == "distance":
        if inferred["distance_km"]:
            st.caption(f"Inferred predicate: distance ({inferred['distance_km']:.1f} km)")
        else:
            st.caption("Inferred predicate: distance (closest)")
    else:
        st.caption(f"Inferred predicate: {inferred['mode']}")

    choice = st.radio(
        "Spatial predicate choice",
        options=["Use inferred", "Choose manually"],
        horizontal=True,
        index=0 if st.session_state["spatial_choice_mode"] == "inferred" else 1,
        key="spatial_choice_radio",
        label_visibility="collapsed",
    )
    st.session_state["spatial_choice_mode"] = "inferred" if choice == "Use inferred" else "manual"

    if st.session_state["spatial_choice_mode"] == "manual":
        manual_mode = st.selectbox(
            "Spatial predicate",
            options=["exclude", "contain", "intersect", "distance"],
            index=["exclude", "contain", "intersect", "distance"].index(
                st.session_state["spatial_manual_mode"]
            ),
            key="spatial_manual_mode_select",
            label_visibility="collapsed",
        )
        st.session_state["spatial_manual_mode"] = manual_mode
        if manual_mode == "distance":
            st.session_state["spatial_manual_distance_km"] = st.number_input(
                "Distance (km)",
                min_value=0.0,
                value=st.session_state["spatial_manual_distance_km"] or 0.0,
                step=1.0,
            )
        else:
            st.session_state["spatial_manual_distance_km"] = None

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
            st.dataframe(tuples_df, use_container_width=True)
        else:
            # Show empty table with headers only
            empty_df = pd.DataFrame(columns=col_names)
            st.dataframe(empty_df, use_container_width=True)
        
        if st.session_state.get("generated_tables_used"):
            tables_used = ", ".join(st.session_state["generated_tables_used"])
            st.caption(f"Tables used: {tables_used}")

    path_tables = st.session_state.get("generated_join_path_tables", [])
    if path_tables:
        path_str = " → ".join(path_tables)
        st.session_state["user_query_paths"] = [path_str]
        st.session_state["user_query_parsed_paths"] = [path_tables]

    if "user_query_path_map_selections" not in st.session_state:
        st.session_state["user_query_path_map_selections"] = {}

    if "selected_user_query_path_for_map" not in st.session_state:
        st.session_state["selected_user_query_path_for_map"] = None

    parsed_paths = st.session_state.get("user_query_parsed_paths", [])

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
        for table_name in st.session_state.data_lake.keys():
            df = st.session_state.data_lake[table_name]
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
                        join_info = get_join_info_user_query(path[i], path[i+1])
                        path_data["joins"].append({
                            "from": path[i],
                            "to": path[i+1],
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
                        
                        const tableInfo = tableMetadata[table];
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
                        # Compute tight bounding box from points in the path
                        path = parsed_paths[idx - 1] if idx - 1 < len(parsed_paths) else []
                        computed_bbox = _compute_bbox_from_path(path, st.session_state.data_lake)
                        if computed_bbox:
                            st.session_state.user_query_path_map_selections[f"user_query_path_{idx}"] = {
                                "bbox": computed_bbox,
                                "region_name": "Computed from path tables",
                                "method": "spatial_coordinates"
                            }
                        else:
                            # Fallback to default if no spatial data found
                            st.session_state.user_query_path_map_selections[f"user_query_path_{idx}"] = {
                                "bbox": west_lafayette_bbox,
                                "region_name": "Path Region",
                                "method": "default_coordinates"
                            }
                        st.rerun()

        with col_map:
            st.markdown("**Map View**")
            map_height = 700
            # Inject CSS to set iframe height for this map
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
                    # Fallback to bbox_selection if no stored info
                    path_map_bbox = st.session_state.get("bbox_selection")
            else:
                path_map_bbox = None

            # Use bbox_selection (drawn bounding box) for filtering points
            bbox_for_filtering = st.session_state.get("bbox_selection")
            
            # Calculate center and zoom based on the bounding box
            if path_map_bbox:
                center_lat = (path_map_bbox[0] + path_map_bbox[2]) / 2
                center_lon = (path_map_bbox[1] + path_map_bbox[3]) / 2
                # Calculate appropriate zoom level based on bbox size
                lat_range = path_map_bbox[2] - path_map_bbox[0]
                lon_range = path_map_bbox[3] - path_map_bbox[1]
                max_range = max(lat_range, lon_range)
                # Adjust zoom based on bbox size (smaller bbox = higher zoom)
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
                
                # Plot points from tables in the path
                if selected_path_idx and parsed_paths:
                    path = parsed_paths[selected_path_idx - 1] if selected_path_idx - 1 < len(parsed_paths) else []
                    points = _extract_points_from_path(path, st.session_state.data_lake, max_points_per_table=200)
                    
                    # Group points by table for color coding
                    table_colors = {}
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                             'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 
                             'lightgreen', 'gray', 'black', 'lightgray']
                    for idx, table_name in enumerate(path):
                        table_colors[table_name] = colors[idx % len(colors)]
                    
                    # Plot points with different colors for each table
                    # Filter by bbox_selection (the drawn bounding box)
                    if bbox_for_filtering:
                        margin = 0.01
                        for lat, lon, table_name in points:
                            # Check if point is within bounding box (with small margin)
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
                        # If no bbox_selection, plot all points
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

            # Use st_folium if available for better height control, otherwise use components.html
            if HAS_ST_FOLIUM:
                st_folium(m_path, width=None, height=map_height, returned_objects=[])
            else:
                map_html = m_path._repr_html_()
                # Modify folium HTML to set map div height
                # Replace height in map div style
                map_html = re.sub(
                    r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*height:\s*)\d+px',
                    lambda m: m.group(1) + f"{map_height}px",
                    map_html,
                )
                # Add height to style if not present
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
                # Add JavaScript to force resize after component renders
                st.markdown(
                    f'<script>(function(){{function forceResize(){{var iframes=document.querySelectorAll("iframe");iframes.forEach(function(iframe){{iframe.style.setProperty("height","{map_height}px","important");iframe.style.setProperty("min-height","{map_height}px","important");iframe.style.setProperty("max-height","{map_height}px","important");iframe.setAttribute("height","{map_height}");}});}}forceResize();setTimeout(forceResize,100);setTimeout(forceResize,500);setTimeout(forceResize,1000);setTimeout(forceResize,2000);var observer=new MutationObserver(forceResize);observer.observe(document.body,{{childList:true,subtree:true}});}})();</script>',
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
