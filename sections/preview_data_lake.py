import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

import graph_joins


def _extract_point_coords(point_str):
    """Extract lat, lon from POINT(lon lat) or similar formats."""
    if not isinstance(point_str, str):
        return None
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', point_str, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lat, lon)
    return None


def _extract_points_from_column(df, column_name, max_points=500):
    """
    Extract point coordinates from a spatial column.
    Returns: list of (lat, lon) tuples
    """
    all_coords = []
    
    if column_name not in df.columns:
        return all_coords
    
    col = df[column_name]
    
    # Check if this is part of a lat/lon pair
    spatial_cols = graph_joins._find_spatial_columns(df)
    
    # Find lat/lon columns
    lat_col = None
    lon_col = None
    for col_name in df.columns:
        col_lower = col_name.lower()
        if lat_col is None and "lat" in col_lower:
            sample = pd.to_numeric(df[col_name].dropna().head(10), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-90, 90).all():
                lat_col = col_name
        if lon_col is None and ("lon" in col_lower or "lng" in col_lower):
            sample = pd.to_numeric(df[col_name].dropna().head(10), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-180, 180).all():
                lon_col = col_name
    
    # If selected column is lat or lon, use the pair
    if column_name == lat_col and lon_col:
        try:
            coords_df = df[[lat_col, lon_col]].dropna()
            for _, row in coords_df.head(max_points).iterrows():
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        all_coords.append((lat, lon))
                except Exception:
                    continue
        except Exception:
            pass
    elif column_name == lon_col and lat_col:
        try:
            coords_df = df[[lat_col, lon_col]].dropna()
            for _, row in coords_df.head(max_points).iterrows():
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        all_coords.append((lat, lon))
                except Exception:
                    continue
        except Exception:
            pass
    elif column_name in spatial_cols:
        # Extract from POINT format columns
        sample = col.dropna().astype(str).head(max_points)
        for val in sample:
            coords = _extract_point_coords(val)
            if coords:
                all_coords.append(coords)
    
    return all_coords


def _compute_bbox_from_coords(coords):
    """
    Compute bounding box from coordinates.
    Returns: [south, west, north, east] or None
    """
    if not coords:
        return None
    
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    bbox = [min(lats), min(lons), max(lats), max(lons)]
    return bbox


def render_preview_section():
    """Render the preview tables section."""
    st.markdown('<div id="preview-data-lake"></div>', unsafe_allow_html=True)
    st.subheader("Data Lake Table 📚")
    
    selected_table = st.session_state.get("preview_selected_table")
    selected_column = st.session_state.get("preview_selected_column")
    num_rows = st.session_state.get("preview_num_rows", 10)
    
    # Track previous table to reset column when table changes
    if "preview_previous_table" not in st.session_state:
        st.session_state.preview_previous_table = selected_table
    elif st.session_state.preview_previous_table != selected_table:
        # Table changed, reset column selection
        st.session_state.preview_selected_column = None
        st.session_state.preview_previous_table = selected_table
    
    if selected_table and selected_table in st.session_state.data_lake:
        # Wrap table name in a box
        st.markdown(
            f'<div style="border: 2px solid #0d47a1; border-radius: 6px; padding: 12px; background-color: #e3f2fd; margin-bottom: 20px;">'
            f'<strong style="font-size: 1.1rem; color: #0d47a1;">{selected_table}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )
        table_df = st.session_state.data_lake[selected_table]
        # Show first N rows
        st.dataframe(table_df.head(num_rows))
        
        # Column selector
        column_options = [None] + list(table_df.columns)
        current_index = 0
        if selected_column in column_options:
            current_index = column_options.index(selected_column)
        
        selected_column = st.selectbox(
            "Select a column to visualize on map",
            options=column_options,
            index=current_index,
            key="preview_column_select"
        )
        st.session_state.preview_selected_column = selected_column
        
        # Check if column is spatial and display map
        is_spatial = False
        if selected_column:
            spatial_cols = graph_joins._find_spatial_columns(table_df)
            # Check if selected column is spatial (could be lat/lon pair or POINT column)
            lat_col = None
            lon_col = None
            for col in table_df.columns:
                col_lower = col.lower()
                if lat_col is None and "lat" in col_lower:
                    sample = pd.to_numeric(table_df[col].dropna().head(10), errors="coerce").dropna()
                    if len(sample) > 0 and sample.between(-90, 90).all():
                        lat_col = col
                if lon_col is None and ("lon" in col_lower or "lng" in col_lower):
                    sample = pd.to_numeric(table_df[col].dropna().head(10), errors="coerce").dropna()
                    if len(sample) > 0 and sample.between(-180, 180).all():
                        lon_col = col
            
            # Column is spatial if it's in spatial_cols or is part of lat/lon pair
            if selected_column in spatial_cols or selected_column == lat_col or selected_column == lon_col:
                is_spatial = True
        
        # Display message or map
        if selected_column and not is_spatial:
            st.info("No spatial attributes found")
        elif selected_column and is_spatial:
            # Extract points from the column
            points = _extract_points_from_column(table_df, selected_column, max_points=500)
            
            if points:
                # Compute bounding box
                bbox = _compute_bbox_from_coords(points)
                
                # Calculate center and zoom
                if bbox:
                    center_lat = (bbox[0] + bbox[2]) / 2
                    center_lon = (bbox[1] + bbox[3]) / 2
                    lat_range = bbox[2] - bbox[0]
                    lon_range = bbox[3] - bbox[1]
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
                    center_lat = 40.4272
                    center_lon = -86.9158
                    zoom_level = 12
            else:
                # No points found, use default location
                center_lat = 40.4272
                center_lon = -86.9158
                zoom_level = 12
                bbox = None
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_level,
                tiles='OpenStreetMap'
            )
            
            # Add bounding box if available
            if bbox:
                folium.Rectangle(
                    bounds=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    color="#0066cc",
                    fillColor="#0066cc",
                    fillOpacity=0.3,
                    weight=2,
                    opacity=0.8,
                    popup=f"Bounding box for {selected_column}",
                    tooltip=f"Bounding box"
                ).add_to(m)
            
            # Add points to map
            if points:
                for lat, lon in points:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f"({lat:.4f}, {lon:.4f})",
                        tooltip=f"({lat:.4f}, {lon:.4f})",
                        color="#0066cc",
                        fillColor="#0066cc",
                        fillOpacity=0.7,
                        weight=1
                    ).add_to(m)
            
            # Display map
            if HAS_ST_FOLIUM:
                st_folium(m, width=None, height=700, returned_objects=[])
            else:
                map_html = m._repr_html_()
                components.html(map_html, height=700)
        elif selected_column is None:
            # No column selected, show empty map or no map
            pass
    
    elif st.session_state.data_lake:
        st.info("Please select a table from the sidebar.")
    else:
        st.info("No data lake loaded.")
