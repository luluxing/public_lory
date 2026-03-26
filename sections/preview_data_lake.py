import folium
import detect_geo
import ddg_spatial_joins
import geocoding
import graph_joins
import pandas as pd
import pandas_helper
import streamlit as st
import streamlit.components.v1 as components
from sections.geo_ui import extract_point_coords

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False


def _style_preview_table(df: pd.DataFrame, grey_cols=None, bold_cols=None):
    grey_cols = set(grey_cols or [])
    bold_cols = set(bold_cols or [])

    def style_column(col: pd.Series):
        styles = ["" for _ in col]
        if col.name in grey_cols:
            styles = ["color: #b8bec7;" for _ in col]
        if col.name in bold_cols:
            styles = [f"{style} font-weight: 700; color: #0d47a1;".strip() for style in styles]
        return styles

    header_styles = []
    for col in df.columns:
        if col in bold_cols:
            header_styles.append({"selector": f"th.col_heading.level0.col{df.columns.get_loc(col)}", "props": "font-weight: 700; color: #0d47a1;"})
        if col in grey_cols:
            header_styles.append({"selector": f"th.col_heading.level0.col{df.columns.get_loc(col)}", "props": "color: #b8bec7;"})

    return df.style.hide(axis="index").apply(style_column, axis=0).set_table_styles(header_styles, overwrite=False)


def _persist_intermediate_table(table_name: str, df: pd.DataFrame):
    output_path = pandas_helper.temp_table_path(table_name)
    df.to_csv(output_path, index=False)
    return output_path


def _implicit_lane_table_names() -> list[str]:
    geometry_map = ddg_spatial_joins.discover_table_geometries(
        chosen_lake=st.session_state.get("chosen_lake"),
        in_memory_tables=st.session_state.data_lake,
        manual_geo_details=st.session_state.get("ddg_manual_geo_augmentation_details", {}),
    )
    implicit_tables: list[str] = []
    for table_name, df in sorted(st.session_state.data_lake.items()):
        if geometry_map.get(table_name):
            continue
        if detect_geo.has_implicit_geo_columns(df):
            implicit_tables.append(table_name)
    return implicit_tables


def _apply_geocoding_to_table(table_name: str) -> tuple[bool, str | None]:
    table_df = st.session_state.data_lake[table_name]
    preview_df, intermediate_df, _ = geocoding.build_intermediate_geocoded_table(table_df)
    new_geo_columns = geocoding.generated_geo_columns(table_df, intermediate_df)
    if not new_geo_columns:
        return False, None

    geocoding.write_geo_association_details(
        table_name,
        geocoding.generated_geo_association_details(table_df, intermediate_df),
    )
    output_path = _persist_intermediate_table(table_name, intermediate_df)
    st.session_state.data_lake[table_name] = intermediate_df
    st.session_state.preview_display_tables[table_name] = preview_df
    st.session_state.preview_generated_geo_columns[table_name] = new_geo_columns
    if table_name in st.session_state.get("uploaded_csv_files", {}):
        st.session_state.uploaded_csv_files[table_name]["dataframe"] = intermediate_df
    refresh_tables = set(st.session_state.get("ddg_spatial_refresh_tables", []))
    refresh_tables.add(table_name)
    st.session_state.ddg_spatial_refresh_tables = sorted(refresh_tables)
    return True, str(output_path)


def _extract_points_from_column(df, column_name, max_points=500):
    """
    Extract point coordinates from a spatial column.
    Returns: list of (lat, lon) tuples
    """
    all_coords = []

    if column_name not in df.columns:
        return all_coords

    col = df[column_name]

    spatial_cols = graph_joins._find_spatial_columns(df)

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
        sample = col.dropna().astype(str).head(max_points)
        for val in sample:
            coords = extract_point_coords(val)
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
    implicit_lane_tables = _implicit_lane_table_names()

    bulk_button_col, _ = st.columns([3, 10])
    with bulk_button_col:
        st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        apply_all_clicked = st.button(
            "Apply Geo-coding to All",
            key="preview_geocoding_all",
            width="stretch",
            disabled=not implicit_lane_tables,
        )

    if apply_all_clicked:
        updated_tables = []
        for table_name in implicit_lane_tables:
            updated, output_path = _apply_geocoding_to_table(table_name)
            if updated:
                updated_tables.append(table_name)
        if updated_tables:
            table_label = ", ".join(updated_tables)
            st.success(
                "Applied geo-coding to implicit-lane tables: "
                f"{table_label}. Downstream app functions now use the temp-table copies stored under temp_tables/."
            )
            st.rerun()
        st.warning("No new geo columns were generated for the current implicit-lane tables.")

    st.subheader("Data Lake Table 📚")

    selected_table = st.session_state.get("preview_selected_table")
    selected_column = st.session_state.get("preview_selected_column")
    num_rows = st.session_state.get("preview_num_rows", 10)

    if "preview_previous_table" not in st.session_state:
        st.session_state.preview_previous_table = selected_table
    elif st.session_state.preview_previous_table != selected_table:
        st.session_state.preview_selected_column = None
        st.session_state.preview_previous_table = selected_table

    if selected_table and selected_table in st.session_state.data_lake:
        table_df = st.session_state.data_lake[selected_table]
        preview_df = st.session_state.get("preview_display_tables", {}).get(selected_table, table_df)
        grey_cols = []
        bold_cols = []

        header_col, button_col = st.columns([10, 3])
        with header_col:
            st.markdown(
                f'<div style="border: 2px solid #0d47a1; border-radius: 6px; padding: 12px; background-color: #e3f2fd; margin-bottom: 20px;">'
                f'<strong style="font-size: 1.1rem; color: #0d47a1;">{selected_table}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with button_col:
            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
            geocode_clicked = st.button(
                "Apply geocoding",
                key=f"preview_geocoding_{selected_table}",
                width="stretch",
                disabled=not geocoding.table_supports_geocoding(table_df),
            )

        if geocoding.table_supports_geocoding(table_df) and geocode_clicked:
            updated, output_path = _apply_geocoding_to_table(selected_table)
            if updated:
                st.success(
                    "Geocoding complete. Preview keeps source columns in light grey, geo columns are appended in bold, "
                    f"and downstream app functions now use the temp-table copy stored at {output_path}."
                )
                st.rerun()
            st.warning("No new geo columns were generated for this table.")
        elif selected_table in st.session_state.get("preview_display_tables", {}):
            preview_df = st.session_state.preview_display_tables[selected_table]
            grey_cols = geocoding.find_geocoding_source_columns(preview_df)
            bold_cols = st.session_state.get("preview_generated_geo_columns", {}).get(selected_table, [])
            st.caption("Geocoding already applied to this table in the current session. Downstream functions use the intermediate version.")
        elif graph_joins._find_spatial_columns(table_df):
            preview_df = table_df
            bold_cols = [col for col in [geocoding.POINT_COLUMN, geocoding.POLYGON_COLUMN] if col in table_df.columns]
            st.caption("This table already contains geo columns.")
        st.table(_style_preview_table(preview_df.head(num_rows), grey_cols=grey_cols, bold_cols=bold_cols))

        table_df = st.session_state.data_lake[selected_table]
        column_options = [None] + list(table_df.columns)
        current_index = 0
        if selected_column in column_options:
            current_index = column_options.index(selected_column)

        selected_column = st.selectbox(
            "Select a column to visualize on map",
            options=column_options,
            index=current_index,
            key="preview_column_select",
        )
        st.session_state.preview_selected_column = selected_column

        is_spatial = False
        if selected_column:
            spatial_cols = graph_joins._find_spatial_columns(table_df)
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

            if selected_column in spatial_cols or selected_column == lat_col or selected_column == lon_col:
                is_spatial = True

        if selected_column and not is_spatial:
            st.info("No spatial attributes found")
        elif selected_column and is_spatial:
            points = _extract_points_from_column(table_df, selected_column, max_points=500)

            if points:
                bbox = _compute_bbox_from_coords(points)

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
                center_lat = 40.4272
                center_lon = -86.9158
                zoom_level = 12
                bbox = None

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_level,
                tiles="OpenStreetMap",
            )

            if bbox:
                folium.Rectangle(
                    bounds=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    color="#0066cc",
                    fillColor="#0066cc",
                    fillOpacity=0.3,
                    weight=2,
                    opacity=0.8,
                    popup=f"Bounding box for {selected_column}",
                    tooltip="Bounding box",
                ).add_to(m)

            for lat, lon in points:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    popup=f"({lat:.4f}, {lon:.4f})",
                    tooltip=f"({lat:.4f}, {lon:.4f})",
                    color="#0066cc",
                    fillColor="#0066cc",
                    fillOpacity=0.7,
                    weight=1,
                ).add_to(m)

            if HAS_ST_FOLIUM:
                st_folium(m, width=None, height=700, returned_objects=[])
            else:
                map_html = m._repr_html_()
                components.html(map_html, height=700)
    elif st.session_state.data_lake:
        st.info("Please select a table from the sidebar.")
    else:
        st.info("No data lake loaded.")
