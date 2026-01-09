import os
import re
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from streamlit_scroll_to_top import scroll_to_here
import pandas as pd

import pandas_helper
from data_path import data_path
from sections.data_discovery_graph import render_data_discovery_graph_section
from sections.explore_data_lake import render_explore_section
from sections.preview_data_lake import render_preview_section
from sections.user_query import render_user_query_section

# ───────────────────── Session-state defaults ────────────────────────────────
if "data_lake" not in st.session_state:
    st.session_state.data_lake = {}

if "uploaded_table" not in st.session_state:
    st.session_state.uploaded_table = None

if "input_table_files" not in st.session_state:
    st.session_state.input_table_files = []

if "chosen_lake" not in st.session_state:
    st.session_state.chosen_lake = None

if "graph_nodes" not in st.session_state:
    st.session_state.graph_nodes = []

if "graph_rel_edges" not in st.session_state:
    st.session_state.graph_rel_edges = []

if "graph_spatial_edges" not in st.session_state:
    st.session_state.graph_spatial_edges = []

if "target_schema" not in st.session_state:
    st.session_state.target_schema = []
if "target_schema_raw" not in st.session_state:
    st.session_state.target_schema_raw = []
if "user_query_target_attr_input" not in st.session_state:
    st.session_state.user_query_target_attr_input = ""

if "selected_section" not in st.session_state:
    st.session_state.selected_section = None

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False
if "lake_loaded" not in st.session_state:
    st.session_state.lake_loaded = False
if "lake_loaded_name" not in st.session_state:
    st.session_state.lake_loaded_name = None
if "in_function_view" not in st.session_state:
    st.session_state.in_function_view = False
if "preview_num_rows" not in st.session_state:
    st.session_state.preview_num_rows = 10
if "preview_selected_table" not in st.session_state:
    st.session_state.preview_selected_table = None
if "graph_equi_joinability_measure" not in st.session_state:
    st.session_state.graph_equi_joinability_measure = 0.0
if "graph_spatial_predicate" not in st.session_state:
    st.session_state.graph_spatial_predicate = "Containment"
if "graph_distance_km" not in st.session_state:
    st.session_state.graph_distance_km = 1.0
if "path_min_len" not in st.session_state:
    st.session_state.path_min_len = ""
if "path_max_len" not in st.session_state:
    st.session_state.path_max_len = ""
if "path_num_relational_joins" not in st.session_state:
    st.session_state.path_num_relational_joins = 0
if "path_num_spatial_joins" not in st.session_state:
    st.session_state.path_num_spatial_joins = 0
if "path_spatial_join_type" not in st.session_state:
    st.session_state.path_spatial_join_type = "Containment"

# ───────────────────── Helpers ──────────────────────────────────────────────
# ───────────────────── Geographic defaults ──────────────────────────────────
# West Lafayette area bbox [south, west, north, east]
# Left upper corner: (40.438092, -86.925687) = Northwest corner
# Right lower corner: (40.416390, -86.905942) = Southeast corner
WEST_LAFAYETTE_BBOX = [40.416390, -86.925687, 40.438092, -86.905942]
# Default Lafayette, IN bbox for map view (same as West Lafayette area)
LAFAYETTE_DEFAULT_BBOX = [40.416390, -86.925687, 40.438092, -86.905942]

# ───────────────────── State reset helpers ──────────────────────────────────
def reset_state_for_new_lake():
    """Reset app state when the data-lake selection changes."""
    st.session_state.data_lake = {}
    st.session_state.uploaded_table = None
    st.session_state.input_table_files = []
    st.session_state.graph_nodes = []
    st.session_state.graph_rel_edges = []
    st.session_state.graph_spatial_edges = []
    st.session_state.selected_section = None
    st.session_state.in_function_view = False


def reset_state_for_new_input_table():
    """Reset state when the input table selection changes within a lake."""
    st.session_state.uploaded_table = None


# ───────────────────── Page config & theme tweaks ────────────────────────────
st.set_page_config(page_title="Lory", layout="wide")

st.markdown(
    """
<style>
div.stButton > button[kind="primary"] {
    background-color: #e3f2fd;
    color: #0d47a1;
    border: 1px solid #90caf9;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #bbdefb;
}
/* Make select boxes show a pointer cursor */
.stSelectbox div[data-baseweb="select"] > div {
    cursor: pointer !important;
}
.stSelectbox svg, .stSelectbox span {
    cursor: pointer !important;
}
/* Fix map width in columns - preserve height */
iframe[title*="leaflet"],
iframe[title*="folium"],
iframe[src*="leaflet"],
iframe[src*="folium"] {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    min-height: 700px !important;
}
/* Ensure map containers maintain height */
div[data-testid="stLeafletMap"] {
    min-height: 700px !important;
}
div[data-testid="column"] {
    width: 100% !important;
}
div.element-container {
    width: 100% !important;
    max-width: 100% !important;
}
/* Sidebar navigation buttons */
div[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid #cfd8dc;
    background: #ffffff;
    color: #0d47a1;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
    background: #e3f2fd;
    border-color: #90caf9;
}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────── Top banner ────────────────────────────────────────────
st.markdown(
    """
<div style="display: inline-flex; align-items: center; justify-content: flex-start; margin-bottom: 15px; margin-top: -12px;">
    <span style="font-size: 42px; font-weight: bold;">Lory</span>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<p style="font-weight: 600; font-size: 1.1rem;">
    Location-aware Data Discovery from Data Lakes
</p>
""",
    unsafe_allow_html=True,
)

# ───────────────────── Data-lake picker ──────────────────────────────────────
DATALAKES_ROOT = data_path() / "datalakes"

st.sidebar.header("🏞️  Data-lake")

if not DATALAKES_ROOT.exists():
    st.warning(f"No data-lakes found under {DATALAKES_ROOT}.")
    st.stop()

datalake_opts = [
    d
    for d in os.listdir(DATALAKES_ROOT)
    if (Path(DATALAKES_ROOT) / d).is_dir()
]
if not datalake_opts:
    datalake_opts = ["<no datalakes found>"]

PREFERRED_DEFAULT_LAKE = "lory_vision_lake"
if PREFERRED_DEFAULT_LAKE in datalake_opts:
    datalake_opts = [PREFERRED_DEFAULT_LAKE] + [
        d for d in datalake_opts if d != PREFERRED_DEFAULT_LAKE
    ]

chosen_lake = st.sidebar.selectbox(
    "Select a data lake",
    options=datalake_opts,
    index=None,
    placeholder="Choose a data lake…",
)

# Load / switch data-lake
if chosen_lake:
    if (
        "chosen_lake" not in st.session_state
        or st.session_state.chosen_lake != chosen_lake
    ):
        reset_state_for_new_lake()
        st.session_state.chosen_lake = chosen_lake

        if chosen_lake != "<no datalakes found>":
            lake_dir = Path(DATALAKES_ROOT) / chosen_lake
            input_dir = lake_dir / "input_tables"

            with st.spinner(f"Loading data lake '{chosen_lake}' …"):
                st.session_state.data_lake = pandas_helper.load_datalake_tables(
                    lake_dir=lake_dir
                )

            if input_dir.exists():
                st.session_state.input_table_files = list(input_dir.glob("*.csv"))
            else:
                st.session_state.input_table_files = []

            st.session_state.lake_loaded = True
            st.session_state.lake_loaded_name = chosen_lake

            st.success(
                f"✅ Loaded data lake “{chosen_lake}” "
                f"({len(st.session_state.data_lake)} tables)."
            )
        else:
            st.warning("No data-lake directories found under ./datalakes/.")

def _set_section(name: str):
    st.session_state.selected_section = name
    st.session_state.scroll_to_top = True
    st.session_state.in_function_view = True


def _back_to_functions():
    st.session_state.in_function_view = False
    st.session_state.scroll_to_top = True


# ───────────────────── Sidebar navigation ───────────────────────────────────
if st.session_state.data_lake:
    if st.session_state.lake_loaded:
        st.sidebar.success(
            f"Loaded \"{st.session_state.lake_loaded_name}\" "
            f"({len(st.session_state.data_lake)} tables)"
        )
    
    if not st.session_state.in_function_view:
        # Top level: Show function selection buttons
        st.sidebar.markdown("### Functions")
        st.sidebar.button(
            "Preview Data Lake",
            use_container_width=True,
            key="nav_preview",
            on_click=_set_section,
            args=("preview",),
        )
        st.sidebar.button(
            "Data Discovery Graph",
            use_container_width=True,
            key="nav_graph",
            on_click=_set_section,
            args=("graph",),
        )
        st.sidebar.button(
            "Explore Data Lake",
            use_container_width=True,
            key="nav_explore",
            on_click=_set_section,
            args=("explore",),
        )
        st.sidebar.button(
            "User Query",
            use_container_width=True,
            key="nav_user_query",
            on_click=_set_section,
            args=("user_query",),
        )
    else:
        # Function view: Show function-specific parameters and back button
        section_names = {
            "preview": "Preview Data Lake",
            "graph": "Data Discovery Graph",
            "explore": "Explore Data Lake",
            "user_query": "User Query"
        }
        current_section_name = section_names.get(st.session_state.selected_section, "Function")
        
        st.sidebar.markdown(f"### {current_section_name}")
        st.sidebar.button(
            "Back to Other Functions",
            use_container_width=True,
            key="nav_back",
            on_click=_back_to_functions,
        )
        st.sidebar.divider()
        
        # Function-specific parameter selection
        if st.session_state.selected_section == "preview":
            st.sidebar.markdown("**Preview Data Lake Parameters**")
            # Number input for number of rows to show
            st.session_state.preview_num_rows = st.sidebar.number_input(
                "Number of rows to show",
                min_value=1,
                value=st.session_state.preview_num_rows,
                key="preview_num_rows_input"
            )
            # Dropdown for table selection
            datalake_table_names = list(st.session_state.data_lake.keys()) if st.session_state.data_lake else []
            if datalake_table_names:
                # Determine initial index
                if st.session_state.preview_selected_table in datalake_table_names:
                    default_index = datalake_table_names.index(st.session_state.preview_selected_table)
                else:
                    default_index = 0
                st.session_state.preview_selected_table = st.sidebar.selectbox(
                    "Select table",
                    options=datalake_table_names,
                    index=default_index,
                    key="preview_table_select"
                )
            else:
                st.sidebar.info("No tables available")
                st.session_state.preview_selected_table = None
        elif st.session_state.selected_section == "graph":
            st.sidebar.markdown("**Data Discovery Graph Parameters**")
            # Equi-Joinability Measure (threshold for join feasibility, 0.0 to 1.0)
            st.session_state.graph_equi_joinability_measure = st.sidebar.slider(
                "Equi-Joinability Measure",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.graph_equi_joinability_measure,
                step=0.01,
                help="Threshold (0.0-1.0) for join feasibility based on column value overlap. Higher values require more matching values.",
                key="graph_equi_joinability_slider"
            )
            # Spatial join predicate selection
            st.session_state.graph_spatial_predicate = st.sidebar.radio(
                "Spatial join predicate selection",
                options=["Containment", "Overlapping", "Distance"],
                index=["Containment", "Overlapping", "Distance"].index(st.session_state.graph_spatial_predicate) if st.session_state.graph_spatial_predicate in ["Containment", "Overlapping", "Distance"] else 0,
                key="graph_spatial_predicate_radio"
            )
            # Distance input (only shown if Distance is selected)
            if st.session_state.graph_spatial_predicate == "Distance":
                st.session_state.graph_distance_km = st.sidebar.number_input(
                    "Distance (kilometers)",
                    min_value=0.0,
                    value=st.session_state.graph_distance_km,
                    key="graph_distance_input"
                )
            
            st.sidebar.divider()
            st.sidebar.markdown("**Join Path Parameters**")
            st.session_state.path_min_len = st.sidebar.text_input(
                "Minimum path length",
                value=st.session_state.path_min_len,
                key="path_min_len_input"
            )
            st.session_state.path_max_len = st.sidebar.text_input(
                "Maximum path length",
                value=st.session_state.path_max_len,
                key="path_max_len_input"
            )
            st.session_state.path_num_relational_joins = st.sidebar.number_input(
                "Number of relational joins",
                min_value=0,
                value=st.session_state.path_num_relational_joins,
                key="path_num_relational_joins_input"
            )
            st.session_state.path_num_spatial_joins = st.sidebar.number_input(
                "Number of spatial joins",
                min_value=0,
                value=st.session_state.path_num_spatial_joins,
                key="path_num_spatial_joins_input"
            )
            st.session_state.path_spatial_join_type = st.sidebar.selectbox(
                "Spatial join type",
                options=["Containment", "Overlapping", "Distance"],
                index=["Containment", "Overlapping", "Distance"].index(st.session_state.path_spatial_join_type) if st.session_state.path_spatial_join_type in ["Containment", "Overlapping", "Distance"] else 0,
                key="path_spatial_join_type_select"
            )
        elif st.session_state.selected_section == "explore":
            # No parameters for Explore Data Lake
            pass
        elif st.session_state.selected_section == "user_query":
            st.sidebar.markdown("**User Query Parameters**")
            st.sidebar.markdown("**Target Attributes**")
            
            # Option to input attributes or upload file
            input_method = st.sidebar.radio(
                "Input method",
                options=["Text Input", "Upload File"],
                key="user_query_input_method"
            )
            
            if input_method == "Text Input":
                st.session_state.user_query_target_attr_input = st.sidebar.text_area(
                    "Target attributes",
                    value=st.session_state.user_query_target_attr_input,
                    placeholder="e.g., accident_id, date, victim_type, hospitals.name",
                    height=100,
                    key="user_query_target_attr_text"
                )
                if st.sidebar.button("Submit", key="user_query_target_attr_submit", use_container_width=True):
                    def _clean_attribute_label(value):
                        value = value.strip()
                        value = re.sub(r"^(closest|nearest)\s+", "", value, flags=re.IGNORECASE)
                        return value
                    
                    raw_items = [
                        item.strip()
                        for item in st.session_state.user_query_target_attr_input.split(",")
                        if item.strip()
                    ]
                    for raw in raw_items:
                        cleaned = _clean_attribute_label(raw)
                        if cleaned and cleaned not in st.session_state.target_schema:
                            st.session_state.target_schema.append(cleaned)
                            st.session_state.target_schema_raw.append(raw)
                    st.session_state.user_query_target_attr_input = ""
                    st.rerun()
            else:  # Upload File
                uploaded_file = st.sidebar.file_uploader(
                    "Upload attributes file",
                    type=['txt', 'csv'],
                    key="user_query_target_attr_file"
                )
                if uploaded_file is not None:
                    # Check if we've already processed this file (using name and size as identifier)
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    if "last_processed_file" not in st.session_state or st.session_state.get("last_processed_file") != file_id:
                        def _clean_attribute_label(value):
                            value = value.strip()
                            value = re.sub(r"^(closest|nearest)\s+", "", value, flags=re.IGNORECASE)
                            return value
                        
                        # Read file content
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                                # Assume attributes are in the first column
                                raw_items = [str(item).strip() for item in df.iloc[:, 0].dropna() if str(item).strip()]
                            else:
                                content = uploaded_file.read().decode('utf-8')
                                raw_items = [item.strip() for item in content.split('\n') if item.strip()]
                                # Also try comma-separated if only one line
                                if len(raw_items) == 1 and ',' in raw_items[0]:
                                    raw_items = [item.strip() for item in raw_items[0].split(',') if item.strip()]
                            
                            # Add to target schema
                            added_count = 0
                            for raw in raw_items:
                                cleaned = _clean_attribute_label(raw)
                                if cleaned and cleaned not in st.session_state.target_schema:
                                    st.session_state.target_schema.append(cleaned)
                                    st.session_state.target_schema_raw.append(raw)
                                    added_count += 1
                            
                            st.session_state["last_processed_file"] = file_id
                            if added_count > 0:
                                st.sidebar.success(f"Loaded {added_count} attributes from file")
                                st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"Error reading file: {str(e)}")
            
            st.sidebar.divider()
            st.sidebar.markdown("**Spatial Join Preference**")
            
            # Initialize spatial join preference state
            if "spatial_choice_mode" not in st.session_state:
                st.session_state["spatial_choice_mode"] = "inferred"
            if "spatial_manual_mode" not in st.session_state:
                st.session_state["spatial_manual_mode"] = "exclude"
            if "spatial_manual_distance_km" not in st.session_state:
                st.session_state["spatial_manual_distance_km"] = None
            
            # Get inferred preferences
            inferred_inputs = st.session_state.target_schema_raw or st.session_state.target_schema
            from sections.generate_target_schema import infer_spatial_preferences
            inferred = infer_spatial_preferences(inferred_inputs)
            
            if inferred["mode"] is None:
                st.sidebar.caption("Inferred predicate: none")
            elif inferred["mode"] == "distance":
                if inferred["distance_km"]:
                    st.sidebar.caption(f"Inferred predicate: distance ({inferred['distance_km']:.1f} km)")
                else:
                    st.sidebar.caption("Inferred predicate: distance (closest)")
            else:
                st.sidebar.caption(f"Inferred predicate: {inferred['mode']}")
            
            choice = st.sidebar.radio(
                "Spatial predicate choice",
                options=["Use inferred", "Choose manually"],
                index=0 if st.session_state["spatial_choice_mode"] == "inferred" else 1,
                key="spatial_choice_radio",
            )
            st.session_state["spatial_choice_mode"] = "inferred" if choice == "Use inferred" else "manual"
            
            if st.session_state["spatial_choice_mode"] == "manual":
                manual_mode = st.sidebar.selectbox(
                    "Spatial predicate",
                    options=["exclude", "contain", "intersect", "distance"],
                    index=["exclude", "contain", "intersect", "distance"].index(
                        st.session_state["spatial_manual_mode"]
                    ),
                    key="spatial_manual_mode_select",
                )
                st.session_state["spatial_manual_mode"] = manual_mode
                if manual_mode == "distance":
                    st.session_state["spatial_manual_distance_km"] = st.sidebar.number_input(
                        "Distance (km)",
                        min_value=0.0,
                        value=st.session_state["spatial_manual_distance_km"] or 0.0,
                        step=1.0,
                        key="spatial_manual_distance_input"
                    )
                else:
                    st.session_state["spatial_manual_distance_km"] = None
    
    st.sidebar.divider()

# Smooth-scroll to top when switching sections using streamlit-scroll-to-top
if st.session_state.get("scroll_to_top"):
    scroll_to_here(0, key="scroll_top_helper")
    st.session_state.scroll_to_top = False

# ───────────────────── Main page: Show data-lake tables ──────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "preview":
    render_preview_section()

# ───────────────────── Data Discovery Graph ─────────────────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "graph":
    render_data_discovery_graph_section(
        west_lafayette_bbox=WEST_LAFAYETTE_BBOX,
        lafayette_default_bbox=LAFAYETTE_DEFAULT_BBOX,
    )

# ───────────────────── Explore Data Lake placeholder ─────────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "explore":
    render_explore_section()

# ───────────────────── User Query ────────────────────────────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "user_query":
    render_user_query_section(
        west_lafayette_bbox=WEST_LAFAYETTE_BBOX,
        lafayette_default_bbox=LAFAYETTE_DEFAULT_BBOX,
    )
