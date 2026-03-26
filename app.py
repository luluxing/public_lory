import os
import re
from pathlib import Path
import streamlit as st
from streamlit_scroll_to_top import scroll_to_here
import pandas as pd

import pandas_helper
from data_path import data_path
from sections.geo_ui import DEFAULT_BBOX_SELECTION
from sections.data_discovery_graph import render_data_discovery_graph_section
from sections.explore_data_lake import render_explore_section
from sections.lake_overview import render_loaded_lake_overview
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

if "uploaded_csv_files" not in st.session_state:
    st.session_state.uploaded_csv_files = {}

if "preview_display_tables" not in st.session_state:
    st.session_state.preview_display_tables = {}
if "preview_generated_geo_columns" not in st.session_state:
    st.session_state.preview_generated_geo_columns = {}

if "graph_nodes" not in st.session_state:
    st.session_state.graph_nodes = []

if "ddg_show_geo_tables" not in st.session_state:
    st.session_state.ddg_show_geo_tables = False
if "ddg_show_nogeo_tables" not in st.session_state:
    st.session_state.ddg_show_nogeo_tables = False

if "discovery_geo_augmentation" not in st.session_state:
    st.session_state.discovery_geo_augmentation = False
if "ddg_manual_geo_augmentation_map" not in st.session_state:
    st.session_state.ddg_manual_geo_augmentation_map = {}
if "ddg_manual_geo_augmentation_details" not in st.session_state:
    st.session_state.ddg_manual_geo_augmentation_details = {}
if "ddg_spatial_edges" not in st.session_state:
    st.session_state.ddg_spatial_edges = []
if "ddg_spatial_refresh_tables" not in st.session_state:
    st.session_state.ddg_spatial_refresh_tables = []
if "ddg_geo_aug_request" not in st.session_state:
    st.session_state.ddg_geo_aug_request = ""
if "ddg_geo_aug_trigger" not in st.session_state:
    st.session_state.ddg_geo_aug_trigger = False
if "ddg_geo_aug_message" not in st.session_state:
    st.session_state.ddg_geo_aug_message = None

if "ddg_show_rel_joins" not in st.session_state:
    st.session_state.ddg_show_rel_joins = False
if "ddg_show_semantic_joins" not in st.session_state:
    st.session_state.ddg_show_semantic_joins = False
if "ddg_show_spatial_joins" not in st.session_state:
    st.session_state.ddg_show_spatial_joins = False

if "graph_rel_edges" not in st.session_state:
    st.session_state.graph_rel_edges = []

if "graph_spatial_edges" not in st.session_state:
    st.session_state.graph_spatial_edges = []

if "graph_semantic_edges" not in st.session_state:
    st.session_state.graph_semantic_edges = []

if "target_schema" not in st.session_state:
    st.session_state.target_schema = []
if "target_schema_raw" not in st.session_state:
    st.session_state.target_schema_raw = []
if "user_query_target_attr_input" not in st.session_state:
    st.session_state.user_query_target_attr_input = ""
if "user_uploaded_table_df" not in st.session_state:
    st.session_state.user_uploaded_table_df = None
if "user_uploaded_table_name" not in st.session_state:
    st.session_state.user_uploaded_table_name = None
if "user_query_scenario1_explicit" not in st.session_state:
    st.session_state.user_query_scenario1_explicit = False

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
if "preview_selected_column" not in st.session_state:
    st.session_state.preview_selected_column = None
if "preview_previous_table" not in st.session_state:
    st.session_state.preview_previous_table = None
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
    st.session_state.preview_display_tables = {}
    st.session_state.preview_generated_geo_columns = {}
    # Note: uploaded_csv_files are preserved when switching lakes
    st.session_state.discovery_geo_augmentation = False
    st.session_state.ddg_manual_geo_augmentation_map = {}
    st.session_state.ddg_manual_geo_augmentation_details = {}
    st.session_state.ddg_spatial_edges = []
    st.session_state.ddg_spatial_refresh_tables = []
    st.session_state.ddg_geo_aug_request = ""
    st.session_state.ddg_geo_aug_trigger = False
    st.session_state.ddg_geo_aug_message = None
    st.session_state.selected_section = None
    st.session_state.in_function_view = False
    reset_data_discovery_graph_view()


def reset_data_discovery_graph_view():
    """Reset discovery-graph visibility so the canvas starts empty."""
    st.session_state.ddg_show_geo_tables = False
    st.session_state.ddg_show_nogeo_tables = False
    st.session_state.ddg_show_rel_joins = False
    st.session_state.ddg_show_semantic_joins = False
    st.session_state.ddg_show_spatial_joins = False
    st.session_state.graph_nodes = []
    st.session_state.graph_rel_edges = []
    st.session_state.graph_spatial_edges = []
    st.session_state.graph_semantic_edges = []
    st.session_state.ddg_spatial_edges = []
    st.session_state.ddg_spatial_refresh_tables = []
    st.session_state.ddg_geo_aug_request = ""
    st.session_state.ddg_geo_aug_trigger = False
    st.session_state.ddg_geo_aug_message = None
    st.session_state.show_join_details = False
    st.session_state["all_join_paths"] = []
    st.session_state["interesting_paths"] = []
    st.session_state["generated_path_count"] = 0
    st.session_state["selected_path_for_map"] = None
    st.session_state["paths_page"] = 0
    st.session_state["path_map_selections"] = {}


def reset_state_for_new_input_table():
    """Reset state when the input table selection changes within a lake."""
    st.session_state.uploaded_table = None


APP_RESET_NOTICE_KEY = "app_reset_notice"
SCENARIO1_ATTR_DRAFT_KEY = "user_query_scenario1_attr_draft"

# Default Scenario 1 columns of interest (demo / vision lake workflow)
DEFAULT_SCENARIO1_TARGET_ATTRIBUTES = [
    "accident_id",
    "date",
    "victim_type",
    "report_id",
    "status",
    "closest hospital.name",
]


def apply_default_scenario1_target_attributes():
    """Reset sidebar table + pipeline target lists to the standard column set."""
    st.session_state.target_schema = list(DEFAULT_SCENARIO1_TARGET_ATTRIBUTES)
    st.session_state.target_schema_raw = list(DEFAULT_SCENARIO1_TARGET_ATTRIBUTES)
    st.session_state.pop(SCENARIO1_ATTR_DRAFT_KEY, None)


def _scenario1_target_attr_cell_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def render_scenario1_target_attributes_sidebar():
    """Editable single-column table for columns of interest; syncs target_schema."""
    col_name = "Column of interest"
    if SCENARIO1_ATTR_DRAFT_KEY not in st.session_state:
        ts = st.session_state.get("target_schema") or []
        st.session_state[SCENARIO1_ATTR_DRAFT_KEY] = (
            [_scenario1_target_attr_cell_str(x) for x in ts] if ts else [""]
        )

    draft = st.session_state[SCENARIO1_ATTR_DRAFT_KEY]
    if not isinstance(draft, list):
        draft = [""]
    rows = draft if draft else [""]
    df = pd.DataFrame({col_name: rows})

    edited = st.sidebar.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="user_query_scenario1_attr_editor",
        column_config={
            col_name: st.column_config.TextColumn(
                col_name,
                help="Name or describe each column you want included in the integrated result.",
            )
        },
    )

    new_draft = [
        _scenario1_target_attr_cell_str(raw) for raw in edited[col_name].tolist()
    ]
    if not new_draft:
        new_draft = [""]
    st.session_state[SCENARIO1_ATTR_DRAFT_KEY] = new_draft
    non_empty = [s for s in new_draft if s]
    st.session_state.target_schema = non_empty
    st.session_state.target_schema_raw = list(non_empty)


def clear_data_lake_and_restart():
    """Remove persisted temp tables and return the app to its initial state."""
    removed_paths = pandas_helper.clear_temp_table_csvs()
    st.session_state.clear()
    # st.session_state[APP_RESET_NOTICE_KEY] = (
    #     f"Started fresh. Removed {len(removed_paths)} temp table(s)."
    # )


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
    and (Path(DATALAKES_ROOT) / d / "all_tables").is_dir()
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
    key="datalake_selector",
)

st.sidebar.button(
    "Clear data lake and start over",
    key="clear_data_lake_and_start_over",
    width="stretch",
    on_click=clear_data_lake_and_restart,
)

reset_notice = st.session_state.pop(APP_RESET_NOTICE_KEY, None)
if reset_notice:
    st.sidebar.success(reset_notice)

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
                # Keep CSVs under temp_tables/;
                # load_datalake_tables overlays them on base all_tables.
                st.session_state.data_lake = pandas_helper.load_datalake_tables(
                    lake_dir=lake_dir
                )
                apply_default_scenario1_target_attributes()

            if input_dir.exists():
                st.session_state.input_table_files = list(input_dir.glob("*.csv"))
            else:
                st.session_state.input_table_files = []

            st.session_state.lake_loaded = True
            st.session_state.lake_loaded_name = chosen_lake

            # Merge uploaded CSV files with lake data
            if st.session_state.uploaded_csv_files:
                for table_name, file_info in st.session_state.uploaded_csv_files.items():
                    st.session_state.data_lake[table_name] = file_info['dataframe']
        else:
            st.warning("No data-lake directories found under ./datalakes/.")

# Upload CSV files
st.sidebar.markdown("---")
st.sidebar.markdown("**Or upload your own CSV files**")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=['csv'],
    accept_multiple_files=True,
    key="csv_file_uploader"
)

# Process uploaded CSV files
if uploaded_files:
    # Track if any new files were added
    new_files_added = False
    for uploaded_file in uploaded_files:
        # Get table name from filename (remove .csv extension)
        table_name = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
        
        # Check if this is a new file or changed file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if table_name not in st.session_state.uploaded_csv_files or \
           st.session_state.uploaded_csv_files[table_name].get('file_id') != file_id:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, dtype=str, on_bad_lines='skip')
                
                # Store the dataframe and file info
                st.session_state.uploaded_csv_files[table_name] = {
                    'dataframe': df,
                    'file_id': file_id,
                    'filename': uploaded_file.name
                }
                
                # Add to data_lake (will overwrite if table_name exists)
                st.session_state.data_lake[table_name] = df
                new_files_added = True
            except Exception as e:
                st.sidebar.error(f"Error reading {uploaded_file.name}: {str(e)}")
    
    if new_files_added:
        uploaded_count = len(st.session_state.uploaded_csv_files)
        st.session_state.selected_section = None
        st.session_state.in_function_view = False
        st.session_state.scroll_to_top = True
        st.sidebar.success(f"✅ Loaded {uploaded_count} uploaded table(s)")
        # Trigger rerun to update UI
        st.rerun()

# Clear uploaded files button
if st.session_state.uploaded_csv_files:
    if st.sidebar.button("Clear uploaded files", key="clear_uploaded_files"):
        # Remove uploaded tables from data_lake
        for table_name in st.session_state.uploaded_csv_files.keys():
            if table_name in st.session_state.data_lake:
                del st.session_state.data_lake[table_name]
        st.session_state.uploaded_csv_files = {}
        st.rerun()

# Ensure uploaded CSV files are always in data_lake (in case lake was loaded before files were uploaded)
if st.session_state.uploaded_csv_files:
    for table_name, file_info in st.session_state.uploaded_csv_files.items():
        st.session_state.data_lake[table_name] = file_info['dataframe']

def _set_section(name: str):
    if name == "graph":
        reset_data_discovery_graph_view()
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
            "Preview Data Lake Tables",
            width="stretch",
            key="nav_preview",
            on_click=_set_section,
            args=("preview",),
        )
        st.sidebar.button(
            "Data Discovery Graph",
            width="stretch",
            key="nav_graph",
            on_click=_set_section,
            args=("graph",),
        )
        st.sidebar.button(
            "Explore by Region",
            width="stretch",
            key="nav_explore",
            on_click=_set_section,
            args=("explore",),
        )
        st.sidebar.button(
            "Location-aware Discovery & Augmentation",
            width="stretch",
            key="nav_user_query",
            on_click=_set_section,
            args=("user_query",),
        )
    else:
        # Function view: Show function-specific parameters
        section_names = {
            "preview": "Preview Data Lake",
            "graph": "Data Discovery Graph",
            "explore": "Explore by Region",
            "user_query": "Location-aware Discovery & Augmentation"
        }
        current_section_name = section_names.get(st.session_state.selected_section, "Function")

        st.sidebar.markdown(f"### {current_section_name}")
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
                st.session_state.preview_selected_column = None
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
                key="path_spatial_join_type_input"
            )
        elif st.session_state.selected_section == "user_query":
            st.sidebar.markdown("**Scenario Selection**")

            scenario_discovery_key = "scenario_1_discovery_integration"
            scenario_augmentation_key = "scenario_2_augmentation"
            scenario_labels = {
                scenario_discovery_key: "Scenario 1: Location-aware Data Discovery and Integration",
                scenario_augmentation_key: "Scenario 2: Location-aware Data Augmentation",
            }

            if "user_query_scenario_key" not in st.session_state:
                legacy_scenario = st.session_state.get("user_query_scenario", "")
                if isinstance(legacy_scenario, str) and "Augmentation" in legacy_scenario:
                    st.session_state.user_query_scenario_key = scenario_augmentation_key
                else:
                    st.session_state.user_query_scenario_key = scenario_discovery_key

            if st.sidebar.button(
                scenario_labels[scenario_discovery_key],
                key="user_query_scenario_discovery_button",
                width="stretch",
            ):
                st.session_state.user_query_scenario_key = scenario_discovery_key
                st.session_state.user_query_scenario1_explicit = True

            if st.sidebar.button(
                scenario_labels[scenario_augmentation_key],
                key="user_query_scenario_augmentation_button",
                width="stretch",
            ):
                st.session_state.user_query_scenario_key = scenario_augmentation_key
                st.session_state.user_query_scenario1_explicit = False
                st.session_state.pop(SCENARIO1_ATTR_DRAFT_KEY, None)

            selected_scenario_key = st.session_state.get(
                "user_query_scenario_key",
                scenario_discovery_key,
            )
            if selected_scenario_key not in scenario_labels:
                selected_scenario_key = scenario_discovery_key
                st.session_state.user_query_scenario_key = scenario_discovery_key

            selected_scenario_label = scenario_labels[selected_scenario_key]
            st.session_state.user_query_scenario = selected_scenario_label
            if selected_scenario_key == scenario_augmentation_key:
                st.sidebar.caption(f"Selected scenario: {selected_scenario_label}")
            elif st.session_state.user_query_scenario1_explicit:
                st.sidebar.caption(f"Selected scenario: {selected_scenario_label}")
            else:
                st.sidebar.caption("Choose Scenario 1 or Scenario 2 above.")

            if selected_scenario_key == scenario_augmentation_key:
                st.sidebar.markdown("**Upload CSV Table**")
                uploaded_csv = st.sidebar.file_uploader(
                    "Choose a CSV file",
                    type=['csv'],
                    key="user_query_csv_uploader",
                    help="Upload a CSV table to append attributes to"
                )

                if uploaded_csv is not None:
                    try:
                        uploaded_csv.seek(0)
                        df = pd.read_csv(uploaded_csv, dtype=str, on_bad_lines='skip')
                        st.session_state.user_uploaded_table_df = df
                        st.session_state.user_uploaded_table_name = uploaded_csv.name.replace('.csv', '').replace('.CSV', '')
                        if len(df.columns) > 0:
                            st.sidebar.success(f"Loaded table with {len(df)} rows and {len(df.columns)} columns")
                    except Exception as e:
                        st.sidebar.error(f"Error reading CSV: {str(e)}")
                elif st.session_state.user_uploaded_table_df is not None:
                    st.sidebar.info(f"Table loaded: {len(st.session_state.user_uploaded_table_df)} rows, {len(st.session_state.user_uploaded_table_df.columns)} columns")

                if st.sidebar.button("Clear Uploaded Table", key="clear_uploaded_table", width="stretch"):
                    st.session_state.user_uploaded_table_df = None
                    st.session_state.user_uploaded_table_name = None
                    st.session_state.target_schema = []
                    st.session_state.target_schema_raw = []
                    st.session_state.user_query_target_attr_input = ""
                    st.session_state.pop(SCENARIO1_ATTR_DRAFT_KEY, None)
                    st.rerun()

                st.sidebar.divider()
                st.sidebar.markdown("**Attributes to Append**")
                st.session_state.user_query_target_attr_input = st.sidebar.text_area(
                    "Enter attributes (comma-separated or one per line)",
                    value=st.session_state.user_query_target_attr_input,
                    key="user_query_target_attr_textarea",
                    height=120,
                )

                if st.sidebar.button("Add Attributes", key="user_query_scenario2_add_attr", width="stretch"):
                    raw_text = st.session_state.user_query_target_attr_input
                    attrs = [a.strip() for a in re.split(r"[,\n]", raw_text) if a.strip()]
                    st.session_state.target_schema = attrs
                    st.session_state.target_schema_raw = attrs
                    st.rerun()
            else:
                if st.session_state.user_query_scenario1_explicit:
                    st.sidebar.divider()
                    st.sidebar.markdown("**Target Attributes**")
                    st.sidebar.caption(
                        "Use the table to add, remove, or edit columns of interest (row **+** / **−**)."
                    )
                    render_scenario1_target_attributes_sidebar()
                    if "spatial_choice_mode" not in st.session_state:
                        st.session_state["spatial_choice_mode"] = "inferred"
                    if "spatial_manual_mode" not in st.session_state:
                        st.session_state["spatial_manual_mode"] = "exclude"
                    if "spatial_manual_distance_km" not in st.session_state:
                        st.session_state["spatial_manual_distance_km"] = None
                else:
                    st.sidebar.info(
                        "Click **Scenario 1** above to enter target attributes."
                    )
        st.sidebar.divider()
        st.sidebar.button(
            "Back to Other Functions",
            width="stretch",
            key="nav_back",
            on_click=_back_to_functions,
        )

    st.sidebar.divider()

# Smooth-scroll to top when switching sections using streamlit-scroll-to-top
if st.session_state.get("scroll_to_top"):
    scroll_to_here(0, key="scroll_top_helper")
    st.session_state.scroll_to_top = False

if st.session_state.data_lake and st.session_state.selected_section is None:
    render_loaded_lake_overview(
        chosen_lake=st.session_state.get("chosen_lake"),
        data_lake=st.session_state.data_lake,
        uploaded_table_names=st.session_state.uploaded_csv_files.keys(),
    )

# ───────────────────── Main page: Show data-lake tables ──────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "preview":
    render_preview_section()

# ───────────────────── Data Discovery Graph ─────────────────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "graph":
    render_data_discovery_graph_section(
        west_lafayette_bbox=WEST_LAFAYETTE_BBOX,
        lafayette_default_bbox=LAFAYETTE_DEFAULT_BBOX,
    )

# ───────────────────── Explore by Region ─────────────────────────
if st.session_state.data_lake and st.session_state.selected_section == "explore":
    render_explore_section()

# ───────────────────── Location-aware Discovery / Augmentation ─────────────
if st.session_state.data_lake and st.session_state.selected_section == "user_query":
    render_user_query_section(
        west_lafayette_bbox=DEFAULT_BBOX_SELECTION,
        lafayette_default_bbox=DEFAULT_BBOX_SELECTION,
    )
