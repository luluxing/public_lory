import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from streamlit_scroll_to_top import scroll_to_here

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

if "selected_section" not in st.session_state:
    st.session_state.selected_section = None

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False
if "lake_loaded" not in st.session_state:
    st.session_state.lake_loaded = False
if "lake_loaded_name" not in st.session_state:
    st.session_state.lake_loaded_name = None

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


# ───────────────────── Sidebar navigation ───────────────────────────────────
if st.session_state.data_lake:
    st.sidebar.markdown("### Sections")
    if st.session_state.lake_loaded:
        st.sidebar.success(
            f"Loaded “{st.session_state.lake_loaded_name}” "
            f"({len(st.session_state.data_lake)} tables)"
        )
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
