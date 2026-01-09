import streamlit as st


def render_preview_section():
    """Render the preview tables section."""
    st.markdown('<div id="preview-data-lake"></div>', unsafe_allow_html=True)
    st.subheader("Data Lake Tables 📚")
    datalake_table_names = list(st.session_state.data_lake.keys())
    selected_datalake_table = st.selectbox(
        "Select a table to view it here:", datalake_table_names
    )

    if selected_datalake_table:
        st.write(f"**{selected_datalake_table}**")
        st.dataframe(st.session_state.data_lake[selected_datalake_table])
