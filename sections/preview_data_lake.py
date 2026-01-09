import streamlit as st


def render_preview_section():
    """Render the preview tables section."""
    st.markdown('<div id="preview-data-lake"></div>', unsafe_allow_html=True)
    st.subheader("Data Lake Table 📚")
    
    selected_table = st.session_state.get("preview_selected_table")
    num_rows = st.session_state.get("preview_num_rows", 10)
    
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
    elif st.session_state.data_lake:
        st.info("Please select a table from the sidebar.")
    else:
        st.info("No data lake loaded.")
