"""
Column and Row Selection Page
"""

import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    """Render the select page"""
    st.header("📑 COLUMN & ROW SELECTOR")
    
    # Column selection
    st.subheader("📋 Column Selection")
    selected_cols = st.multiselect(
        "Choose columns to display",
        df.columns.tolist(),
        default=[]
    )
    
    if not selected_cols:
        st.info("🎯 Select at least one column to proceed")
        return
    
    # Row selection
    st.subheader("🔢 Row Selection")
    selection_mode = st.radio("Select rows by:", ["Index Range", "Specific Indices"])
    
    if selection_mode == "Index Range":
        col1, col2 = st.columns(2)
        
        with col1:
            start = st.number_input(
                "Start index",
                min_value=0,
                max_value=len(df)-1,
                value=0
            )
        
        with col2:
            end = st.number_input(
                "End index",
                min_value=0,
                max_value=len(df),
                value=min(5, len(df))
            )
        
        if start < end:
            result_df = df.iloc[int(start):int(end)][selected_cols]
            st.success(f"✅ Displaying rows {start} to {end-1}")
            st.dataframe(result_df)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Selection as CSV",
                data=csv,
                file_name="selected_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Start index must be less than end index.")
    
    elif selection_mode == "Specific Indices":
        selected_rows = st.multiselect(
            "Choose row indices",
            options=list(range(len(df))),
            default=[]
        )
        
        if selected_rows:
            result_df = df.iloc[selected_rows][selected_cols]
            st.success(f"✅ Displaying {len(selected_rows)} selected rows")
            st.dataframe(result_df)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Selection as CSV",
                data=csv,
                file_name="selected_data.csv",
                mime="text/csv"
            )
        else:
            st.info("🎯 No rows selected.")