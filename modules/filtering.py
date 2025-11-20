"""
Data Filtering Page
"""

import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    """Render the filtering page"""
    st.header("🔍 DATA FILTERING MODULE")
    
    # Single column filter
    st.subheader("Filter by Column Values")
    
    all_cols = df.columns.tolist()
    col_to_filter = st.selectbox("Select a column to filter", all_cols)
    filter_condition = st.selectbox(
        "Select filter condition",
        ["Equals", "Greater than", "Less than", "Contains"]
    )
    
    # Input for filter value
    if df[col_to_filter].dtype == 'object':
        filter_value = st.text_input(f"Enter value to filter {col_to_filter} by")
    else:
        filter_value = st.number_input(
            f"Enter value to filter {col_to_filter} by",
            value=0.0
        )
    
    if st.button("🚀 Apply Filter"):
        try:
            if filter_condition == "Equals":
                filtered_df = df[df[col_to_filter] == filter_value]
            elif filter_condition == "Greater than":
                filtered_df = df[df[col_to_filter] > filter_value]
            elif filter_condition == "Less than":
                filtered_df = df[df[col_to_filter] < filter_value]
            elif filter_condition == "Contains":
                filtered_df = df[df[col_to_filter].astype(str).str.contains(
                    str(filter_value), na=False
                )]
            else:
                filtered_df = df
            
            if not filtered_df.empty:
                st.success(f"✅ Found {len(filtered_df)} matching rows")
                st.dataframe(filtered_df)
            else:
                st.warning("⚠️ No data found with the specified filter.")
        except Exception as e:
            st.error(f"❌ Error applying filter: {str(e)}")
    
    # Multiple conditions filter
    st.subheader("Filter by Multiple Conditions")
    st.info("💡 This feature allows you to filter by multiple columns simultaneously")
    
    if st.button("🚀 Setup Multiple Filters"):
        st.session_state.show_multi_filter = True
    
    if st.session_state.get('show_multi_filter', False):
        filter_conditions = []
        
        for col in all_cols:
            with st.expander(f"Filter by {col}"):
                if df[col].dtype == 'object':
                    value = st.text_input(f"Filter {col} by (text)", key=f"multi_{col}")
                    if value:
                        filter_conditions.append(df[col].str.contains(value, na=False))
                else:
                    use_filter = st.checkbox(f"Apply filter to {col}", key=f"check_{col}")
                    if use_filter:
                        value = st.number_input(
                            f"Filter {col} by (number)",
                            key=f"multi_{col}",
                            value=0.0
                        )
                        filter_conditions.append(df[col] == value)
        
        if st.button("🚀 Apply Multiple Filters") and filter_conditions:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter &= condition
            
            filtered_df = df[combined_filter]
            
            if not filtered_df.empty:
                st.success(f"✅ Found {len(filtered_df)} matching rows")
                st.dataframe(filtered_df)
            else:
                st.warning("⚠️ No data found with the specified filters.")