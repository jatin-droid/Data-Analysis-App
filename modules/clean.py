"""
Data Cleaning Page
"""

import streamlit as st
import pandas as pd
from utils.data_processor import convert_to_numeric, get_missing_values

def render(df: pd.DataFrame):
    """Render the clean page"""
    st.header("🧹 DATA CLEANING MODULE")
    
    # Initialize cleaned dataframe in session state
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = df.copy()
    
    df_clean = st.session_state.df_cleaned
    
    # Convert to numeric
    st.subheader("🔄 Convert to Numeric (if applicable)")
    if st.button("🚀 Convert All Columns"):
        df_clean = convert_to_numeric(df_clean)
        st.session_state.df_cleaned = df_clean
        st.success("✅ Conversion complete!")
    
    # Show missing values
    st.subheader("🔍 Missing Values Overview")
    missing_vals = get_missing_values(df_clean)
    
    if missing_vals.empty:
        st.success("✅ No missing values found.")
    else:
        st.dataframe(missing_vals)
        st.warning(f"⚠️ Total missing values: {missing_vals['Missing Count'].sum()}")
    
    # Cleaning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑 Remove Duplicates",
        "🔢 Fill with Mean",
        "🔣 Fill with Median",
        "🔁 Fill with Mode",
        "🧯 Remove Columns"
    ])
    
    with tab1:
        st.subheader("Remove Duplicate Rows")
        duplicates = df_clean.duplicated().sum()
        st.info(f"📊 Found {duplicates} duplicate rows")
        
        if st.button("🚀 Remove Duplicates", key="remove_dup"):
            initial_shape = df_clean.shape
            df_clean.drop_duplicates(inplace=True)
            st.session_state.df_cleaned = df_clean
            st.success(f"✅ Removed duplicates. New shape: {df_clean.shape} (was {initial_shape})")
    
    with tab2:
        st.subheader("Fill Missing Values with Mean")
        numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns available")
        else:
            col = st.selectbox("Select numeric column", numeric_cols, key="mean_col")
            
            if st.button("🚀 Fill with Mean", key="fill_mean"):
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
                st.session_state.df_cleaned = df_clean
                st.success(f"✅ Filled `{col}` with mean: {mean_val:.2f}")
    
    with tab3:
        st.subheader("Fill Missing Values with Median")
        numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns available")
        else:
            col = st.selectbox("Select numeric column", numeric_cols, key="median_col")
            
            if st.button("🚀 Fill with Median", key="fill_median"):
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                st.session_state.df_cleaned = df_clean
                st.success(f"✅ Filled `{col}` with median: {median_val:.2f}")
    
    with tab4:
        st.subheader("Fill Missing Values with Mode")
        all_cols = df_clean.columns.tolist()
        col = st.selectbox("Select a column", all_cols, key="mode_col")
        
        if st.button("🚀 Fill with Mode", key="fill_mode"):
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
                st.session_state.df_cleaned = df_clean
                st.success(f"✅ Filled `{col}` with mode: {mode_val[0]}")
            else:
                st.warning("⚠️ No mode available for this column.")
    
    with tab5:
        st.subheader("Remove Unnecessary Columns")
        all_cols = df_clean.columns.tolist()
        cols_to_remove = st.multiselect("Select columns to remove", all_cols, key="remove_cols")
        
        if st.button("🚀 Remove Selected Columns", key="remove_btn"):
            if cols_to_remove:
                df_clean.drop(columns=cols_to_remove, inplace=True)
                st.session_state.df_cleaned = df_clean
                st.success(f"🗑 Removed columns: {', '.join(cols_to_remove)}")
            else:
                st.warning("⚠️ No columns selected for removal.")
    
    # Display cleaned data
    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df_clean.head())
    
    # Download cleaned data
    csv = df_clean.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
    
    # Reset button
    if st.button("🔄 Reset to Original Data"):
        st.session_state.df_cleaned = df.copy()
        st.success("✅ Data reset to original!")
        st.rerun()