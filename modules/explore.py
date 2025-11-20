"""
Data Exploration Page
"""

import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    """Render the explore page"""
    st.header("🔍 DATA EXPLORATION MODULE")
    
    option = st.selectbox(
        "Choose an operation",
        ["None", "Head", "Tail", "To Markdown", "Describe", "Dtypes", "Columns", "iLoc", "Missing Values"]
    )
    
    if option == "Head":
        st.subheader("📄 First 5 Rows")
        st.dataframe(df.head())
        
    elif option == "Tail":
        st.subheader("📄 Last 5 Rows")
        st.dataframe(df.tail())
        
    elif option == "To Markdown":
        st.subheader("📝 Markdown Format")
        st.markdown(df.to_markdown())
        
    elif option == "Describe":
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe())
        
    elif option == "Dtypes":
        st.subheader("🔤 Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtypes_df)
        
    elif option == "Columns":
        st.subheader("📋 Column Names")
        cols = df.columns.tolist()
        for i, col in enumerate(cols, 1):
            st.write(f"{i}. {col}")
            
    elif option == "iLoc":
        st.subheader("🔢 Rows 10-20")
        st.dataframe(df.iloc[10:20])
        
    elif option == "Missing Values":
        st.subheader("❓ Missing Values Count")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if missing_df.empty:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing_df)
            
    else:
        st.info("🎯 Select an option to explore your data")