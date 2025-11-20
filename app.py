"""
Main Application Entry Point
Futuristic Data Analysis Hub
"""

import streamlit as st
import pandas as pd
from styles.theme import apply_theme
from modules import explore, filtering, select, clean, visualize, grouping, encoding, modeling, about

# Configure page
st.set_page_config(
    page_title="Data Analysis Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed to collapsed initially
)

# Apply futuristic theme
apply_theme()

# Welcome section
Name = st.text_input("🚀 Enter Your Name")
st.title(f"⚡ Hi {Name}! Welcome to the 🌌 DATA ANALYSIS HUB")

# File uploader
st.subheader("📤 Upload Your Dataset")
uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")

if uploaded_file:
    # Read CSV and store in session state
    if 'df' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.uploaded_file_name = uploaded_file.name
    else:
        df = st.session_state.df
    
    st.success("✅ File uploaded successfully!")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Rows", df.shape[0])
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
    
    # Sidebar navigation - ONLY when file is uploaded
    st.sidebar.title('🎮 CONTROL PANEL')
    st.sidebar.write("Welcome to the Futuristic Data Analysis Hub! Navigate through different modules to explore, visualize, and analyze your data with cutting-edge tools.")
    
    # Page selection
    page = st.sidebar.radio(
        "🗂️ Select Module",
        ['Explore', 'Filtering', 'Select', 'Clean', 'Visualize', 'Grouping', 'Encoding', 'Modeling', 'About']
    )
    
    # Route to appropriate page
    if page == 'Explore':
        explore.render(df)
    elif page == 'Filtering':
        filtering.render(df)
    elif page == 'Select':
        select.render(df)
    elif page == 'Clean':
        clean.render(df)
    elif page == 'Visualize':
        visualize.render(df)
    elif page == 'Grouping':
        grouping.render(df)
    elif page == 'Encoding':
        encoding.render(df)
    elif page == 'Modeling':
        modeling.render(df)
    elif page == 'About':
        about.render()

else:
    # Welcome screen when no file is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='font-size: 3em;'>🌌</h2>
        <h3>Welcome to the  of Data Analysis</h3>
        <p style='font-size: 1.2em; margin-top: 20px;'>
            Upload your CSV file to begin your journey into advanced data exploration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("📁 Please upload a CSV file using the uploader above to access all features")
