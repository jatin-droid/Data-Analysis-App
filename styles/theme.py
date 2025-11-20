"""
Futuristic Theme CSS Styling
"""

import streamlit as st

def apply_theme():
    """Apply futuristic cyberpunk theme to the Streamlit app"""
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top left, #15192e 0%, #050716 40%, #050716 100%);
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            linear-gradient(rgba(255, 182, 193, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 182, 193, 0.05) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.4;
    }

    .block-container {
        position: relative;
        z-index: 1;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #FFB6C1 !important;
        text-shadow: 0 0 8px #FFB6C1;
        letter-spacing: 2px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050816 0%, #1a1d3f 100%);
        border-right: 1px solid rgba(255, 182, 193, 0.7);
        box-shadow: 4px 0 20px rgba(255, 182, 193, 0.25);
    }
    
    [data-testid="stSidebar"] * {
        color: #FFB6C1 !important;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8a2be2 0%, #FFB6C1 100%);
        color: #000 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(255, 182, 193, 0.6);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 0 25px rgba(255, 182, 193, 0.9);
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background: rgba(26, 29, 63, 0.85) !important;
        border: 1px solid rgba(255, 182, 193, 0.7) !important;
        border-radius: 10px !important;
        color: #FFB6C1 !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
    }
    
    .stDataFrame {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid rgba(255, 182, 193, 0.5);
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(255, 182, 193, 0.25);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 29, 63, 0.7);
        border-radius: 14px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.3), rgba(255, 182, 193, 0.25));
        border-radius: 10px;
        color: #FFB6C1 !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        border: 1px solid rgba(255, 182, 193, 0.6);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8a2be2, #FFB6C1) !important;
        color: #000 !important;
        box-shadow: 0 0 16px rgba(255, 182, 193, 0.8);
    }
    
    [data-testid="stMetricValue"] {
        color: #FFB6C1 !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 8px #FFB6C1;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #050816;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8a2be2, #FFB6C1);
        border-radius: 8px;
    }

    .info-box {
        text-align: center;
        padding: 40px 10px;
        color: #b8c5d6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)