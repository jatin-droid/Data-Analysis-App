"""
Futuristic Theme CSS Styling
"""

import streamlit as st
import base64
from pathlib import Path


def get_img_as_base64(path: str) -> str:
    img_path = Path(path)
    with img_path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


def apply_theme(no_data_bg_path: str = "assets/s3.png"):
    """Apply futuristic cyberpunk theme to the Streamlit app"""
    img_base64 = get_img_as_base64(no_data_bg_path)

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
        
        /* ---------- GLOBAL APP BACKGROUND ---------- */
        /* ---------- SELECTBOX DROPDOWN (OPEN MENU) ---------- */
        ul[role="listbox"] {{
            background: rgba(12, 15, 35, 0.97);
            border-radius: 16px;
            border: 1px solid rgba(255, 182, 193, 0.7);
            box-shadow: 0 0 24px rgba(255, 182, 193, 0.6);
        padding: 0.4rem 0;
        }}

ul[role="listbox"] li {{
    color: #e6e6fa;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    padding: 0.45rem 1rem;
}}

/* Hover state */
ul[role="listbox"] li:hover {{
    background: linear-gradient(
        135deg,
        rgba(138, 43, 226, 0.8),
        rgba(255, 182, 193, 0.7)
    );
    color: #050816;
}}
/* Selected option */
ul[role="listbox"] li[aria-selected="true"] {{
    background: linear-gradient(
        135deg,
        #8a2be2,
        #e6e6fa
    );
    color: #050816;
    font-weight: 600;
}}

        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Rajdhani', sans-serif;
        }}

        .stApp::before {{
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
        }}

        /* ---------- MAIN CONTENT LAYOUT ---------- */
        .block-container {{
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}

        .content-shell {{
            background: rgba(5, 8, 22, 0.82);
            border-radius: 24px;
            padding: 1.75rem 2.25rem 2.25rem;
            backdrop-filter: blur(14px);
            box-shadow:
                0 0 35px rgba(0, 0, 0, 0.7),
                0 0 24px rgba(138, 43, 226, 0.45);
        }}

        .section-card {{
            background: rgba(12, 15, 35, 0.95);
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            border: 1px solid rgba(255, 182, 193, 0.35);
            margin-bottom: 1.2rem;
        }}

        .section-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #e6e6fa;
            opacity: 0.9;
            margin-bottom: 0.4rem;
        }}

        .metric-row {{
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
        }}

        .metric-card {{
            flex: 1;
            background: radial-gradient(circle at top,
                                        rgba(138, 43, 226, 0.4),
                                        rgba(5, 8, 22, 0.98));
            border-radius: 16px;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(255, 182, 193, 0.35);
            box-shadow: 0 0 14px rgba(255, 182, 193, 0.35);
        }}

        @media (max-width: 900px) {{
            .content-shell {{
                padding: 1.25rem 1.1rem 1.6rem;
            }}
            .metric-row {{
                flex-direction: column;
            }}
        }}

        /* ---------- TYPOGRAPHY ---------- */
        h1, h2, h3 {{
            font-family: 'Orbitron', sans-serif !important;
            color: #e6e6fa !important;
            text-shadow: 0 0 8px #e6e6fa;
            letter-spacing: 2px;
        }}

        /* ---------- SIDEBAR ---------- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #050816 0%, #1a1d3f 100%);
            border-right: 1px solid rgba(255, 182, 193, 0.7);
            box-shadow: 4px 0 20px rgba(255, 182, 193, 0.25);
        }}
        
        [data-testid="stSidebar"] * {{
            color: #e6e6fa !important;
            font-family: 'Rajdhani', sans-serif;
        }}

        /* Sidebar radio menu highlight */
        [data-testid="stSidebar"] [role="radiogroup"] {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        [data-testid="stSidebar"] [role="radio"] {{
            display: flex;
            align-items: center;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            transition:
                background 0.15s ease,
                transform 0.15s ease,
                box-shadow 0.15s ease;
        }}

        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {{
            background: linear-gradient(135deg,
                        rgba(138, 43, 226, 0.9),
                        rgba(255, 182, 193, 0.9));
            color: #050816 !important;
            box-shadow: 0 0 16px rgba(255, 182, 193, 0.9);
            transform: translateX(4px);
        }}

        /* ---------- FORM ELEMENTS & BUTTONS ---------- */
        .stButton > button {{
            background: linear-gradient(135deg, #8a2be2 0%, #e6e6fa 100%);
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
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px) scale(1.02);
            box-shadow: 0 0 25px rgba(255, 182, 193, 0.9);
        }}
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div,
        .stMultiSelect > div > div > div {{
            background: rgba(26, 29, 63, 0.85) !important;
            border: 1px solid rgba(255, 182, 193, 0.7) !important;
            border-radius: 10px !important;
            color: #e6e6fa !important;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.9rem;
        }}

        /* ---------- FILE UPLOADER ---------- */
        [data-testid="stFileUploaderDropzone"] {{
            background: rgba(26, 29, 63, 0.9);
            border-radius: 18px;
            border: 1px dashed rgba(255, 182, 193, 0.7);
            padding: 1.4rem;
            box-shadow: 0 0 18px rgba(0, 0, 0, 0.5);
            transition: border 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
        }}

        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: rgba(255, 255, 255, 0.95);
            box-shadow: 0 0 24px rgba(255, 182, 193, 0.9);
            transform: translateY(-1px);
        }}

        [data-testid="stFileUploaderDropzone"] span {{
            color: #e6e6fa !important;
            font-size: 0.9rem;
        }}

        /* ---------- DATAFRAME & TABS ---------- */
        .stDataFrame {{
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid rgba(255, 182, 193, 0.5);
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(255, 182, 193, 0.25);
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: rgba(26, 29, 63, 0.7);
            border-radius: 14px;
            padding: 6px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: linear-gradient(135deg,
                        rgba(138, 43, 226, 0.3),
                        rgba(255, 182, 193, 0.25));
            border-radius: 10px;
            color: #e6e6fa !important;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            border: 1px solid rgba(255, 182, 193, 0.6);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #8a2be2, #e6e6fa) !important;
            color: #000 !important;
            box-shadow: 0 0 16px rgba(255, 182, 193, 0.8);
        }}

        /* ---------- METRICS ---------- */
        [data-testid="stMetricValue"] {{
            color: #e6e6fa !important;
            font-family: 'Orbitron', sans-serif;
            text-shadow: 0 0 8px #e6e6fa;
        }}

        [data-testid="stMetricLabel"] {{
            color: #b8c5d6 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.75rem;
        }}

        /* ---------- SCROLLBARS ---------- */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #050816;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, #8a2be2, #e6e6fa);
            border-radius: 8px;
        }}

        /* ---------- MISC INFO BOX (OPTIONAL) ---------- */
        .info-box {{
            text-align: center;
            padding: 40px 10px;
            color: #b8c5d6;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
