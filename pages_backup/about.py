"""
About Page
"""

import streamlit as st

def render():
    """Render the about page"""
    st.header("ℹ️ ABOUT THIS MODULE")
    
    st.markdown("""
    ### 🌌 **FUTURISTIC DATA ANALYSIS HUB**
    
    Welcome to the next generation of data analysis tools, built with cutting-edge technology and designed for the future!
    
    ---
    
    #### 🎯 **Key Features:**
    
    **🗂 Data Exploration**
    - Upload and explore CSV datasets with lightning speed
    - View data statistics, structure, and insights instantly
    - Identify missing values and data types
    
    **🔍 Advanced Filtering**
    - Apply complex filters to your data
    - Multi-condition filtering for precise data selection
    - Real-time filtering with instant results
    
    **📊 Dynamic Visualizations**
    - Create stunning charts and graphs
    - Interactive visual analytics powered by Matplotlib & Seaborn
    - Multiple chart types: bar, histogram, line, scatter, heatmap, box plot, and pie charts
    
    **🧹 Intelligent Data Cleaning**
    - Handle missing values with multiple strategies
    - Remove duplicates and outliers effortlessly
    - Fill missing data with mean, median, or mode
    - Remove unnecessary columns
    
    **📈 Grouping & Aggregation**
    - Group data by single or multiple columns
    - Perform statistical aggregations on the fly
    - Custom aggregation functions for advanced analysis
    
    **🔠 Smart Encoding**
    - One-hot encoding for categorical variables
    - Label encoding with automatic handling
    - Visual mapping of encoded values
    
    **🤖 Machine Learning Integration**
    - Build ML models directly in the app
    - Support for Logistic Regression, Decision Trees, and Random Forests
    - Automatic feature encoding and data preprocessing
    - Model performance metrics and visualizations
    
    ---
    
    #### 👨‍💻 **Developer Information:**
    
    **Name:** Jatin Kumar  
    **Education:** B.Tech (Electronics and Communication Engineering), 3rd Year  
    **Institution:** Guru Nanak Dev University, Amritsar  
    
    ---
    
    #### 🎨 **Technology Stack:**
    
    - **Frontend:** Streamlit with Custom CSS
    - **Data Processing:** Pandas, NumPy
    - **Visualizations:** Matplotlib, Seaborn
    - **Machine Learning:** Scikit-learn
    - **Design:** Futuristic UI with neon gradients and animations
    
    ---
    
    #### 📫 **Contact:**
    
    - **Phone:** 9888197119  
    - **Email:** dhjatin4@gmail.com
    
    ---
    
    ### 💡 **Design Philosophy:**
    
    This app combines functionality with aesthetics, featuring:
    - 🌈 Animated gradient backgrounds
    - ⚡ Neon glow effects on interactive elements
    - 🎮 Cyberpunk-inspired color scheme (cyan & purple)
    - 🚀 Smooth transitions and hover effects
    - 🔮 Futuristic Orbitron & Rajdhani fonts
    - 🌌 Grid overlay for that high-tech feel
    
    ---
    
    ### 🚀 **How to Use:**
    
    1. **Upload Data**: Start by uploading a CSV file
    2. **Explore**: Get familiar with your data structure
    3. **Clean**: Handle missing values and duplicates
    4. **Visualize**: Create beautiful charts and graphs
    5. **Analyze**: Apply filters and grouping operations
    6. **Model**: Build machine learning models
    7. **Export**: Download processed data and results
    
    ---
    
    ### 📦 **Project Structure:**
    
    This app is built with a modular architecture for easy maintenance:
    
    ```
    data-analysis-app/
    ├── main.py                 # Main application
    ├── styles/theme.py        # CSS styling
    ├── modules/                 # Individual page modules
    │   ├── explore.py
    │   ├── filtering.py
    │   ├── select.py
    │   ├── clean.py
    │   ├── visualize.py
    │   ├── grouping.py
    │   ├── encoding.py
    │   ├── modeling.py
    │   └── about.py
    └── utils/                 # Utility functions
        ├── data_processor.py
        └── ml_helper.py
    ```
    
    ---
    
    ### 🙏 **Thank You!**
    
    Thank you for using the Futuristic Data Analysis Hub! We hope this tool empowers your data journey and makes analysis an exciting experience.
    
    *Built with ❤️ and ⚡ by Jatin Kumar*
    
    ---
    """)
    
    # Add some visual flair with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(0, 255, 255, 0.2)); border-radius: 15px; border: 2px solid rgba(0, 255, 255, 0.3);'>
            <h2 style='font-size: 3em;'>🚀</h2>
            <h4>Fast Processing</h4>
            <p>Lightning-fast data analysis and visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(0, 255, 255, 0.2)); border-radius: 15px; border: 2px solid rgba(0, 255, 255, 0.3);'>
            <h2 style='font-size: 3em;'>🎨</h2>
            <h4>Beautiful UI</h4>
            <p>Stunning futuristic design that's easy to use</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(0, 255, 255, 0.2)); border-radius: 15px; border: 2px solid rgba(0, 255, 255, 0.3);'>
            <h2 style='font-size: 3em;'>🔒</h2>
            <h4>Secure & Private</h4>
            <p>Your data never leaves your browser</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Version and updates
    st.subheader("📅 Version History")
    st.markdown("""
    **Version 2.0.0** (Current) - Modular Architecture
    - ✨ Refactored into modular structure
    - 🎨 Enhanced futuristic UI theme
    - 🚀 Improved performance
    - 📊 Better data visualization
    
    **Version 1.0.0** - Initial Release
    - 📤 CSV file upload
    - 🔍 Basic data exploration
    - 📈 Simple visualizations
    """)