import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Automated Data Analysis App", layout="wide")


# Futuristic CSS Theme
st.markdown("""
    <style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3f 25%, #0f1b3d 50%, #1e1e3f 75%, #0a0e27 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Rajdhani', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glowing grid overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Headers with neon glow */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffff !important;
        text-shadow: 0 0 7px #00ffff, 0 0 10px #00ffff, 0 0 15px #00ffff;
        animation: pulseGlow 2s ease-in-out infinite;
        letter-spacing: 2px;
    }
    
    @keyframes pulseGlow {
        0%, 100% { text-shadow: 0 0 7px #00ffff, 0 0 15px #00ffff; }
        50% { text-shadow: 0 0 6px #00ffff, 0 0 12px #00ffff, 0 0 20px #00ffff; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #1a1d3f 100%);
        border-right: 2px solid #00ffff;
        box-shadow: 5px 0 20px rgba(0, 255, 255, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #00ffff !important;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Radio buttons with futuristic style */
    .stRadio > label {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.1), rgba(138, 43, 226, 0.1));
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 255, 0.3);
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    .stRadio > label:hover {
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        transform: translateX(5px);
    }
    
    /* Buttons with neon effect */
    .stButton > button {
        background: linear-gradient(135deg, #8a2be2 0%, #00ffff 100%);
        color: #000 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.6), 0 0 40px rgba(0, 255, 255, 0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(138, 43, 226, 0.8), 0 0 60px rgba(0, 255, 255, 0.6);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(26, 29, 63, 0.8) !important;
        border: 2px solid rgba(0, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        color: #00ffff !important;
        font-family: 'Rajdhani', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(13, 17, 23, 0.9);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(26, 29, 63, 0.5);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.3), rgba(0, 255, 255, 0.3));
        border-radius: 10px;
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        border: 1px solid rgba(0, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.6), rgba(0, 255, 255, 0.6));
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8a2be2, #00ffff) !important;
        color: #000 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.7);
    }
    
    /* Success/Warning/Info messages */
    .stSuccess, .stWarning, .stInfo {
        background: rgba(26, 29, 63, 0.8) !important;
        border-left: 4px solid #00ffff !important;
        border-radius: 10px !important;
        color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    /* Multiselect */
    .stMultiSelect > div > div > div {
        background: rgba(26, 29, 63, 0.8) !important;
        border: 2px solid rgba(0, 255, 255, 0.5) !important;
        border-radius: 10px !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(0, 255, 255, 0.2));
        border: 2px dashed rgba(0, 255, 255, 0.5);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e27;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8a2be2, #00ffff);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00ffff, #8a2be2);
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px #00ffff;
    }
    
    /* General text */
    p, span, div {
        color: #b8c5d6 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(138, 43, 226, 0.3), rgba(0, 255, 255, 0.3));
        border: 1px solid rgba(0, 255, 255, 0.5);
        border-radius: 10px;
        color: #00ffff !important;
        font-family: 'Orbitron', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

Name = st.text_input("🚀 Enter Your Name")
st.title(f"⚡ Hi {Name}! 😊")

# App title with emoji
st.title("🌌  DATA ANALYSIS HUB")

# File uploader
st.subheader("📤 Upload Your Dataset")
uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Rows", df.shape[0])
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
    
    # Sidebar
    st.sidebar.title('🎮 CONTROL PANEL')
    st.sidebar.write("Welcome to the Data Analysis Hub! Navigate through different modules to explore, visualize, and analyze your data with cutting-edge tools.")
    
    # Page selection
    page = st.sidebar.radio("🗂️ Select Module", ['Explore', 'Filtering', 'Select', 'Clean', 'Visualize', 'Grouping', 'Encoding', "Modeling", "About"])
    
    # Explore page
    if page == 'Explore':
        st.header("🔍 DATA EXPLORATION MODULE")
        option = st.selectbox("Choose an operation", ["None", "Head", "Tail", "Describe", "Dtypes", "Columns", "iLoc", "Missing Values"])
        
        if option == "Head":
            st.dataframe(df.head())
        elif option == "Tail":
            st.dataframe(df.tail())
        elif option == "Describe":
            st.dataframe(df.describe())
        elif option == "Dtypes":
            st.dataframe(df.dtypes)
        elif option == "Columns":
            st.write(df.columns.tolist())
        elif option == "iLoc":
            st.dataframe(df.iloc[10:20])
        elif option == "Missing Values":
            st.dataframe(df.isnull().sum())
        else:
            st.info("🎯 Select an option to explore your data")
    
    elif page == 'Select':
        st.header("📑 COLUMN & ROW SELECTOR")
        selected_cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=[])
        
        st.subheader("🔢 Row Selection")
        selection_mode = st.radio("Select rows by:", ["Index Range", "Specific Indices"])
        
        if selection_mode == "Index Range":
            start = st.number_input("Start index", min_value=0, max_value=len(df)-1, value=0)
            end = st.number_input("End index", min_value=0, max_value=len(df), value=5)
            if start < end:
                st.dataframe(df.iloc[int(start):int(end)][selected_cols])
            else:
                st.warning("⚠️ Start index must be less than end index.")
        elif selection_mode == "Specific Indices":
            selected_rows = st.multiselect("Choose row indices", options=list(range(len(df))))
            if selected_rows:
                st.dataframe(df.iloc[selected_rows][selected_cols])
            else:
                st.info("🎯 No rows selected.")
    
    # Filtering
    elif page == 'Filtering':
        st.header("🔍 DATA FILTERING MODULE")
        st.subheader("Filter by Column Values")
        
        all_cols = df.columns.tolist()
        col_to_filter = st.selectbox("Select a column to filter", all_cols)
        filter_condition = st.selectbox("Select filter condition", ["Equals", "Greater than", "Less than", "Contains"])
        
        if df[col_to_filter].dtype == 'object':
            filter_value = st.text_input(f"Enter value to filter {col_to_filter} by")
        else:
            filter_value = st.number_input(f"Enter value to filter {col_to_filter} by", value=0.0)
        
        if st.button("🚀 Apply Filter"):
            if filter_condition == "Equals":
                filtered_df = df[df[col_to_filter] == filter_value]
            elif filter_condition == "Greater than":
                filtered_df = df[df[col_to_filter] > filter_value]
            elif filter_condition == "Less than":
                filtered_df = df[df[col_to_filter] < filter_value]
            elif filter_condition == "Contains":
                filtered_df = df[df[col_to_filter].astype(str).str.contains(filter_value, na=False)]
            else:
                filtered_df = df
            
            if not filtered_df.empty:
                st.dataframe(filtered_df)
            else:
                st.warning("⚠️ No data found with the specified filter.")
    
    # Grouping
    elif page == 'Grouping':
        st.header("📊 GROUPING & AGGREGATION MODULE")
        st.subheader("Group by Column")
        
        group_col = st.selectbox("Select a column to group by", df.columns.tolist())
        agg_func = st.selectbox("Select aggregation function", ["Count", "Sum", "Mean", "Median", "Max", "Min"])
        
        if st.button("🚀 Group and Aggregate"):
            if agg_func == "Count":
                grouped_df = df.groupby(group_col).size().reset_index(name='Count')
            elif agg_func == "Sum":
                grouped_df = df.groupby(group_col).sum(numeric_only=True).reset_index()
            elif agg_func == "Mean":
                grouped_df = df.groupby(group_col).mean(numeric_only=True).reset_index()
            elif agg_func == "Median":
                grouped_df = df.groupby(group_col).median(numeric_only=True).reset_index()
            elif agg_func == "Max":
                grouped_df = df.groupby(group_col).max(numeric_only=True).reset_index()
            elif agg_func == "Min":
                grouped_df = df.groupby(group_col).min(numeric_only=True).reset_index()
            else:
                grouped_df = df
            st.dataframe(grouped_df)
        
        st.subheader("Group by Multiple Columns")
        group_cols = st.multiselect("Select columns to group by", df.columns.tolist())
        
        if group_cols:
            agg_func = st.selectbox("Select aggregation function for multiple columns", ["Count", "Sum", "Mean", "Median", "Max", "Min"])
            if st.button("🚀 Group by Multiple Columns"):
                if agg_func == "Count":
                    grouped_df = df.groupby(group_cols).size().reset_index(name='Count')
                elif agg_func == "Sum":
                    grouped_df = df.groupby(group_cols).sum(numeric_only=True).reset_index()
                elif agg_func == "Mean":
                    grouped_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
                elif agg_func == "Median":
                    grouped_df = df.groupby(group_cols).median(numeric_only=True).reset_index()
                elif agg_func == "Max":
                    grouped_df = df.groupby(group_cols).max(numeric_only=True).reset_index()
                elif agg_func == "Min":
                    grouped_df = df.groupby(group_cols).min(numeric_only=True).reset_index()
                else:
                    grouped_df = df
                st.dataframe(grouped_df)
    
    # Visualizations
    elif page == 'Visualize':
        st.header("📈 VISUALIZATION MODULE")
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        
        if not numeric_columns:
            st.warning("⚠️ No numeric columns to plot.")
        else:
            col1 = st.selectbox("Select column 1", numeric_columns)
            col2 = st.selectbox("Select column 2 (for scatter/heatmap)", numeric_columns)
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Bar Chart", "📊 Histogram", "📉 Line Plot", "🔵 Scatter Plot", "🔥 Heatmap", "📦 Box Plot", "🥧 Pie Chart"
            ])
            
            with tab1:
                st.subheader("Bar Chart")
                st.bar_chart(df[col1].value_counts())
            
            with tab2:
                st.subheader("Histogram")
                fig, ax = plt.subplots()
                ax.hist(df[col1].dropna(), bins=20, color='#00ffff', edgecolor='#8a2be2')
                ax.set_title(f"Histogram of {col1}")
                ax.set_facecolor('#0a0e27')
                fig.patch.set_facecolor('#0a0e27')
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Line Plot")
                st.line_chart(df[col1])
            
            with tab4:
                st.subheader("Scatter Plot")
                fig, ax = plt.subplots()
                ax.scatter(df[col1], df[col2], alpha=0.7, color='#00ffff', edgecolors='#8a2be2')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"{col1} vs {col2}")
                ax.set_facecolor('#0a0e27')
                fig.patch.set_facecolor('#0a0e27')
                st.pyplot(fig)
            
            with tab5:
                st.subheader("Heatmap (Correlation)")
                numeric_df = df.select_dtypes(include='number')
                if numeric_df.shape[1] < 2:
                    st.warning("⚠️ Not enough numeric columns to display a correlation heatmap.")
                else:
                    corr = numeric_df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="cool", ax=ax)
                    ax.set_facecolor('#0a0e27')
                    fig.patch.set_facecolor('#0a0e27')
                    st.pyplot(fig)
            
            with tab6:
                st.subheader("Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[col1], ax=ax, color='#00ffff')
                ax.set_title(f"Boxplot of {col1}")
                ax.set_facecolor('#0a0e27')
                fig.patch.set_facecolor('#0a0e27')
                st.pyplot(fig)
            
            with tab7:
                st.subheader("🥧 Pie Chart (Categorical Distribution)")
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                if not cat_cols:
                    st.warning("⚠️ No categorical columns available for pie chart.")
                else:
                    pie_col = st.selectbox("Select a categorical column", cat_cols, key="pie_col")
                    if pie_col:
                        pie_data = df[pie_col].value_counts()
                        fig, ax = plt.subplots()
                        colors = ['#00ffff', '#8a2be2', '#ff00ff', '#00ff00', '#ffff00']
                        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax.axis("equal")
                        fig.patch.set_facecolor('#0a0e27')
                        st.pyplot(fig)
    
    elif page == 'Clean':
        st.header("🧹 DATA CLEANING MODULE")
        
        st.subheader("🔄 Convert to Numeric (if applicable)")
        df_numeric = df.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        st.info("✨ All columns attempted to convert to numeric. Non-convertible values are set to NaN.")
        
        st.subheader("🔍 Missing Values Overview After Conversion")
        missing_vals = df_numeric.isnull().sum()
        st.dataframe(missing_vals[missing_vals > 0])
        
        if missing_vals.sum() == 0:
            st.success("✅ No missing values found.")
        else:
            st.warning("⚠️ Missing values found. Use the options below to clean.")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗑 Remove Duplicates",
            "🔢 Fill with Mean",
            "🔣 Fill with Median",
            "🔁 Fill with Mode",
            "🧯 Remove Columns"
        ])
        
        with tab1:
            st.subheader("Remove Duplicate Rows")
            if st.button("🚀 Remove Duplicates"):
                initial_shape = df_numeric.shape
                df_numeric.drop_duplicates(inplace=True)
                st.success(f"✅ Removed duplicates. New shape: {df_numeric.shape} (was {initial_shape})")
        
        with tab2:
            st.subheader("Fill Missing Values with Mean")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            col = st.selectbox("Select numeric column", numeric_cols, key="mean_col")
            if st.button("🚀 Fill with Mean"):
                mean_val = df_numeric[col].mean()
                df_numeric[col].fillna(mean_val, inplace=True)
                st.success(f"✅ Filled `{col}` with mean: {mean_val:.2f}")
        
        with tab3:
            st.subheader("Fill Missing Values with Median")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            col = st.selectbox("Select numeric column", numeric_cols, key="median_col")
            if st.button("🚀 Fill with Median"):
                median_val = df_numeric[col].median()
                df_numeric[col].fillna(median_val, inplace=True)
                st.success(f"✅ Filled `{col}` with median: {median_val:.2f}")
        
        with tab4:
            st.subheader("Fill Missing Values with Mode")
            all_cols = df_numeric.columns.tolist()
            col = st.selectbox("Select a column", all_cols, key="mode_col")
            if st.button("🚀 Fill with Mode"):
                mode_val = df_numeric[col].mode()
                if not mode_val.empty:
                    df_numeric[col].fillna(mode_val[0], inplace=True)
                    st.success(f"✅ Filled `{col}` with mode: {mode_val[0]}")
                else:
                    st.warning("⚠️ No mode available for this column.")
        
        with tab5:
            st.subheader("Remove Unnecessary Columns")
            all_cols = df_numeric.columns.tolist()
            cols_to_remove = st.multiselect("Select columns to remove", all_cols)
            if st.button("🚀 Remove Selected Columns"):
                if cols_to_remove:
                    df_numeric.drop(columns=cols_to_remove, inplace=True)
                    st.success(f"🗑 Removed columns: {', '.join(cols_to_remove)}")
                else:
                    st.warning("⚠️ No columns selected for removal.")
    
    # Encoding
    elif page == 'Encoding':
        st.header("🔠 ENCODING MODULE")
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols:
            st.warning("⚠️ No categorical columns to encode.")
        else:
            col = st.selectbox("Select a categorical column to encode", categorical_cols)
            encoding_type = st.selectbox("Select encoding type", ["One-Hot Encoding", "Label Encoding"])
            
            if encoding_type == "One-Hot Encoding":
                encoded_df = pd.get_dummies(df, columns=[col], drop_first=True)
                st.dataframe(encoded_df.head())
            elif encoding_type == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.dataframe(df.head())
            else:
                st.info("🎯 Select an encoding type to proceed.")
    
    # Modeling - UPDATED SECTION
    elif page == 'Modeling':
        st.header("🤖 MACHINE LEARNING MODULE")
        
        # Problem type selection
        problem_type = st.radio("🎯 Select Problem Type", ["Classification", "Regression"])
        
        target_col = st.selectbox("🎯 Select the target column", df.columns)
        feature_cols = st.multiselect("📊 Select feature columns", [col for col in df.columns if col != target_col])
        
        # Model selection based on problem type
        if problem_type == "Classification":
            model_type = st.selectbox("Choose Classification Model", [
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "K-Nearest Neighbors (KNN)",
                "Support Vector Machine (SVM)",
                "Naive Bayes",
                "Gradient Boosting Classifier",
                "XGBoost Classifier"
            ])
        else:
            model_type = st.selectbox("Choose Regression Model", [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Support Vector Regressor (SVR)",
                "Gradient Boosting Regressor",
                "XGBoost Regressor"
            ])
        
        # Advanced options
        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        if st.button("🚀 Train Model"):
            if not feature_cols:
                st.error("❌ Please select at least one feature column!")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                            mean_squared_error, r2_score, mean_absolute_error)
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC, SVR
                from sklearn.naive_bayes import GaussianNB
                
                try:
                    data = df[feature_cols + [target_col]].dropna()
                    
                    if data.empty:
                        st.error("❌ No data available after removing missing values!")
                    else:
                        # Encode categorical features
                        label_encoders = {}
                        for col in feature_cols:
                            if data[col].dtype == 'object':
                                label_encoders[col] = LabelEncoder()
                                data[col] = label_encoders[col].fit_transform(data[col])
                        
                        # Encode target if classification
                        target_encoder = None
                        if problem_type == "Classification" and data[target_col].dtype == 'object':
                            target_encoder = LabelEncoder()
                            data[target_col] = target_encoder.fit_transform(data[target_col])
                        
                        X = data[feature_cols]
                        y = data[target_col]
                        
                        # Scale features for certain models
                        scale_models = ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", 
                                      "Support Vector Regressor (SVR)", "Ridge Regression", "Lasso Regression"]
                        scaler = None
                        if model_type in scale_models:
                            scaler = StandardScaler()
                            X = scaler.fit_transform(X)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
                        
                        # Initialize model
                        if model_type == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000, random_state=int(random_state))
                        elif model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Ridge Regression":
                            model = Ridge(random_state=int(random_state))
                        elif model_type == "Lasso Regression":
                            model = Lasso(random_state=int(random_state))
                        elif model_type == "Decision Tree Classifier":
                            model = DecisionTreeClassifier(random_state=int(random_state))
                        elif model_type == "Decision Tree Regressor":
                            model = DecisionTreeRegressor(random_state=int(random_state))
                        elif model_type == "Random Forest Classifier":
                            model = RandomForestClassifier(random_state=int(random_state))
                        elif model_type == "Random Forest Regressor":
                            model = RandomForestRegressor(random_state=int(random_state))
                        elif model_type == "K-Nearest Neighbors (KNN)":
                            model = KNeighborsClassifier()
                        elif model_type == "Support Vector Machine (SVM)":
                            model = SVC(random_state=int(random_state))
                        elif model_type == "Support Vector Regressor (SVR)":
                            model = SVR()
                        elif model_type == "Naive Bayes":
                            model = GaussianNB()
                        elif model_type == "Gradient Boosting Classifier":
                            model = GradientBoostingClassifier(random_state=int(random_state))
                        elif model_type == "Gradient Boosting Regressor":
                            model = GradientBoostingRegressor(random_state=int(random_state))
                        elif model_type in ["XGBoost Classifier", "XGBoost Regressor"]:
                            try:
                                import xgboost as xgb
                                if model_type == "XGBoost Classifier":
                                    model = xgb.XGBClassifier(random_state=int(random_state), eval_metric='logloss')
                                else:
                                    model = xgb.XGBRegressor(random_state=int(random_state))
                            except ImportError:
                                st.error("❌ XGBoost not installed. Using Gradient Boosting instead.")
                                if model_type == "XGBoost Classifier":
                                    model = GradientBoostingClassifier(random_state=int(random_state))
                                else:
                                    model = GradientBoostingRegressor(random_state=int(random_state))
                        
                        # Train model
                        with st.spinner("🔄 Training model..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Display results based on problem type
                        if problem_type == "Classification":
                            st.subheader("📊 Classification Metrics")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                accuracy = accuracy_score(y_test, y_pred)
                                st.metric("🎯 Accuracy", f"{accuracy:.4f}")
                            
                            with col2:
                                st.metric("📈 Training Samples", len(X_train))
                            
                            st.subheader("📋 Classification Report")
                            st.text(classification_report(y_test, y_pred))
                            
                            st.subheader("🔥 Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='cool', ax=ax)
                            ax.set_title("Confusion Matrix")
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                            
                        else:  # Regression
                            st.subheader("📊 Regression Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                mse = mean_squared_error(y_test, y_pred)
                                st.metric("📉 MSE", f"{mse:.4f}")
                            with col2:
                                mae = mean_absolute_error(y_test, y_pred)
                                st.metric("📊 MAE", f"{mae:.4f}")
                            with col3:
                                r2 = r2_score(y_test, y_pred)
                                st.metric("📈 R² Score", f"{r2:.4f}")
                            
                            # Prediction vs Actual plot
                            st.subheader("📈 Predictions vs Actual Values")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_test, y_pred, alpha=0.6, color='#00ffff', edgecolors='#8a2be2')
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax.set_xlabel("Actual Values")
                            ax.set_ylabel("Predicted Values")
                            ax.set_title("Actual vs Predicted")
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                        
                        # Feature importance for tree-based models
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("🎯 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#00ffff', edgecolor='#8a2be2')
                            ax.set_xlabel('Importance')
                            ax.set_title('Feature Importance')
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                except:
                    print("try again")
    
    # About page
    elif page == 'About':
        st.header("ℹ️ ABOUT THIS MODULE")
        st.markdown("""
        ### 🌌 **FUTURISTIC DATA ANALYSIS HUB**
        
        Welcome to the next generation of data analysis tools, built with cutting-edge technology and designed for the future!
        
        ---
        
        #### 🎯 **Key Features:**
        
        **🗂 Data Exploration**
        - Upload and explore CSV datasets with lightning speed
        - View data statistics, structure, and insights instantly
        
        **🔍 Advanced Filtering**
        - Apply complex filters to your data
        - Multi-condition filtering for precise data selection
        
        **📊 Dynamic Visualizations**
        - Create stunning charts and graphs
        - Interactive visual analytics powered by Matplotlib & Seaborn
        
        **🧹 Intelligent Data Cleaning**
        - Handle missing values with multiple strategies
        - Remove duplicates and outliers effortlessly
        
        **📈 Grouping & Aggregation**
        - Group data by single or multiple columns
        - Perform statistical aggregations on the fly
        
        **🔠 Smart Encoding**
        - One-hot encoding for categorical variables
        - Label encoding with automatic handling
        
        **🤖 Machine Learning Integration**
        - Build ML models directly in the app
        - **16 Total Models** - 8 Classification & 8 Regression algorithms
        - Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes, Gradient Boosting, XGBoost
        - Regression: Linear, Ridge, Lasso, Decision Tree, Random Forest, SVR, Gradient Boosting, XGBoost
        - Advanced metrics and visualizations
        - Feature importance analysis
        - Confusion matrix and performance plots
        
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
        - **Machine Learning:** Scikit-learn, XGBoost
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
        
        ---
        
        ### 🙏 **Thank You!**
        
        Thank you for using the Futuristic Data Analysis Hub! We hope this tool empowers your data journey and makes analysis an exciting experience.
        
        *Built with ❤️ and ⚡ by Jatin Kumar*
        """)
        
        # Add some visual flair with columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🚀")
            st.write("**Fast Processing**")
        with col2:
            st.markdown("### 🎨")
            st.write("**Beautiful UI**")
        with col3:
            st.markdown("### 🔒")
            st.write("**Secure & Private**")

else:
    # Welcome screen when no file is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='font-size: 3em;'>🌌</h2>
        <h3>Welcome to the Future of Data Analysis</h3>
        <p style='font-size: 1.2em; margin-top: 20px;'>
            Upload your CSV file to begin your journey into advanced data exploration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("📁 Please upload a CSV file using the uploader above to access all features")
