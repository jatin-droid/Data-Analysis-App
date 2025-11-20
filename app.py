import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Global matplotlib config – smaller figs = faster
plt.rcParams["figure.figsize"] = (6, 4)

st.set_page_config(page_title="Automated Data Analysis App", layout="wide")

# -------------------- PERFORMANCE HELPERS (CACHING) -------------------- #

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()

@st.cache_data
def value_counts_series(df: pd.DataFrame, col: str):
    return df[col].value_counts()

@st.cache_data
def compute_corr(numeric_df: pd.DataFrame):
    return numeric_df.corr()

@st.cache_data
def to_numeric_df(df: pd.DataFrame):
    df_numeric = df.copy()
    for c in df_numeric.columns:
        df_numeric[c] = pd.to_numeric(df_numeric[c], errors="coerce")
    return df_numeric

# -------------------- LIGHTER FUTURISTIC PINK THEME -------------------- #

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

# -------------------- TOP UI -------------------- #

Name = st.text_input("🚀 Enter Your Name")
st.title(f"⚡ Hi {Name}! 😊" if Name else "⚡ Hi Data Explorer! 😊")
st.title("🌌  DATA ANALYSIS HUB")

st.subheader("📤 Upload Your Dataset")
uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")

if uploaded_file:
    # Cached CSV loading
    df = load_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Rows", df.shape[0])
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        st.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
    
    st.sidebar.title('🎮 CONTROL PANEL')
    st.sidebar.write(
        "Welcome to the Data Analysis Hub! Navigate through different modules "
        "to explore, visualize, and analyze your data with cutting-edge tools."
    )
    
    page = st.sidebar.radio(
        "🗂️ Select Module",
        ['Explore', 'Filtering', 'Select', 'Clean', 'Visualize', 'Grouping', 'Encoding', "Modeling", "About"]
    )
    
    # -------------------- EXPLORE -------------------- #
    if page == 'Explore':
        st.header("🔍 DATA EXPLORATION MODULE")
        option = st.selectbox(
            "Choose an operation",
            ["None", "Info", "Head", "Tail", "Describe", "Dtypes", "Columns", "iLoc", "Missing Values"]
        )
        
        if option == "Head":
            st.dataframe(df.head())
        elif option == "Info":
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        elif option == "Tail":
            st.dataframe(df.tail())
        elif option == "Describe":
            st.dataframe(df.describe(include="all"))
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
    
    # -------------------- SELECT -------------------- #
    elif page == 'Select':
        st.header("📑 COLUMN & ROW SELECTOR")
        selected_cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=[])
        
        st.subheader("🔢 Row Selection")
        selection_mode = st.radio("Select rows by:", ["Index Range", "Specific Indices"])
        
        if selection_mode == "Index Range":
            start = st.number_input("Start index", min_value=0, max_value=len(df)-1, value=0)
            end = st.number_input("End index", min_value=0, max_value=len(df), value=min(5, len(df)))
            if start < end:
                subset = df.iloc[int(start):int(end)]
                if selected_cols:
                    subset = subset[selected_cols]
                st.dataframe(subset)
            else:
                st.warning("⚠️ Start index must be less than end index.")
        else:
            selected_rows = st.multiselect("Choose row indices", options=list(range(len(df))))
            if selected_rows:
                subset = df.iloc[selected_rows]
                if selected_cols:
                    subset = subset[selected_cols]
                st.dataframe(subset)
            else:
                st.info("🎯 No rows selected.")
    
    # -------------------- FILTERING -------------------- #
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
                filtered_df = df[df[col_to_filter].astype(str).str.contains(str(filter_value), na=False)]
            else:
                filtered_df = df
            
            if not filtered_df.empty:
                st.dataframe(filtered_df)
            else:
                st.warning("⚠️ No data found with the specified filter.")
    
    # -------------------- GROUPING -------------------- #
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
            agg_func_multi = st.selectbox(
                "Select aggregation function for multiple columns",
                ["Count", "Sum", "Mean", "Median", "Max", "Min"]
            )
            if st.button("🚀 Group by Multiple Columns"):
                if agg_func_multi == "Count":
                    grouped_df = df.groupby(group_cols).size().reset_index(name='Count')
                elif agg_func_multi == "Sum":
                    grouped_df = df.groupby(group_cols).sum(numeric_only=True).reset_index()
                elif agg_func_multi == "Mean":
                    grouped_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()
                elif agg_func_multi == "Median":
                    grouped_df = df.groupby(group_cols).median(numeric_only=True).reset_index()
                elif agg_func_multi == "Max":
                    grouped_df = df.groupby(group_cols).max(numeric_only=True).reset_index()
                elif agg_func_multi == "Min":
                    grouped_df = df.groupby(group_cols).min(numeric_only=True).reset_index()
                else:
                    grouped_df = df
                st.dataframe(grouped_df)
    
    # -------------------- VISUALIZE -------------------- #
    elif page == 'Visualize':
        st.header("📈 VISUALIZATION MODULE")
        numeric_columns = get_numeric_columns(df)
        
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
                vc = value_counts_series(df, col1)
                st.bar_chart(vc)
            
            with tab2:
                st.subheader("Histogram")
                fig, ax = plt.subplots()
                ax.hist(df[col1].dropna(), bins=20, color='#FFB6C1', edgecolor='#8a2be2')
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
                ax.scatter(df[col1], df[col2], alpha=0.7, color='#FFB6C1', edgecolors='#8a2be2')
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
                    if st.button("🔍 Compute Correlation Heatmap"):
                        corr = compute_corr(numeric_df)
                        fig, ax = plt.subplots()
                        sns.heatmap(corr, annot=True, cmap="cool", ax=ax)
                        ax.set_facecolor('#0a0e27')
                        fig.patch.set_facecolor('#0a0e27')
                        st.pyplot(fig)
                    else:
                        st.info("Click the button to compute the heatmap (saves resources).")
            
            with tab6:
                st.subheader("Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[col1], ax=ax, color='#FFB6C1')
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
                        pie_data = value_counts_series(df, pie_col)
                        fig, ax = plt.subplots()
                        colors = ['#FFB6C1', '#8a2be2', '#ff99aa', '#ffdeea', '#ffccff']
                        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax.axis("equal")
                        fig.patch.set_facecolor('#0a0e27')
                        st.pyplot(fig)
    
    # -------------------- CLEAN -------------------- #
    elif page == 'Clean':
        st.header("🧹 DATA CLEANING MODULE")
        
        st.subheader("🔄 Convert to Numeric (if applicable)")
        df_numeric = to_numeric_df(df)
        st.info("✨ All columns attempted to convert to numeric. Non-convertible values are set to NaN.")
        
        st.subheader("🔍 Missing Values Overview After Conversion")
        missing_vals = df_numeric.isnull().sum()
        mv = missing_vals[missing_vals > 0]
        if not mv.empty:
            st.dataframe(mv)
            st.warning("⚠️ Missing values found. Use the options below to clean.")
        else:
            st.success("✅ No missing values found.")
        
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
                df_numeric = df_numeric.drop_duplicates()
                st.success(f"✅ Removed duplicates. New shape: {df_numeric.shape} (was {initial_shape})")
        
        with tab2:
            st.subheader("Fill Missing Values with Mean")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column", numeric_cols, key="mean_col")
                if st.button("🚀 Fill with Mean"):
                    mean_val = df_numeric[col].mean()
                    df_numeric[col] = df_numeric[col].fillna(mean_val)
                    st.success(f"✅ Filled `{col}` with mean: {mean_val:.2f}")
            else:
                st.info("No numeric columns available.")
        
        with tab3:
            st.subheader("Fill Missing Values with Median")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column", numeric_cols, key="median_col")
                if st.button("🚀 Fill with Median"):
                    median_val = df_numeric[col].median()
                    df_numeric[col] = df_numeric[col].fillna(median_val)
                    st.success(f"✅ Filled `{col}` with median: {median_val:.2f}")
            else:
                st.info("No numeric columns available.")
        
        with tab4:
            st.subheader("Fill Missing Values with Mode")
            all_cols = df_numeric.columns.tolist()
            col = st.selectbox("Select a column", all_cols, key="mode_col")
            if st.button("🚀 Fill with Mode"):
                mode_val = df_numeric[col].mode()
                if not mode_val.empty:
                    df_numeric[col] = df_numeric[col].fillna(mode_val[0])
                    st.success(f"✅ Filled `{col}` with mode: {mode_val[0]}")
                else:
                    st.warning("⚠️ No mode available for this column.")
        
        with tab5:
            st.subheader("Remove Unnecessary Columns")
            all_cols = df_numeric.columns.tolist()
            cols_to_remove = st.multiselect("Select columns to remove", all_cols)
            if st.button("🚀 Remove Selected Columns"):
                if cols_to_remove:
                    df_numeric = df_numeric.drop(columns=cols_to_remove)
                    st.success(f"🗑 Removed columns: {', '.join(cols_to_remove)}")
                else:
                    st.warning("⚠️ No columns selected for removal.")
        
        st.subheader("Preview of Cleaned Data (Not persisted)")
        st.dataframe(df_numeric.head())
    
    # -------------------- ENCODING -------------------- #
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
            else:
                from sklearn.preprocessing import LabelEncoder
                data_enc = df.copy()
                le = LabelEncoder()
                data_enc[col] = le.fit_transform(data_enc[col])
                st.dataframe(data_enc.head())
    
    # -------------------- MODELING -------------------- #
    elif page == 'Modeling':
        st.header("🤖 MACHINE LEARNING MODULE")
        
        problem_type = st.radio("🎯 Select Problem Type", ["Classification", "Regression"])
        
        target_col = st.selectbox("🎯 Select the target column", df.columns)
        feature_cols = st.multiselect(
            "📊 Select feature columns",
            [col for col in df.columns if col != target_col]
        )
        
        if problem_type == "Classification":
            model_type = st.selectbox(
                "Choose Classification Model",
                [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "K-Nearest Neighbors (KNN)",
                    "Support Vector Machine (SVM)",
                    "Naive Bayes",
                    "Gradient Boosting Classifier",
                    "XGBoost Classifier"
                ]
            )
        else:
            model_type = st.selectbox(
                "Choose Regression Model",
                [
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "Support Vector Regressor (SVR)",
                    "Gradient Boosting Regressor",
                    "XGBoost Regressor"
                ]
            )
        
        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        if st.button("🚀 Train Model"):
            if not feature_cols:
                st.error("❌ Please select at least one feature column!")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import (
                    accuracy_score, classification_report, confusion_matrix,
                    mean_squared_error, r2_score, mean_absolute_error
                )
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.linear_model import (
                    LogisticRegression, LinearRegression, Ridge, Lasso
                )
                from sklearn.tree import (
                    DecisionTreeClassifier, DecisionTreeRegressor
                )
                from sklearn.ensemble import (
                    RandomForestClassifier, RandomForestRegressor,
                    GradientBoostingClassifier, GradientBoostingRegressor
                )
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC, SVR
                from sklearn.naive_bayes import GaussianNB

                try:
                    data = df[feature_cols + [target_col]].dropna().copy()
                    
                    if data.empty:
                        st.error("❌ No data available after removing missing values!")
                    else:
                        label_encoders = {}
                        for col in feature_cols:
                            if data[col].dtype == 'object':
                                le = LabelEncoder()
                                data[col] = le.fit_transform(data[col])
                                label_encoders[col] = le
                        
                        target_encoder = None
                        if problem_type == "Classification" and data[target_col].dtype == 'object':
                            target_encoder = LabelEncoder()
                            data[target_col] = target_encoder.fit_transform(data[target_col])
                        
                        X = data[feature_cols]
                        y = data[target_col]
                        
                        scale_models = [
                            "Support Vector Machine (SVM)",
                            "K-Nearest Neighbors (KNN)",
                            "Support Vector Regressor (SVR)",
                            "Ridge Regression",
                            "Lasso Regression",
                        ]
                        scaler = None
                        if model_type in scale_models:
                            scaler = StandardScaler()
                            X = scaler.fit_transform(X)
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=int(random_state)
                        )
                        
                        # Model selection
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
                                st.error("❌ XGBoost not installed. Falling back to Gradient Boosting.")
                                if model_type == "XGBoost Classifier":
                                    model = GradientBoostingClassifier(random_state=int(random_state))
                                else:
                                    model = GradientBoostingRegressor(random_state=int(random_state))
                        else:
                            st.error("❌ Unknown model type selected.")
                            st.stop()
                        
                        with st.spinner("🔄 Training model..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        st.success("✅ Model trained successfully!")
                        
                        if problem_type == "Classification":
                            st.subheader("📊 Classification Metrics")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                acc = accuracy_score(y_test, y_pred)
                                st.metric("🎯 Accuracy", f"{acc:.4f}")
                            with col_b:
                                st.metric("📈 Training Samples", len(X_train))
                            
                            st.subheader("📋 Classification Report")
                            st.text(classification_report(y_test, y_pred))
                            
                            st.subheader("🔥 Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='cool', ax=ax)
                            ax.set_title("Confusion Matrix")
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                        else:
                            st.subheader("📊 Regression Metrics")
                            col_a, col_b, col_c = st.columns(3)
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            with col_a:
                                st.metric("📉 MSE", f"{mse:.4f}")
                            with col_b:
                                st.metric("📊 MAE", f"{mae:.4f}")
                            with col_c:
                                st.metric("📈 R² Score", f"{r2:.4f}")
                            
                            st.subheader("📈 Predictions vs Actual Values")
                            fig, ax = plt.subplots()
                            ax.scatter(y_test, y_pred, alpha=0.6, color='#FFB6C1', edgecolors='#8a2be2')
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                            ax.set_xlabel("Actual Values")
                            ax.set_ylabel("Predicted Values")
                            ax.set_title("Actual vs Predicted")
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                        
                        if hasattr(model, "feature_importances_"):
                            st.subheader("🎯 Feature Importance")
                            importance_df = pd.DataFrame({
                                "Feature": feature_cols,
                                "Importance": model.feature_importances_
                            }).sort_values("Importance", ascending=False)
                            
                            fig, ax = plt.subplots()
                            ax.barh(importance_df["Feature"], importance_df["Importance"], color='#F4C2C2', edgecolor='#8a2be2')
                            ax.set_xlabel("Importance")
                            ax.set_title("Feature Importance")
                            ax.set_facecolor('#0a0e27')
                            fig.patch.set_facecolor('#0a0e27')
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ An error occurred during modeling: {e}")
    
    # -------------------- ABOUT -------------------- #
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
        - **16 Total Models** – 8 Classification & 8 Regression algorithms
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
        
        ---
        
        ### 🙏 Thank You
        
        Thank you for using the Futuristic Data Analysis Hub!  
        *Built with ❤️ and ⚡ by Jatin Kumar*
        """)
        
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
    st.markdown(
        """
        <div class="info-box">
            <h2 style='font-size: 2.4em;'>🌌 Futuristic Data Analysis Hub</h2>
            <p style='font-size: 1.1em; margin-top: 15px;'>
                Upload your CSV file to begin your journey into advanced data exploration,
                visualization, cleaning, encoding, and machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("📁 Please upload a CSV file using the uploader above to access all features.")
