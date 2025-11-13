import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Inject custom CSS based on your theme
st.markdown("""
    <style>
        .stApp {
            background-color: #F1D9D0;  /* Main background */
            color: #56DFCF;             /* Text color */
        }

        section[data-testid="stSidebar"] {
            background-color: #097d5e !important; /* Sidebar */
            color: #03A6A1;
        }

        .stButton>button {
            background-color: #0ABAB5 !important;
            color: black !important;
            border-radius: 8px;
        }

        .stDataFrame, .stTable {
            background-color: #FFEDF3 !important;
        }

        h1, h2, h3 {
            color: #9e8b91;
        }

        .css-1r6slb0 {
            color: #56DFCF !important;
        }
    </style>
""", unsafe_allow_html=True)

Name=st.text_input("Enter Your Name")
st.title(f"Hi {Name} ! 😊")


# App title
st.title("📊 Data Analysis App")


# File uploader
st.subheader("Upload a CSV file")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write(f"📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    # Sidebar
    st.sidebar.title('🧭 Sidebar')
    st.sidebar.write("welcome to the Data Analysis App! This app allows you to explore, filter, visualize, and clean your data interactively. Upload your CSV file to get started.")


    # Page selection
    page = st.sidebar.radio("Select page", ['Explore', 'Filtering', 'Select','Clean','Visualize','Grouping','Encoding',"Modeling","About"])

    # Explore page
    if page == 'Explore':
        st.header("🔍 Explore the Data")
        option = st.selectbox("Choose an option", ["None", "Head", "Tail", "To Markdown", "Describe", "Dtypes", "Columns", "iLoc", "Missing Values"])
        if option == "Head":
            st.dataframe(df.head())
        elif option == "Tail":
            st.dataframe(df.tail())
        elif option == "To Markdown":
            st.markdown(df.to_markdown())
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
            st.info("Select an option to explore.")

    elif page == 'Select':
        st.header("📑 Select Specific Columns and Rows")

        selected_cols = st.multiselect("Choose columns to display", df.columns.tolist(), default=[])
    
        st.subheader("🔢 Row Selection")
        selection_mode = st.radio("Select rows by:", ["Index Range", "Specific Indices"])

        if selection_mode == "Index Range":
            start = st.number_input("Start index", min_value=0, max_value=len(df)-1, value=0)
            end = st.number_input("End index", min_value=0, max_value=len(df), value=5)
            if start < end:
                st.dataframe(df.iloc[int(start):int(end)][selected_cols])
            else:
                st.warning("Start index must be less than end index.")

        elif selection_mode == "Specific Indices":
            selected_rows = st.multiselect("Choose row indices", options=list(range(len(df))))
            if selected_rows:
                st.dataframe(df.iloc[selected_rows][selected_cols])
            else:
                st.info("No rows selected.")
 
            
    # Filtering
    elif page == 'Filtering':
        st.header("🔍 Filter the Data")
        st.subheader("Filter by Column Values")

        # Select column to filter
        all_cols = df.columns.tolist()
        col_to_filter = st.selectbox("Select a column to filter", all_cols)

        # Select filter condition
        filter_condition = st.selectbox("Select filter condition", ["Equals", "Greater than", "Less than", "Contains"])

        # Input for filter value
        if df[col_to_filter].dtype == 'object':
            filter_value = st.text_input(f"Enter value to filter {col_to_filter} by")
        else:
            filter_value = st.number_input(f"Enter value to filter {col_to_filter} by", value=0.0)

        if st.button("Apply Filter"):
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
                st.warning("No data found with the specified filter.")
    # Filtering by multiple conditions
        st.subheader("Filter by Multiple Conditions")
        if st.button("Filter by Multiple Conditions"):
            filter_conditions = []
            for col in all_cols:
                if df[col].dtype == 'object':
                    value = st.text_input(f"Filter {col} by (text)", key=col)
                    if value:
                        filter_conditions.append(df[col].str.contains(value, na=False))
                else:
                    value = st.number_input(f"Filter {col} by (number)", key=col, value=0.0)
                    filter_conditions.append(df[col] == value)

            if filter_conditions:
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter &= condition

                filtered_df = df[combined_filter]
                if not filtered_df.empty:
                    st.dataframe(filtered_df)
                else:
                    st.warning("No data found with the specified filters.")

    # Grouping
    elif page == 'Grouping':
        st.header("📊 Grouping and Aggregation")
        st.subheader("Group by Column")

        # Select column to group by
        group_col = st.selectbox("Select a column to group by", df.columns.tolist())

        # Select aggregation function
        agg_func = st.selectbox("Select aggregation function", ["Count", "Sum", "Mean", "Median", "Max", "Min"])

        if st.button("Group and Aggregate"):
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
    # Group by multiple columns
        st.subheader("Group by Multiple Columns")                                               
        group_cols = st.multiselect("Select columns to group by", df.columns.tolist())                  
        if group_cols:      
            agg_func = st.selectbox("Select aggregation function for multiple columns", ["Count", "Sum", "Mean", "Median", "Max", "Min"])
            if st.button("Group by Multiple Columns"):
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
        st.header("📈 Visualize the Data")
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        if not numeric_columns:
            st.warning("No numeric columns to plot.")
        else:
            col1 = st.selectbox("Select column 1", numeric_columns)
            col2 = st.selectbox("Select column 2 (for scatter/heatmap)", numeric_columns)

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Bar Chart", "📊 Histogram", "📉 Line Plot", "🔵 Scatter Plot", "🔥 Heatmap", "📦 Box Plot","🥧 pie chart"
            ])

            with tab1:
                st.subheader("Bar Chart")
                st.bar_chart(df[col1].value_counts())

            with tab2:
                st.subheader("Histogram")
                fig, ax = plt.subplots()
                ax.hist(df[col1].dropna(), bins=20, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {col1}")
                st.pyplot(fig)

            with tab3:
                st.subheader("Line Plot")
                st.line_chart(df[col1])

            with tab4:
                st.subheader("Scatter Plot")
                fig, ax = plt.subplots()
                ax.scatter(df[col1], df[col2], alpha=0.7)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"{col1} vs {col2}")
                st.pyplot(fig)

            with tab5:
                st.subheader("Heatmap (Correlation)")

            # Select only numeric columns for correlation
                numeric_df = df.select_dtypes(include='number')

                if numeric_df.shape[1] < 2:
                    st.warning("Not enough numeric columns to display a correlation heatmap.")
                else:
                    corr = numeric_df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
    

            with tab6:
                st.subheader("Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[col1], ax=ax)
                ax.set_title(f"Boxplot of {col1}")
                st.pyplot(fig)
            with tab7:
                st.subheader("🥧 Pie Chart (Categorical Distribution)")

                cat_cols = df.select_dtypes(include='object').columns.tolist()
                if not cat_cols:
                    st.warning("No categorical columns available for pie chart.")
                else:
                    pie_col = st.selectbox("Select a categorical column", cat_cols, key="pie_col")
                    if pie_col:
                        pie_data = df[pie_col].value_counts()
                        fig, ax = plt.subplots()
                        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                        ax.axis("equal")  # Equal aspect ratio ensures the pie is circular
                        st.pyplot(fig)

 
    elif page == 'Clean':
        st.header("🧹 Clean the Data")

    # 1. Convert applicable columns to numeric
        st.subheader("🔄 Convert to Numeric (if applicable)")
        df_numeric = df.copy()
        for col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        st.info("All columns attempted to convert to numeric. Non-convertible values are set to NaN.")

# 2. Show null value overview after conversion
        st.subheader("🔍 Missing Values Overview After Conversion")
        missing_vals = df_numeric.isnull().sum()
        st.dataframe(missing_vals[missing_vals > 0])

        if missing_vals.sum() == 0:
            st.success("✅ No missing values found.")
        else:
            st.warning("⚠️ Missing values found. Use the options below to clean.")

    # 3. Tabbed cleaning interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑 Remove Duplicates",
        "🔢 Fill with Mean",
        "🔣 Fill with Median",
        "🔁 Fill with Mode",
        "🧯 Remove Columns"
        ])

    # Tab 1: Remove Duplicates
        with tab1:
            st.subheader("Remove Duplicate Rows")
            if st.button("Remove Duplicates"):
                initial_shape = df_numeric.shape
                df_numeric.drop_duplicates(inplace=True)
                st.success(f"✅ Removed duplicates. New shape: {df_numeric.shape} (was {initial_shape})")

    # Tab 2: Fill with Mean
        with tab2:
            st.subheader("Fill Missing Values with Mean")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            col = st.selectbox("Select numeric column", numeric_cols, key="mean_col")
            if st.button("Fill with Mean"):
                mean_val = df_numeric[col].mean()
                df_numeric[col].fillna(mean_val, inplace=True)
                st.success(f"✅ Filled `{col}` with mean: {mean_val:.2f}")

    # Tab 3: Fill with Median
        with tab3:
            st.subheader("Fill Missing Values with Median")
            numeric_cols = df_numeric.select_dtypes(include='number').columns.tolist()
            col = st.selectbox("Select numeric column", numeric_cols, key="median_col")
            if st.button("Fill with Median"):
                median_val = df_numeric[col].median()
                df_numeric[col].fillna(median_val, inplace=True)
                st.success(f"✅ Filled `{col}` with median: {median_val:.2f}")

    # Tab 4: Fill with Mode
        with tab4:
            st.subheader("Fill Missing Values with Mode")
            all_cols = df_numeric.columns.tolist()
            col = st.selectbox("Select a column", all_cols, key="mode_col")
            if st.button("Fill with Mode"):
                mode_val = df_numeric[col].mode()
                if not mode_val.empty:
                    df_numeric[col].fillna(mode_val[0], inplace=True)
                    st.success(f"✅ Filled `{col}` with mode: {mode_val[0]}")
                else:
                    st.warning("⚠️ No mode available for this column.")

    # Tab 5: Remove Columns
        with tab5:
            st.subheader("Remove Unnecessary Columns")
            all_cols = df_numeric.columns.tolist()
            cols_to_remove = st.multiselect("Select columns to remove", all_cols)
            if st.button("Remove Selected Columns"):
                if cols_to_remove:
                    df_numeric.drop(columns=cols_to_remove, inplace=True)
                    st.success(f"🗑 Removed columns: {', '.join(cols_to_remove)}")
                else:
                    st.warning("⚠️ No columns selected for removal.")

        
    # Encoding
    elif page == 'Encoding':
        st.header("🔠 Encoding Categorical Data")
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if not categorical_cols:
            st.warning("No categorical columns to encode.")
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
                st.info("Select an encoding type to proceed.")
    # Modeling
    elif page == 'Modeling':
        st.header("🤖 Machine Learning Model Builder")

        # Step 1: Select target column
        target_col = st.selectbox("🎯 Select the target column", df.columns)

        # Step 2: Feature selection (exclude target)
        feature_cols = st.multiselect("📊 Select feature columns", [col for col in df.columns if col != target_col])

        # Step 3: Model choice
        model_type = st.radio("Choose model type", ["Logistic Regression", "Decision Tree", "Random Forest"])

        if st.button("Train Model"):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier

            # Drop rows with missing values in selected columns
            data = df[feature_cols + [target_col]].dropna()

            # Encode categorical variables
            for col in feature_cols:
                if data[col].dtype == 'object':
                    data[col] = LabelEncoder().fit_transform(data[col])

            # Encode target if it's categorical
            if data[target_col].dtype == 'object':
                data[target_col] = LabelEncoder().fit_transform(data[target_col])

            X = data[feature_cols]
            y = data[target_col]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model training
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            st.subheader("✅ Model Evaluation")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    # About page
    elif page == 'About':
        st.header("ℹ️ About This App")
        st.markdown("""
        **📊 Data Analysis App**

        This is an interactive data analysis tool built with **Streamlit**.  
        It allows you to:
        - 🗂 Upload and explore CSV datasets  
        - 🔍 Filter and select specific data  
        - 📉 Create powerful visualizations  
        - 🧹 Clean and preprocess your data  
        - 📊 Group and aggregate values  
        - 🔠 Encode categorical variables  

        ---

        **👨‍💻 Developer**: Jatin Kumar  ,Btech(elctronics and communication engineering),3nd year
                ##Guru Nanak Dev University,Amritsar##
        **🎨 Theme**: #FFEDF3 background, #0ABAB5 buttons, #56DFCF text  
        **🛠 Built With**: Python, Streamlit, Pandas, Matplotlib, Seaborn  
        **📫 Contact**:9888197119 [dhjatin4@gmail.com]

        Thank you for using the app!
        """)

else:
    st.warning("📁 Please upload a CSV file to proceed.")
