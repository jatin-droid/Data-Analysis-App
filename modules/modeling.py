"""
Machine Learning Modeling Page
"""

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def render(df: pd.DataFrame):
    """Render the modeling page"""
    st.header("🤖 MACHINE LEARNING MODULE")
    st.info("💡 Build and train machine learning models directly on your data!")

    # ---------------- STEP 1: TARGET SELECTION ---------------- #
    st.subheader("🎯 Step 1: Select Target Variable")
    target_col = st.selectbox(
        "Select the target column (what you want to predict)",
        df.columns.tolist(),
        key="target_col",
    )

    if not target_col:
        st.warning("⚠️ Please select a target column to proceed.")
        return

    # Show target distribution
    target_series = df[target_col].dropna()
    if not target_series.empty:
        st.write("**Target Distribution:**")
        st.bar_chart(target_series.value_counts())
        unique_targets = target_series.nunique()
        st.metric("Unique Target Values", unique_targets)
    else:
        st.warning("⚠️ Target column is empty after dropping NaNs.")
        return

    # ---------------- STEP 2: FEATURE SELECTION ---------------- #
    st.subheader("📊 Step 2: Select Features")
    available_features = [col for col in df.columns if col != target_col]
    feature_cols = st.multiselect(
        "Select feature columns (independent variables)",
        available_features,
        default=available_features[: min(5, len(available_features))],
        key="feature_cols",
    )

    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column.")
        return

    st.write(f"**Selected {len(feature_cols)} features:**")
    feature_types = df[feature_cols].dtypes.value_counts()
    st.write(feature_types)

    # ---------------- STEP 3: PROBLEM TYPE & MODEL SELECTION ---------------- #
    st.subheader("🔧 Step 3: Choose Problem Type and Model")

    col_prob, col_model = st.columns(2)

    with col_prob:
        problem_type = st.radio(
            "Select Problem Type",
            ["Classification", "Regression"],
            key="problem_type",
        )

    with col_model:
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
                    "XGBoost Classifier",
                ],
                key="model_type",
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
                    "XGBoost Regressor",
                ],
                key="model_type",
            )

    # Test size
    test_size = (
        st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="test_size",
        )
        / 100
    )

    # Model descriptions
    with st.expander("ℹ️ Learn about models"):
        st.markdown(
            """
        **Logistic Regression:**
        - Best for: Binary / multiclass classification
        - Fast and interpretable
        
        **Decision Tree:**
        - Handles non-linear relationships
        - Easy to visualize
        
        **Random Forest:**
        - Ensemble of decision trees
        - Robust, good generalization
        
        **KNN & SVM:**
        - KNN: Distance-based method
        - SVM: Works well with clear margins
        
        **Naive Bayes:**
        - Simple and fast for text/categorical
        
        **Gradient Boosting / XGBoost:**
        - Powerful ensemble methods
        - Great for tabular data
        
        **Linear / Ridge / Lasso Regression:**
        - Linear models for regression
        - Ridge/Lasso add regularization
        
        **Tree / Forest / GB / XGB Regressors:**
        - Capture non-linear relationships in regression problems
        """
        )

    # ---------------- STEP 4: TRAIN MODEL ---------------- #
    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training model... ⚙️"):
            try:
                # Prepare data
                data = df[feature_cols + [target_col]].copy()
                initial_rows = len(data)
                data = data.dropna()
                dropped_rows = initial_rows - len(data)

                if dropped_rows > 0:
                    st.warning(f"⚠️ Dropped {dropped_rows} rows with missing values")

                if len(data) < 10:
                    st.error(
                        "❌ Not enough data after removing missing values. Need at least 10 rows."
                    )
                    return

                # ---------------- ENCODING ---------------- #
                label_encoders = {}
                # Encode categorical features
                for col in feature_cols:
                    if data[col].dtype == "object":
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        label_encoders[col] = le

                # Encode target if classification and object
                target_encoder = None
                if problem_type == "Classification" and data[target_col].dtype == "object":
                    target_encoder = LabelEncoder()
                    data[target_col] = target_encoder.fit_transform(
                        data[target_col].astype(str)
                    )

                X = data[feature_cols]
                y = data[target_col]

                # ---------------- SCALING FOR SOME MODELS ---------------- #
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

                # ---------------- TRAIN/TEST SPLIT ---------------- #
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # ---------------- MODEL SELECTION ---------------- #
                model = None
                is_xgb = False

                if model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_type == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_type == "Random Forest Classifier":
                    model = RandomForestClassifier(random_state=42)
                elif model_type == "K-Nearest Neighbors (KNN)":
                    model = KNeighborsClassifier()
                elif model_type == "Support Vector Machine (SVM)":
                    model = SVC(random_state=42)
                elif model_type == "Naive Bayes":
                    model = GaussianNB()
                elif model_type == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(random_state=42)
                elif model_type == "XGBoost Classifier":
                    try:
                        import xgboost as xgb

                        model = xgb.XGBClassifier(
                            random_state=42, eval_metric="logloss"
                        )
                        is_xgb = True
                    except ImportError:
                        st.error(
                            "❌ XGBoost is not installed on this environment. Falling back to Gradient Boosting Classifier."
                        )
                        model = GradientBoostingClassifier(random_state=42)

                elif model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge(random_state=42)
                elif model_type == "Lasso Regression":
                    model = Lasso(random_state=42)
                elif model_type == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=42)
                elif model_type == "Random Forest Regressor":
                    model = RandomForestRegressor(random_state=42)
                elif model_type == "Support Vector Regressor (SVR)":
                    model = SVR()
                elif model_type == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor(random_state=42)
                elif model_type == "XGBoost Regressor":
                    try:
                        import xgboost as xgb

                        model = xgb.XGBRegressor(random_state=42)
                        is_xgb = True
                    except ImportError:
                        st.error(
                            "❌ XGBoost is not installed on this environment. Falling back to Gradient Boosting Regressor."
                        )
                        model = GradientBoostingRegressor(random_state=42)

                if model is None:
                    st.error("❌ No valid model selected.")
                    return

                # ---------------- TRAIN ---------------- #
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.success("✅ Model trained successfully!")

                # ---------------- METRICS & VISUALS ---------------- #
                if problem_type == "Classification":
                    # Metrics
                    st.subheader("📊 Model Performance (Classification)")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    acc = accuracy_score(y_test, y_pred)
                    with col_m1:
                        st.metric("🎯 Accuracy", f"{acc:.4f}")
                    with col_m2:
                        st.metric("📈 Training Samples", len(X_train))
                    with col_m3:
                        st.metric("🧪 Test Samples", len(X_test))

                    # Classification report
                    st.subheader("📋 Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    # Confusion matrix
                    st.subheader("🔲 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="cool",
                        ax=ax,
                        cbar_kws={"label": "Count"},
                    )
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                else:
                    # Regression metrics
                    st.subheader("📊 Model Performance (Regression)")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    with col_m1:
                        st.metric("📉 MSE", f"{mse:.4f}")
                    with col_m2:
                        st.metric("📊 MAE", f"{mae:.4f}")
                    with col_m3:
                        st.metric("📈 R² Score", f"{r2:.4f}")

                    # Predictions vs Actual
                    st.subheader("📈 Predictions vs Actual Values")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(y_test, y_pred, alpha=0.6, color="#FFB6C1", edgecolors="#8a2be2")
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)

                # Feature importance if available
                if hasattr(model, "feature_importances_"):
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame(
                        {
                            "Feature": feature_cols,
                            "Importance": model.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(
                        importance_df["Feature"],
                        importance_df["Importance"],
                        color="#F4C2C2",
                        edgecolor="#8a2be2",
                    )
                    ax.set_xlabel("Importance")
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                    st.dataframe(importance_df)

                # Sample predictions
                st.subheader("🔮 Sample Predictions")
                # align sizes for display
                y_test_arr = np.array(y_test)
                y_pred_arr = np.array(y_pred)
                n_show = min(10, len(y_test_arr))
                predictions_df = pd.DataFrame(
                    {"Actual": y_test_arr[:n_show], "Predicted": y_pred_arr[:n_show]}
                )
                st.dataframe(predictions_df)

                # Model info
                with st.expander("ℹ️ Model Information"):
                    st.write("**Problem Type:**", problem_type)
                    st.write("**Model Type:**", model_type)
                    st.write("**Features Used:**", feature_cols)
                    st.write("**Target Variable:**", target_col)
                    st.write("**Training Size:**", f"{(1 - test_size) * 100:.0f}%")
                    st.write("**Test Size:**", f"{test_size * 100:.0f}%")

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.write("**Debug Info:**")
                st.write(f"- Features: {feature_cols}")
                st.write(f"- Target: {target_col}")
                st.write(f"- Data shape: {df[feature_cols + [target_col]].shape}")
