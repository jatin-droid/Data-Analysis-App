"""
Machine Learning Modeling Page
"""

import streamlit as st
import pandas as pd
from utils.ml_helper import train_model, encode_data

def render(df: pd.DataFrame):
    """Render the modeling page"""
    st.header("🤖 MACHINE LEARNING MODULE")
    
    st.info("💡 Build and train machine learning models directly on your data!")
    
    # Step 1: Select target column
    st.subheader("🎯 Step 1: Select Target Variable")
    target_col = st.selectbox(
        "Select the target column (what you want to predict)",
        df.columns.tolist(),
        key="target_col"
    )
    
    # Show target distribution
    if target_col:
        st.write("**Target Distribution:**")
        target_dist = df[target_col].value_counts()
        st.bar_chart(target_dist)
        
        unique_targets = df[target_col].nunique()
        st.metric("Unique Target Values", unique_targets)
        
        if unique_targets > 20:
            st.warning("⚠️ Target has many unique values. Consider if this is a regression task.")
    
    # Step 2: Feature selection
    st.subheader("📊 Step 2: Select Features")
    available_features = [col for col in df.columns if col != target_col]
    feature_cols = st.multiselect(
        "Select feature columns (independent variables)",
        available_features,
        default=available_features[:min(5, len(available_features))],
        key="feature_cols"
    )
    
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column")
        return
    
    # Show feature info
    st.write(f"**Selected {len(feature_cols)} features:**")
    feature_types = df[feature_cols].dtypes.value_counts()
    st.write(feature_types)
    
    # Step 3: Model selection
    st.subheader("🔧 Step 3: Choose Model Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Select model",
            ["Logistic Regression", "Decision Tree", "Random Forest"],
            key="model_type"
        )
    
    with col2:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="test_size"
        ) / 100
    
    # Model descriptions
    with st.expander("ℹ️ Learn about models"):
        st.markdown("""
        **Logistic Regression:**
        - Best for: Binary classification problems
        - Fast and interpretable
        - Works well with linearly separable data
        
        **Decision Tree:**
        - Best for: Non-linear relationships
        - Easy to interpret and visualize
        - Can overfit if not pruned
        
        **Random Forest:**
        - Best for: Complex patterns and high accuracy
        - Ensemble of decision trees
        - More robust and less prone to overfitting
        """)
    
    # Train button
    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training model... ⚙️"):
            try:
                # Prepare data
                data = df[feature_cols + [target_col]].copy()
                
                # Drop rows with missing values
                initial_rows = len(data)
                data = data.dropna()
                dropped_rows = initial_rows - len(data)
                
                if dropped_rows > 0:
                    st.warning(f"⚠️ Dropped {dropped_rows} rows with missing values")
                
                if len(data) < 10:
                    st.error("❌ Not enough data after removing missing values. Need at least 10 rows.")
                    return
                
                # Encode data
                data_encoded, encoders = encode_data(data, feature_cols, target_col)
                
                # Train model
                results = train_model(
                    data_encoded,
                    feature_cols,
                    target_col,
                    model_type,
                    test_size
                )
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                # Metrics
                st.subheader("📊 Model Performance")
                
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Accuracy", f"{results['accuracy']:.2%}")
                with metric_cols[1]:
                    st.metric("Training Samples", len(results['y_train']))
                with metric_cols[2]:
                    st.metric("Test Samples", len(results['y_test']))
                
                # Classification report
                st.subheader("📋 Detailed Classification Report")
                st.text(results['classification_report'])
                
                # Feature importance (for tree-based models)
                if model_type in ["Decision Tree", "Random Forest"]:
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': results['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
                    st.dataframe(importance_df)
                
                # Confusion Matrix
                st.subheader("🔲 Confusion Matrix")
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                cm = confusion_matrix(results['y_test'], results['y_pred'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='cool', ax=ax, 
                           cbar_kws={'label': 'Count'})
                ax.set_title("Confusion Matrix", color='#00ffff', fontsize=16)
                ax.set_xlabel("Predicted", color='#00ffff')
                ax.set_ylabel("Actual", color='#00ffff')
                ax.tick_params(colors='#00ffff')
                fig.patch.set_facecolor('#0a0e27')
                st.pyplot(fig)
                
                # Predictions
                st.subheader("🔮 Sample Predictions")
                predictions_df = pd.DataFrame({
                    'Actual': results['y_test'].values[:10],
                    'Predicted': results['y_pred'][:10]
                })
                st.dataframe(predictions_df)
                
                # Model info
                with st.expander("ℹ️ Model Information"):
                    st.write("**Model Type:**", model_type)
                    st.write("**Features Used:**", feature_cols)
                    st.write("**Target Variable:**", target_col)
                    st.write("**Training Size:**", f"{(1-test_size)*100:.0f}%")
                    st.write("**Test Size:**", f"{test_size*100:.0f}%")
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.write("**Debug Info:**")
                st.write(f"- Data shape: {data.shape}")
                st.write(f"- Features: {feature_cols}")
                st.write(f"- Target: {target_col}")