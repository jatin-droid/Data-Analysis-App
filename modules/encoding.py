"""
Encoding Page
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def render(df: pd.DataFrame):
    """Render the encoding page"""
    st.header("🔠 ENCODING MODULE")
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    if not categorical_cols:
        st.warning("⚠️ No categorical columns to encode.")
        st.info("💡 Categorical columns are text-based columns that contain categories or labels.")
        return
    
    # Initialize encoded dataframe in session state
    if 'df_encoded' not in st.session_state:
        st.session_state.df_encoded = df.copy()
    
    df_encoded = st.session_state.df_encoded
    
    st.info(f"📊 Found {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
    
    # Select column to encode
    col = st.selectbox("Select a categorical column to encode", categorical_cols, key="encode_col")
    
    # Show unique values
    unique_vals = df[col].nunique()
    st.metric("Unique Values", unique_vals)
    
    if unique_vals <= 20:
        st.write("**Sample values:**", df[col].unique()[:10].tolist())
    else:
        st.write(f"**Sample values (showing 10/{unique_vals}):**", df[col].unique()[:10].tolist())
    
    # Encoding type selection
    encoding_type = st.selectbox(
        "Select encoding type",
        ["One-Hot Encoding", "Label Encoding"],
        key="encoding_type"
    )
    
    # Explanation
    with st.expander("ℹ️ Learn about encoding types"):
        st.markdown("""
        **One-Hot Encoding:**
        - Creates binary columns for each unique value
        - Best for: Nominal data (no inherent order)
        - Example: Color (Red, Blue, Green) → Red_0/1, Blue_0/1, Green_0/1
        - Warning: Can create many columns if there are many unique values
        
        **Label Encoding:**
        - Converts categories to numbers (0, 1, 2, ...)
        - Best for: Ordinal data (has natural order)
        - Example: Size (Small, Medium, Large) → 0, 1, 2
        - Warning: May imply order when there isn't one
        """)
    
    # Perform encoding
    if encoding_type == "One-Hot Encoding":
        st.subheader("🔄 One-Hot Encoding")
        
        drop_first = st.checkbox(
            "Drop first category to avoid multicollinearity",
            value=True,
            help="Recommended for machine learning to avoid the dummy variable trap"
        )
        
        if st.button("🚀 Apply One-Hot Encoding", key="apply_onehot"):
            try:
                # Perform one-hot encoding
                encoded_df = pd.get_dummies(
                    df_encoded,
                    columns=[col],
                    drop_first=drop_first,
                    prefix=col
                )
                
                st.session_state.df_encoded = encoded_df
                
                new_cols = [c for c in encoded_df.columns if c.startswith(col + '_')]
                st.success(f"✅ Created {len(new_cols)} new columns!")
                st.write("**New columns:**", new_cols)
                
                st.subheader("📊 Encoded Data Preview")
                st.dataframe(encoded_df.head())
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif encoding_type == "Label Encoding":
        st.subheader("🔢 Label Encoding")
        
        if st.button("🚀 Apply Label Encoding", key="apply_label"):
            try:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                st.session_state.df_encoded = df_encoded
                
                # Show mapping
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                
                st.success(f"✅ Encoded {col} to numeric labels!")
                
                st.subheader("🗺️ Encoding Mapping")
                mapping_df = pd.DataFrame(list(mapping.items()), columns=['Original Value', 'Encoded Value'])
                st.dataframe(mapping_df)
                
                st.subheader("📊 Encoded Data Preview")
                st.dataframe(df_encoded.head())
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display current encoded dataframe
    st.markdown("---")
    st.subheader("📋 Current Encoded Data")
    st.dataframe(df_encoded.head(10))
    
    col_count1, col_count2 = st.columns(2)
    with col_count1:
        st.metric("Original Columns", df.shape[1])
    with col_count2:
        st.metric("Current Columns", df_encoded.shape[1])
    
    # Download encoded data
    csv = df_encoded.to_csv(index=False)
    st.download_button(
        label="📥 Download Encoded Data",
        data=csv,
        file_name="encoded_data.csv",
        mime="text/csv"
    )
    
    # Reset button
    if st.button("🔄 Reset to Original Data", key="reset_encoding"):
        st.session_state.df_encoded = df.copy()
        st.success("✅ Reset to original data!")
        st.rerun()