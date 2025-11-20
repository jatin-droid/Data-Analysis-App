"""
Grouping and Aggregation Page
"""

import streamlit as st
import pandas as pd

def perform_aggregation(df, group_cols, agg_func):
    """Perform aggregation on grouped data"""
    if agg_func == "Count":
        return df.groupby(group_cols).size().reset_index(name='Count')
    elif agg_func == "Sum":
        return df.groupby(group_cols).sum(numeric_only=True).reset_index()
    elif agg_func == "Mean":
        return df.groupby(group_cols).mean(numeric_only=True).reset_index()
    elif agg_func == "Median":
        return df.groupby(group_cols).median(numeric_only=True).reset_index()
    elif agg_func == "Max":
        return df.groupby(group_cols).max(numeric_only=True).reset_index()
    elif agg_func == "Min":
        return df.groupby(group_cols).min(numeric_only=True).reset_index()
    else:
        return df

def render(df: pd.DataFrame):
    """Render the grouping page"""
    st.header("📊 GROUPING & AGGREGATION MODULE")
    
    # Single column grouping
    st.subheader("🔹 Group by Single Column")
    
    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox("Select a column to group by", df.columns.tolist(), key="single_group")
    
    with col2:
        agg_func = st.selectbox(
            "Select aggregation function",
            ["Count", "Sum", "Mean", "Median", "Max", "Min"],
            key="single_agg"
        )
    
    if st.button("🚀 Group and Aggregate", key="single_btn"):
        try:
            grouped_df = perform_aggregation(df, group_col, agg_func)
            
            st.success(f"✅ Grouped by {group_col} using {agg_func}")
            st.dataframe(grouped_df)
            
            # Download button
            csv = grouped_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Grouped Data",
                data=csv,
                file_name=f"grouped_by_{group_col}.csv",
                mime="text/csv",
                key="single_download"
            )
            
            # Visualize if count
            if agg_func == "Count":
                st.subheader("📊 Visualization")
                st.bar_chart(grouped_df.set_index(group_col)['Count'])
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Multiple columns grouping
    st.subheader("🔹 Group by Multiple Columns")
    
    group_cols = st.multiselect(
        "Select columns to group by",
        df.columns.tolist(),
        key="multi_group"
    )
    
    if group_cols:
        agg_func_multi = st.selectbox(
            "Select aggregation function for multiple columns",
            ["Count", "Sum", "Mean", "Median", "Max", "Min"],
            key="multi_agg"
        )
        
        if st.button("🚀 Group by Multiple Columns", key="multi_btn"):
            try:
                grouped_df = perform_aggregation(df, group_cols, agg_func_multi)
                
                st.success(f"✅ Grouped by {', '.join(group_cols)} using {agg_func_multi}")
                st.dataframe(grouped_df)
                
                # Show group sizes
                if agg_func_multi == "Count":
                    total_groups = len(grouped_df)
                    st.info(f"📊 Total unique groups: {total_groups}")
                
                # Download button
                csv = grouped_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Grouped Data",
                    data=csv,
                    file_name=f"grouped_by_multiple.csv",
                    mime="text/csv",
                    key="multi_download"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("🎯 Select at least one column to group by")
    
    st.markdown("---")
    
    # Advanced grouping
    st.subheader("🔹 Advanced: Custom Aggregations")
    
    with st.expander("💡 Multiple Aggregations on Same Groups"):
        st.write("Apply different aggregation functions to different columns")
        
        if group_cols:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            
            if numeric_cols:
                agg_dict = {}
                
                for col in numeric_cols:
                    selected_aggs = st.multiselect(
                        f"Aggregations for {col}",
                        ["sum", "mean", "median", "min", "max", "std", "count"],
                        key=f"agg_{col}"
                    )
                    if selected_aggs:
                        agg_dict[col] = selected_aggs
                
                if st.button("🚀 Apply Custom Aggregations", key="custom_agg") and agg_dict:
                    try:
                        grouped_df = df.groupby(group_cols).agg(agg_dict).reset_index()
                        grouped_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                             for col in grouped_df.columns.values]
                        
                        st.success("✅ Custom aggregations applied!")
                        st.dataframe(grouped_df)
                        
                        # Download button
                        csv = grouped_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Custom Aggregated Data",
                            data=csv,
                            file_name="custom_aggregation.csv",
                            mime="text/csv",
                            key="custom_download"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ No numeric columns available for aggregation")
        else:
            st.info("🎯 First select grouping columns in the section above")