"""
Data Visualization Page
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render(df: pd.DataFrame):
    """Render the visualize page"""
    st.header("📈 VISUALIZATION MODULE")
    
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    
    if not numeric_columns:
        st.warning("⚠️ No numeric columns to plot.")
        return
    
    # Column selection
    col1_select, col2_select = st.columns(2)
    
    with col1_select:
        col1 = st.selectbox("Select column 1", numeric_columns, key="viz_col1")
    
    with col2_select:
        col2 = st.selectbox("Select column 2 (for scatter/heatmap)", numeric_columns, key="viz_col2")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Bar Chart",
        "📊 Histogram",
        "📉 Line Plot",
        "🔵 Scatter Plot",
        "🔥 Heatmap",
        "📦 Box Plot",
        "🥧 Pie Chart"
    ])
    
    with tab1:
        st.subheader("📊 Bar Chart")
        value_counts = df[col1].value_counts()
        st.bar_chart(value_counts)
        st.info(f"Showing distribution of {col1}")
    
    with tab2:
        st.subheader("📊 Histogram")
        bins = st.slider("Number of bins", 5, 50, 20, key="hist_bins")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[col1].dropna(), bins=bins, color='#00ffff', edgecolor='#8a2be2', alpha=0.7)
        ax.set_title(f"Histogram of {col1}", color='#00ffff', fontsize=16)
        ax.set_xlabel(col1, color='#00ffff')
        ax.set_ylabel("Frequency", color='#00ffff')
        ax.set_facecolor('#0a0e27')
        ax.tick_params(colors='#00ffff')
        fig.patch.set_facecolor('#0a0e27')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("📉 Line Plot")
        st.line_chart(df[col1])
        st.info(f"Showing trend of {col1}")
    
    with tab4:
        st.subheader("🔵 Scatter Plot")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[col1], df[col2], alpha=0.6, color='#00ffff', edgecolors='#8a2be2', s=50)
        ax.set_xlabel(col1, color='#00ffff', fontsize=12)
        ax.set_ylabel(col2, color='#00ffff', fontsize=12)
        ax.set_title(f"{col1} vs {col2}", color='#00ffff', fontsize=16)
        ax.set_facecolor('#0a0e27')
        ax.tick_params(colors='#00ffff')
        ax.grid(True, alpha=0.2, color='#00ffff')
        fig.patch.set_facecolor('#0a0e27')
        st.pyplot(fig)
    
    with tab5:
        st.subheader("🔥 Heatmap (Correlation Matrix)")
        numeric_df = df.select_dtypes(include='number')
        
        if numeric_df.shape[1] < 2:
            st.warning("⚠️ Not enough numeric columns to display a correlation heatmap.")
        else:
            corr = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="cool", ax=ax, 
                       cbar_kws={'label': 'Correlation'}, 
                       linewidths=0.5, linecolor='#8a2be2')
            ax.set_title("Correlation Heatmap", color='#00ffff', fontsize=16, pad=20)
            ax.tick_params(colors='#00ffff')
            fig.patch.set_facecolor('#0a0e27')
            st.pyplot(fig)
            
            # Show strongest correlations
            st.subheader("🔝 Strongest Correlations")
            corr_pairs = corr.unstack()
            corr_pairs = corr_pairs[corr_pairs != 1.0]
            corr_pairs = corr_pairs.sort_values(ascending=False).drop_duplicates()
            st.dataframe(corr_pairs.head(10))
    
    with tab6:
        st.subheader("📦 Box Plot")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(df[col1].dropna(), vert=True, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#00ffff')
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:
            whisker.set(color='#8a2be2', linewidth=1.5)
        
        for cap in bp['caps']:
            cap.set(color='#8a2be2', linewidth=1.5)
        
        for median in bp['medians']:
            median.set(color='#8a2be2', linewidth=2)
        
        ax.set_ylabel(col1, color='#00ffff', fontsize=12)
        ax.set_title(f"Boxplot of {col1}", color='#00ffff', fontsize=16)
        ax.set_facecolor('#0a0e27')
        ax.tick_params(colors='#00ffff')
        ax.grid(True, alpha=0.2, color='#00ffff', axis='y')
        fig.patch.set_facecolor('#0a0e27')
        st.pyplot(fig)
        
        # Display statistics
        col_stats = st.columns(4)
        with col_stats[0]:
            st.metric("Min", f"{df[col1].min():.2f}")
        with col_stats[1]:
            st.metric("Q1", f"{df[col1].quantile(0.25):.2f}")
        with col_stats[2]:
            st.metric("Median", f"{df[col1].median():.2f}")
        with col_stats[3]:
            st.metric("Max", f"{df[col1].max():.2f}")
    
    with tab7:
        st.subheader("🥧 Pie Chart (Categorical Distribution)")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        
        if not cat_cols:
            st.warning("⚠️ No categorical columns available for pie chart.")
        else:
            pie_col = st.selectbox("Select a categorical column", cat_cols, key="pie_col")
            
            if pie_col:
                pie_data = df[pie_col].value_counts()
                
                # Limit to top 10 categories
                if len(pie_data) > 10:
                    st.info(f"Showing top 10 categories out of {len(pie_data)}")
                    pie_data = pie_data.head(10)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['#00ffff', '#8a2be2', '#ff00ff', '#00ff00', '#ffff00', 
                         '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
                
                wedges, texts, autotexts = ax.pie(
                    pie_data,
                    labels=pie_data.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(pie_data)],
                    textprops={'color': '#00ffff', 'fontsize': 10}
                )
                
                for autotext in autotexts:
                    autotext.set_color('#000')
                    autotext.set_fontweight('bold')
                
                ax.axis("equal")
                ax.set_title(f"Distribution of {pie_col}", color='#00ffff', fontsize=16, pad=20)
                fig.patch.set_facecolor('#0a0e27')
                st.pyplot(fig)
                
                # Show value counts
                st.subheader("📊 Value Counts")
                st.dataframe(pie_data)