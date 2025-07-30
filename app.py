import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# Configure Streamlit page
st.set_page_config(page_title="Backlink Semantic Similarity & Authority Analysis", layout="wide")
st.title("🔗 Advanced Backlink Analysis: Semantic Similarity & Contextual Authority")

# Upload Excel file
uploaded_file = st.file_uploader(
    "📥 Upload Excel file (must contain 'Referring page URL', 'Target URL', 'UR', and 'External links')",
    type=["xlsx"]
)

# Model selection
model_choice = st.selectbox(
    "Choose a SentenceTransformer model:",
    [
        'all-MiniLM-L6-v2',
        'paraphrase-mpnet-base-v2',
        'all-mpnet-base-v2',
        'distiluse-base-multilingual-cased-v2'
    ]
)

if uploaded_file and model_choice:
    if 'processed_df' not in st.session_state:
        df = pd.read_excel(uploaded_file)
        
        # Check required columns
        required_columns = ['Referring page URL', 'Target URL', 'UR', 'External links']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Required columns: 'Referring page URL', 'Target URL', 'UR', 'External links'")
            st.stop()

        with st.spinner(f"🔄 Loading model: {model_choice}"):
            model = SentenceTransformer(model_choice)

        def tokenize_url(url):
            if pd.isna(url) or not isinstance(url, str):
                return []
            url = re.sub(r"https?://", "", url)
            tokens = re.split(r"[\/\.\-\?\=\_\&]+", url)
            return [t.lower() for t in tokens if t]

        def get_average_embedding(tokens):
            if not tokens:
                return np.zeros(model.get_sentence_embedding_dimension())
            embeddings = model.encode(tokens)
            return np.mean(embeddings, axis=0)

        def compute_token_based_similarity(ref_url, tgt_url):
            ref_tokens = tokenize_url(ref_url)
            tgt_tokens = tokenize_url(tgt_url)
            ref_vec = get_average_embedding(ref_tokens)
            tgt_vec = get_average_embedding(tgt_tokens)
            if np.all(ref_vec == 0) or np.all(tgt_vec == 0):
                return np.nan
            return 1 - cosine(ref_vec, tgt_vec)

        with st.spinner("⚙️ Calculating semantic similarities..."):
            cosine_similarities = []
            progress_bar = st.progress(0, text="Processing rows...")
            for i, row in df.iterrows():
                sim = compute_token_based_similarity(row['Referring page URL'], row['Target URL'])
                cosine_similarities.append(sim)
                progress = (i + 1) / len(df)
                progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")
            progress_bar.empty()
            
            # Add cosine similarity (keep as float for CAS calculation)
            df['Cosine Similarity'] = np.round(cosine_similarities, 3)

        # Convert columns to numeric for CAS calculation
        df['UR'] = pd.to_numeric(df['UR'], errors='coerce')
        df['External links'] = pd.to_numeric(df['External links'], errors='coerce')
        
        # Calculate Contextual Authority Score
        def calculate_contextual_authority_score(row):
            url_rating = row['UR']
            external_links = row['External links']
            cosine_sim = row['Cosine Similarity']
            
            if (pd.isna(url_rating) or pd.isna(external_links) or pd.isna(cosine_sim) or
                external_links == 0):
                return np.nan
            
            authority_ratio = url_rating / external_links
            contextual_authority_score = authority_ratio * cosine_sim
            return contextual_authority_score

        df['Contextual Authority Score'] = df.apply(calculate_contextual_authority_score, axis=1)
        df['Contextual Authority Score'] = df['Contextual Authority Score'].round(3)

        # Convert Domain rating to int if present
        if 'Domain rating' in df.columns:
            df['Domain rating'] = pd.to_numeric(df['Domain rating'], errors='coerce').fillna(0).astype(int)

        # Scale to 0-100 and convert to integers for display
        df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round().astype(int)
        df['Contextual Authority Score'] = (df['Contextual Authority Score'] * 100).round().astype(int)

        st.session_state.processed_df = df.copy()
        
        # Create Excel buffer
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.session_state.excel_buffer = buffer

    df = st.session_state.processed_df

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🏆 Top Performers", 
        "🌊 Sankey Diagram", 
        "📈 Scatter Analysis", 
        "📋 Data Table", 
        "📥 Download"
    ])

    with tab1:
        st.markdown("### 📊 Analysis Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Backlinks", len(df))
        with col2:
            avg_similarity = df['Cosine Similarity'].mean()
            st.metric("Avg. Cosine Similarity", f"{avg_similarity:.0f}")
        with col3:
            avg_cas = df['Contextual Authority Score'].mean()
            st.metric("Avg CAS", f"{avg_cas:.0f}")
        with col4:
            max_similarity = df['Cosine Similarity'].max()
            st.metric("Max Cosine Similarity", f"{max_similarity}")

        # Distribution charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Cosine Similarity Distribution
            df['Similarity Range'] = pd.cut(
                df['Cosine Similarity'],
                bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
                include_lowest=True
            )
            similarity_dist = df.groupby('Similarity Range')['Referring page URL'].count().reset_index()
            similarity_dist.rename(columns={'Referring page URL': 'Count'}, inplace=True)
            
            fig_sim_dist = px.bar(
                similarity_dist,
                x='Similarity Range',
                y='Count',
                title='Cosine Similarity Distribution',
                labels={'Count': 'Count of Backlinks', 'Similarity Range': 'Similarity Range'}
            )
            st.plotly_chart(fig_sim_dist, use_container_width=True)

        with col2:
            # CAS Distribution
            df['CAS Range'] = pd.cut(
                df['Contextual Authority Score'],
                bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
                include_lowest=True
            )
            cas_dist = df.groupby('CAS Range')['Referring page URL'].count().reset_index()
            cas_dist.rename(columns={'Referring page URL': 'Count'}, inplace=True)
            
            fig_cas_dist = px.bar(
                cas_dist,
                x='CAS Range',
                y='Count',
                title='Contextual Authority Score Distribution',
                labels={'Count': 'Count of Backlinks', 'CAS Range': 'CAS Range'},
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_cas_dist, use_container_width=True)

    with tab2:
        st.markdown("### 🏆 Top Performing Backlinks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 by Cosine Similarity
            top_10_similarity = (
                df[['Referring page URL', 'Target URL', 'Cosine Similarity']]
                .sort_values(by='Cosine Similarity', ascending=False)
                .dropna()
                .head(10)
            )
            
            fig_top_sim = px.bar(
                top_10_similarity,
                x='Cosine Similarity',
                y='Referring page URL',
                orientation='h',
                title='Top 10 by Cosine Similarity',
                custom_data=['Target URL']
            )
            fig_top_sim.update_traces(
                hovertemplate="<b>%{y}</b><br>Similarity: %{x}<br>Target: %{customdata[0]}"
            )
            st.plotly_chart(fig_top_sim, use_container_width=True)

        with col2:
            # Top 10 by CAS
            top_10_cas = (
                df[['Referring page URL', 'Target URL', 'Contextual Authority Score']]
                .sort_values(by='Contextual Authority Score', ascending=False)
                .dropna()
                .head(10)
            )
            
            fig_top_cas = px.bar(
                top_10_cas,
                x='Contextual Authority Score',
                y='Referring page URL',
                orientation='h',
                title='Top 10 by Contextual Authority Score',
                custom_data=['Target URL'],
                color_discrete_sequence=['#ff6b6b']
            )
            fig_top_cas.update_traces(
                hovertemplate="<b>%{y}</b><br>CAS: %{x}<br>Target: %{customdata[0]}"
            )
            st.plotly_chart(fig_top_cas, use_container_width=True)


# Prepare the top 25 dataframe
df_top25 = df.sort_values(by='Contextual Authority Score', ascending=False).head(25)

# Build edge list: [source_label, target_label, value]
edges = []
for _, row in df_top25.iterrows():
    source = row['Referring page URL']
    target = row['Target URL']
    cas = row['Contextual Authority Score']
    value = (row.get('Domain rating', row['UR']) + cas)
    edges.append([source, target, value])

# Create Sankey element
sankey = hv.Sankey(edges, kdims=['Source', 'Target'], vdims=['Value'])

# Optional styling: node labels, edge colors, etc.
sankey = sankey.opts(
    width=800, height=600,
    node_width=20, node_padding=10,
    edge_color='Value', cmap='Blues',
    labels='index', label_position='right',
    edge_alpha=0.8, title="Top 25 Backlinks by CAS"
)

with tab3:
    st.markdown("### 🌊 Sankey Diagram – Top 25 Backlinks by CAS")
    st.components.v1.html(hv.render(sankey, backend='bokeh').to_html(), height=650)


    with tab4:
        st.markdown("### 📈 Scatter Plot")
        
        # Domain Rating vs Cosine Similarity scatter plot
        if 'Domain rating' in df.columns:
            fig_scatter = px.scatter(
                df, 
                x='Cosine Similarity', 
                y='Domain rating',
                title='Domain Rating vs Cosine Similarity - Correlation Analysis',
                hover_data=['Referring page URL']
            )
        else:
            fig_scatter = px.scatter(
                df, 
                x='Cosine Similarity', 
                y='UR',
                title='URL Rating vs Cosine Similarity - Correlation Analysis',
                hover_data=['Referring page URL']
            )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab5:
        st.markdown("### 📋 Complete Data Table")
        
        # Display columns based on what's available
        display_columns = ['Referring page URL', 'Target URL', 'UR', 'External links', 
                          'Cosine Similarity', 'Contextual Authority Score']
        
        if 'Domain rating' in df.columns:
            display_columns.insert(1, 'Domain rating')
        
        # Filter to only show columns that exist
        available_columns = [col for col in display_columns if col in df.columns]
        
        st.dataframe(
            df[available_columns].sort_values(by='Contextual Authority Score', ascending=False),
            use_container_width=True, 
            height=600
        )

    with tab6:
        st.markdown("### 📥 Download Results")
        
        st.markdown("""
        **Download your processed data with:**
        - Original backlink data
        - Cosine Similarity scores
        - Contextual Authority Scores
        - All calculated metrics
        """)
        
        st.download_button(
            label="📥 Download Enhanced Results as Excel",
            data=st.session_state.excel_buffer,
            file_name="enhanced_backlink_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("👆 Please upload an Excel file and select a model to begin the analysis.")
    
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        **This tool analyzes backlink semantic similarity and contextual authority:**
        
        - **Cosine Similarity**: Measures semantic similarity between referring and target URLs
        - **Contextual Authority Score (CAS)**: a relevance-weighted backlink metric that combines link authority, link dilution, and topical similarity to assess the true SEO value of an external link.
        - **Formula**: CAS = (UR / External Links) × Cosine Similarity
        
        **Required columns in your Excel file:**
        - `Referring page URL`: The source URL of the backlink
        - `Target URL`: The destination URL
        - `UR`: URL Rating score
        - `External links`: Number of external links from the referring page
        - `Domain rating` (optional): Domain authority score
        """)
