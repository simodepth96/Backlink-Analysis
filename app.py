import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(page_title="Backlink Analysis", layout="wide")
st.title("Backlink Analysis")

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

            df['Cosine Similarity'] = np.round(cosine_similarities, 3)

        df['UR'] = pd.to_numeric(df['UR'], errors='coerce')
        df['External links'] = pd.to_numeric(df['External links'], errors='coerce')

        def calculate_contextual_authority_score(row):
            url_rating = row['UR']
            external_links = row['External links']
            cosine_sim = row['Cosine Similarity']
            if (pd.isna(url_rating) or pd.isna(external_links) or pd.isna(cosine_sim) or external_links == 0):
                return np.nan
            authority_ratio = url_rating / external_links
            contextual_authority_score = authority_ratio * cosine_sim
            return contextual_authority_score

        df['Contextual Authority Score'] = df.apply(calculate_contextual_authority_score, axis=1)
        df['Contextual Authority Score'] = df['Contextual Authority Score'].round(3)

        if 'Domain rating' in df.columns:
            df['Domain rating'] = pd.to_numeric(df['Domain rating'], errors='coerce').fillna(0).astype(int)

        df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round().astype(int)
        df['Contextual Authority Score'] = (df['Contextual Authority Score'] * 100).round().astype(int)

        st.session_state.processed_df = df.copy()
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.session_state.excel_buffer = buffer

    df = st.session_state.processed_df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏆 Top Performers",
        "🔗 Backlink Flow",
        "📋 Data Table",
        "📥 Download"
    ])

    with tab1:
        st.markdown("### 📊 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Backlinks", len(df))
        with col2:
            st.metric("Avg. Cosine Similarity", f"{df['Cosine Similarity'].mean():.0f}")
        with col3:
            st.metric("Avg CAS", f"{df['Contextual Authority Score'].mean():.0f}")
        with col4:
            st.metric("Max Cosine Similarity", f"{df['Cosine Similarity'].max()}")

        col1, col2 = st.columns(2)
        with col1:
            df['Similarity Range'] = pd.cut(
                df['Cosine Similarity'],
                bins=[0,10,20,30,40,50,60,70,80,90,100],
                labels=[f'{i}-{i+10}' for i in range(0,100,10)],
                include_lowest=True
            )
            sim_dist = df.groupby('Similarity Range')['Referring page URL'].count().reset_index().rename(columns={'Referring page URL': 'Count'})
            st.plotly_chart(px.bar(sim_dist, x='Similarity Range', y='Count', title='Cosine Similarity Distribution'), use_container_width=True)
            st.caption("💡 **Cosine Similarity** measures how closely the content of referring pages matches your target page's content. Higher scores indicate more topically relevant backlinks.")

        with col2:
            df['CAS Range'] = pd.cut(
                df['Contextual Authority Score'],
                bins=[0,10,20,30,40,50,60,70,80,90,100],
                labels=[f'{i}-{i+10}' for i in range(0,100,10)],
                include_lowest=True
            )
            cas_dist = df.groupby('CAS Range')['Referring page URL'].count().reset_index().rename(columns={'Referring page URL': 'Count'})
            st.plotly_chart(px.bar(cas_dist, x='CAS Range', y='Count', title='Contextual Authority Score Distribution',color_discrete_sequence=['#ff6b6b']), use_container_width=True)
            st.caption("📈 **Contextual Authority Score (CAS)** combines link authority with topical relevance. It factors in the page's URL Rating, external links, and semantic similarity for a comprehensive quality score. The higher the score, the more authoritative and topically relevant a link is.")

    with tab2:
        st.markdown("### 🏆 Top Performing Backlinks")
        
        top_sim = df.sort_values(by='Cosine Similarity', ascending=False).head(10)
        st.plotly_chart(px.bar(top_sim, x='Cosine Similarity', y='Referring page URL', orientation='h', title='Top 10 by Cosine Similarity', hover_data=['Target URL']), use_container_width=True)
    
    
        top_cas = df.sort_values(by='Contextual Authority Score', ascending=False).head(10)
        st.plotly_chart(px.bar(top_cas, x='Contextual Authority Score', y='Referring page URL', orientation='h', title='Top 10 by Contextual Authority Score', color_discrete_sequence=['#ff6b6b'], hover_data=['Target URL']), use_container_width=True)
    

    with tab3:
        st.markdown("### Referring Domains Relevance to your Target Domain")
        
        # Create Sankey diagram data
        # Limit to top 15 backlinks for readability
        top_backlinks = df.nlargest(15, 'Contextual Authority Score')
        
        # Extract domain names for cleaner visualization
        def extract_domain(url):
            if pd.isna(url):
                return "Unknown"
            # Remove protocol and www
            domain = re.sub(r'https?://(www\.)?', '', str(url))
            # Get the domain part (before first slash)
            domain = domain.split('/')[0]
            # Truncate long domains
            if len(domain) > 30:
                domain = domain[:27] + "..."
            return domain
        
        top_backlinks['Source Domain'] = top_backlinks['Referring page URL'].apply(extract_domain)
        top_backlinks['Target Domain'] = top_backlinks['Target URL'].apply(extract_domain)
        
        # Create unique node lists
        source_nodes = top_backlinks['Source Domain'].unique().tolist()
        target_nodes = top_backlinks['Target Domain'].unique().tolist()
        all_nodes = source_nodes + [node for node in target_nodes if node not in source_nodes]
        
        # Create node indices
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        source_indices = [node_indices[domain] for domain in top_backlinks['Source Domain']]
        target_indices = [node_indices[domain] for domain in top_backlinks['Target Domain']]
        values = top_backlinks['Contextual Authority Score'].tolist()
        
        # Create the Sankey diagram
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=["lightblue" if node in source_nodes else "lightcoral" for node in all_nodes]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color="rgba(255, 107, 107, 0.4)"
            )
        )])
        
        fig_sankey.update_layout(
            title_text="Backlink Flow: Referring Domains → Target Domains<br><sub>Flow thickness represents Contextual Authority Score</sub>",
            font_size=10,
            height=600
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.caption("🌊 **Link Flow Diagram** shows the relationship between referring domains and target domains. The thickness of each flow represents the Contextual Authority Score, helping you visualize which domains are sending the most valuable backlinks.")

    with tab4:
        st.markdown("### 📋 Complete Data Table")
        cols = ['Referring page URL', 'Target URL', 'UR', 'External links', 'Cosine Similarity', 'Contextual Authority Score']
        if 'Domain rating' in df.columns:
            cols.insert(1, 'Domain rating')
        st.dataframe(df[cols].sort_values(by='Contextual Authority Score', ascending=False), use_container_width=True, height=600)

    with tab5:
        st.markdown("### 📥 Download Results")
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
        **It is highly recommended that you upload an XLSX file with backlinks placed in the content area of the target site.**

        - **Cosine Similarity**: Measures semantic similarity between referring and target URLs
        - **Contextual Authority Score (CAS)**: A relevance-weighted backlink metric that combines link authority, link dilution, and topical similarity
        - **Formula**: CAS = (UR / External Links) × Cosine Similarity

        **Required columns:**
        - `Referring page URL`
        - `Target URL`
        - `UR`
        - `External links`
        - `Domain rating` (optional)
        """)
