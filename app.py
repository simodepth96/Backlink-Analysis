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
            st.plotly_chart(px.bar(sim_dist, x='Similarity Range', y='Count', title='🔍 Cosine Similarity Distribution'), use_container_width=True)

        with col2:
            df['CAS Range'] = pd.cut(
                df['Contextual Authority Score'],
                bins=[0,10,20,30,40,50,60,70,80,90,100],
                labels=[f'{i}-{i+10}' for i in range(0,100,10)],
                include_lowest=True
            )
            cas_dist = df.groupby('CAS Range')['Referring page URL'].count().reset_index().rename(columns={'Referring page URL': 'Count'})
            st.plotly_chart(px.bar(cas_dist, x='CAS Range', y='Count', title='⚡ Contextual Authority Score Distribution'), use_container_width=True)

    with tab2:
        st.markdown("### 🏆 Top Performing Backlinks")
        col1, col2 = st.columns(2)
        with col1:
            top_sim = df.sort_values(by='Cosine Similarity', ascending=False).head(10)
            st.plotly_chart(px.bar(top_sim, x='Cosine Similarity', y='Referring page URL', orientation='h', title='🎯 Top 10 by Cosine Similarity'), use_container_width=True)
        with col2:
            top_cas = df.sort_values(by='Contextual Authority Score', ascending=False).head(10)
            st.plotly_chart(px.bar(top_cas, x='Contextual Authority Score', y='Referring page URL', orientation='h', title='⚡ Top 10 by Contextual Authority Score'), use_container_width=True)

    with tab3:
        st.markdown("### 🌊 Sankey Diagram - Top 25 Backlinks by CAS")
        df_top25 = df.sort_values(by='Contextual Authority Score', ascending=False).head(25)
        if not df_top25.empty:
            sources = df_top25['Referring page URL'].tolist()
            targets = df_top25['Target URL'].tolist()
            all_nodes = list(set(sources + targets))
            node_map = {node: idx for idx, node in enumerate(all_nodes)}
            source_indices = [node_map[src] for src in sources]
            target_indices = [node_map[tgt] for tgt in targets]
            values = (df_top25['Domain rating'] + df_top25['Contextual Authority Score']).tolist() if 'Domain rating' in df.columns else (df_top25['UR'] + df_top25['Contextual Authority Score']).tolist()
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(label=[n[:50]+"..." if len(n)>50 else n for n in all_nodes], pad=15, thickness=20),
                link=dict(source=source_indices, target=target_indices, value=values)
            )])
            fig_sankey.update_layout(title_text="Top 25 Backlinks by CAS", height=600)
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("No data available for Sankey diagram.")

    with tab4:
        st.markdown("### 📈 Scatter Plot Analysis")
        y_col = 'Domain rating' if 'Domain rating' in df.columns else 'UR'
        fig_scatter = px.scatter(df, x='Cosine Similarity', y=y_col, hover_data=['Referring page URL'], title=f'{y_col} vs Cosine Similarity')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab5:
        st.markdown("### 📋 Complete Data Table")
        cols = ['Referring page URL', 'Target URL', 'UR', 'External links', 'Cosine Similarity', 'Contextual Authority Score']
        if 'Domain rating' in df.columns:
            cols.insert(1, 'Domain rating')
        st.dataframe(df[cols].sort_values(by='Contextual Authority Score', ascending=False), use_container_width=True, height=600)

    with tab6:
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
