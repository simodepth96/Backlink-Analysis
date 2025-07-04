import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Backlink Semantic Similarity", layout="wide")

st.title("🔗 Semantic Similarity of Backlink URLs")

uploaded_file = st.file_uploader("📥 Upload Excel file (must contain 'Referring page URL' and 'Target URL')", type=["xlsx"])

model_choice = st.selectbox(
    "🤖 Choose a SentenceTransformer model:",
    [
        'all-MiniLM-L6-v2',
        'paraphrase-mpnet-base-v2',
        'all-mpnet-base-v2',
        'distiluse-base-multilingual-cased-v2'
    ]
)

# Ensure we only reprocess if a file is uploaded
if uploaded_file and model_choice:
    if 'processed_df' not in st.session_state:
        df = pd.read_excel(uploaded_file)

        required_columns = ['Referring page URL', 'Target URL']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"❌ Missing required column: {col}")
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
            df['Cosine Similarity'] = df.apply(
                lambda row: compute_token_based_similarity(row['Referring page URL'], row['Target URL']), axis=1
            )
            df['Cosine Similarity'] = df['Cosine Similarity'].astype(float).round(2)

            if 'Domain rating' in df.columns:
                df['Domain rating'] = df['Domain rating'].astype(int).round(1)

            # Save to session to avoid reprocessing on download
            st.session_state.processed_df = df.copy()

            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.session_state.excel_buffer = buffer

    # Retrieve processed dataframe from session state
    df = st.session_state.processed_df

    # Scatter plot
    if 'Domain rating' in df.columns:
        fig = px.scatter(
            df, x='Cosine Similarity', y='Domain rating',
            title='📊 Domain Rating vs. Cosine Similarity'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart - Top 10 referring pages
    df_agg = df.groupby('Referring page URL')['Cosine Similarity'].mean().reset_index()
    top_10 = df_agg.sort_values(by='Cosine Similarity', ascending=False).head(10)
    fig_bar = px.bar(
        top_10, x='Cosine Similarity', y='Referring page URL',
        title='🏆 Top 10 Backlinks by Cosine Similarity', orientation='h'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Display full-size DataFrame
    st.markdown("### 🔍 Full Data")
    st.dataframe(df, use_container_width=True, height=1000)

    # Download button
    st.download_button(
        label="📥 Download results as Excel",
        data=st.session_state.excel_buffer,
        file_name="semantic_url_similarity.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
