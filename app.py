import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from io import BytesIO
import plotly.express as px

# Configure Streamlit page
st.set_page_config(page_title="Backlink Semantic Similarity", layout="wide")
st.title("🔗 Audit Semantic Similarity of Backlink URLs")

# Upload Excel file
uploaded_file = st.file_uploader(
    "📥 Upload Excel file (must contain 'Referring page URL' and 'Target URL')",
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
            cosine_similarities = []
            progress_bar = st.progress(0, text="Processing rows...")

            for i, row in df.iterrows():
                sim = compute_token_based_similarity(row['Referring page URL'], row['Target URL'])
                cosine_similarities.append(sim)
                progress = (i + 1) / len(df)
                progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")

            progress_bar.empty()
            df['Cosine Similarity'] = (np.round(cosine_similarities, 4) * 100).round().astype(int)

        if 'Domain rating' in df.columns:
            df['Domain rating'] = df['Domain rating'].astype(int)

        st.session_state.processed_df = df.copy()
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.session_state.excel_buffer = buffer

    df = st.session_state.processed_df

    if 'Domain rating' in df.columns:
        fig = px.scatter(
            df,
            x='Cosine Similarity',
            y='Domain rating',
            title='📊 Domain Rating vs. Cosine Similarity'
        )
        st.plotly_chart(fig, use_container_width=True)

    top_10_with_target = (
        df[['Referring page URL', 'Target URL', 'Cosine Similarity']]
        .sort_values(by='Cosine Similarity', ascending=False)
        .dropna()
        .head(10)
    )

    fig_bar = px.bar(
        top_10_with_target,
        x='Cosine Similarity',
        y='Referring page URL',
        orientation='h',
        title='🏆 Top 10 Backlinks by Cosine Similarity',
        custom_data=['Target URL']
    )

    fig_bar.update_traces(
        hovertemplate="<b>%{y}</b><br>Similarity: %{x}<br>Target: %{customdata[0]}"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Cosine Similarity Histogram ---
    df['Cosine Similarity Range'] = pd.cut(
        df['Cosine Similarity'],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
        include_lowest=True
    )

    similarity_distribution = (
        df.groupby('Cosine Similarity Range')['Referring page URL']
        .count()
        .reset_index()
        .rename(columns={'Referring page URL': 'Count'})
    )

    similarity_bar = px.bar(
        similarity_distribution,
        x='Cosine Similarity Range',
        y='Count',
        title='🔍 Distribution of Backlinks by Cosine Similarity Range',
        labels={'Count': 'Number of Backlinks', 'Cosine Similarity Range': 'Similarity Range'}
    )

    similarity_bar.update_layout(
        width=1024,
        height=500,
        xaxis_title='Cosine Similarity (%)',
        yaxis_title='Number of Backlinks',
        bargap=0.2
    )
    st.plotly_chart(similarity_bar, use_container_width=True)

    st.markdown("### 📄 Backlink URLs with DR & Cosine Similarity Scores")
    st.dataframe(df, use_container_width=True, height=1000)

    st.download_button(
        label="📥 Download results as Excel",
        data=st.session_state.excel_buffer,
        file_name="semantic_url_similarity.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
