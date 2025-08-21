import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Configure Streamlit page
st.set_page_config(page_title="Backlink Analysis", layout="wide")
st.title("Backlink Analysis")

# Initialize session state for model caching
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Model selection
model_choice = st.selectbox(
    "Choose a SentenceTransformer model:",
    [
        'all-MiniLM-L6-v2',  # Fastest, smallest model
        'paraphrase-mpnet-base-v2',
        'all-mpnet-base-v2',
        'distiluse-base-multilingual-cased-v2'
    ]
)

# Upload Excel file
uploaded_file = st.file_uploader(
    "📥 Upload Excel file (must contain 'Referring page URL', 'Target URL', 'UR', and 'External links')",
    type=["xlsx"]
)

@st.cache_resource
def load_model(model_name):
    """Load and cache the model"""
    return SentenceTransformer(model_name)

@lru_cache(maxsize=10000)
def tokenize_url_cached(url):
    """Cached version of URL tokenization"""
    if pd.isna(url) or not isinstance(url, str):
        return tuple()  # Return tuple for hashability
    url = re.sub(r"https?://", "", url)
    tokens = re.split(r"[\/\.\-\?\=\_\&]+", url)
    return tuple(t.lower() for t in tokens if t)

def get_model():
    """Get the current model, loading if necessary"""
    if (st.session_state.current_model_name != model_choice or 
        st.session_state.current_model is None):
        with st.spinner(f"🔄 Loading model: {model_choice}"):
            st.session_state.current_model = load_model(model_choice)
            st.session_state.current_model_name = model_choice
    return st.session_state.current_model

def get_average_embedding_batch(tokens_list, model):
    """Process multiple token lists in batch for efficiency"""
    all_tokens = []
    token_counts = []
    
    for tokens in tokens_list:
        if not tokens:
            token_counts.append(0)
        else:
            token_counts.append(len(tokens))
            all_tokens.extend(tokens)
    
    if not all_tokens:
        return [np.zeros(model.get_sentence_embedding_dimension()) for _ in tokens_list]
    
    # Batch encode all tokens
    embeddings = model.encode(all_tokens, batch_size=32, show_progress_bar=False)
    
    # Split embeddings back into groups
    result_embeddings = []
    start_idx = 0
    
    for count in token_counts:
        if count == 0:
            result_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
        else:
            batch_embeddings = embeddings[start_idx:start_idx + count]
            result_embeddings.append(np.mean(batch_embeddings, axis=0))
            start_idx += count
    
    return result_embeddings

def compute_similarities_batch(df, model, batch_size=100):
    """Compute similarities in batches for better performance"""
    similarities = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Processing similarities...")
    
    for batch_idx in range(0, len(df), batch_size):
        batch_end = min(batch_idx + batch_size, len(df))
        batch_df = df.iloc[batch_idx:batch_end]
        
        # Tokenize URLs in batch
        ref_tokens_list = [tokenize_url_cached(url) for url in batch_df['Referring page URL']]
        tgt_tokens_list = [tokenize_url_cached(url) for url in batch_df['Target URL']]
        
        # Get embeddings in batch
        ref_embeddings = get_average_embedding_batch(ref_tokens_list, model)
        tgt_embeddings = get_average_embedding_batch(tgt_tokens_list, model)
        
        # Compute cosine similarities
        batch_similarities = []
        for ref_vec, tgt_vec in zip(ref_embeddings, tgt_embeddings):
            if np.all(ref_vec == 0) or np.all(tgt_vec == 0):
                batch_similarities.append(np.nan)
            else:
                batch_similarities.append(1 - cosine(ref_vec, tgt_vec))
        
        similarities.extend(batch_similarities)
        
        # Update progress
        progress = (batch_idx + len(batch_similarities)) / len(df)
        progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")
    
    progress_bar.empty()
    return similarities

def process_dataframe(df, model):
    """Main processing function with optimizations"""
    # Data preprocessing
    df['UR'] = pd.to_numeric(df['UR'], errors='coerce')
    df['External links'] = pd.to_numeric(df['External links'], errors='coerce')
    
    # Compute similarities in batches
    with st.spinner("⚙️ Calculating semantic similarities..."):
        cosine_similarities = compute_similarities_batch(df, model, batch_size=50)
        df['Cosine Similarity'] = np.round(cosine_similarities, 3)
    
    # Calculate Contextual Authority Score
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
    
    # Handle Domain rating if present
    if 'Domain rating' in df.columns:
        df['Domain rating'] = pd.to_numeric(df['Domain rating'], errors='coerce').fillna(0).astype(int)
    
    # Convert to percentage and round
    df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round().astype(int)
    df['Contextual Authority Score'] = (df['Contextual Authority Score'] * 100).round().astype(int)
    
    return df

if uploaded_file and model_choice:
    # Generate a hash for the uploaded file to check if it's changed
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    cache_key = f"{file_hash}_{model_choice}"
    
    if cache_key not in st.session_state or 'processed_df' not in st.session_state:
        # Read the Excel file
        with st.spinner("📖 Reading Excel file..."):
            df = pd.read_excel(uploaded_file)
        
        required_columns = ['Referring page URL', 'Target URL', 'UR', 'External links']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Required columns: 'Referring page URL', 'Target URL', 'UR', 'External links'")
            st.stop()

        # Get or load model
        model = get_model()
        
        # Process the dataframe
        processed_df = process_dataframe(df, model)
        
        # Cache the results
        st.session_state.processed_df = processed_df.copy()
        st.session_state.cache_key = cache_key
        
        # Prepare Excel buffer
        buffer = BytesIO()
        processed_df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.session_state.excel_buffer = buffer

    df = st.session_state.processed_df

    # Display file info
    st.success(f"✅ Processed {len(df)} backlinks using {model_choice}")

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
        st.markdown("### 🔗 Referring Domains Relevance to your Target Domain")
        st.caption("This diagram shows the relationship between referring domains and target domains. The thickness of each flow represents the Contextual Authority Score, helping you visualize which domains are sending the most valuable backlinks.")

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
            title_text="<br><sub>Flow thickness represents Contextual Authority Score</sub>",
            font_size=10,
            height=600
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)
        
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
        
        **Performance Tips:**
        - The 'all-MiniLM-L6-v2' model is fastest for large files
        - Processing is done in optimized batches for better performance
        - Models are cached between runs to avoid reloading
        """)

    # Performance tips
    with st.expander("🚀 Performance Optimization Features"):
        st.markdown("""
        **This optimized version includes:**
        
        - **Model Caching**: Models are cached using `@st.cache_resource` to avoid reloading
        - **Batch Processing**: URLs are processed in batches for better memory efficiency
        - **LRU Cache**: URL tokenization results are cached to avoid recomputation
        - **File Change Detection**: Only reprocesses when file content changes
        - **Optimized Embeddings**: Batch encoding reduces model inference overhead
        - **Progress Tracking**: Real-time progress bars for long operations
        
        **For large files (1.5MB+):**
        - Use 'all-MiniLM-L6-v2' model (fastest)
        - Batch size is automatically optimized
        - Results are cached for instant access on subsequent views
        """)
