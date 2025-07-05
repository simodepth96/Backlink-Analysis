#!pip install sentence-transformers
#!pip install plotly
#use ! if using Colab otherwise run without !

'''
Here are other available models from sentence-transformers
all-mpnet-base-v2 --> most accurate but a fairly slow
paraphrase-mpnet-base-v2 --> probably the most robust, but still slowish
distiluse-base-multilingual-cased-v2 --> lightweight, ideal for multilingual URLs encoding
'''

import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import plotly.express as px

# Load Excel file
file_path = "/content/backlink profile twinset.xlsx"
df = pd.read_excel(file_path)

# Define the columns of interest
url_columns = ['Referring page URL', 'Target URL']
for col in url_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Initialize Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2') #possibly the most robust, lightweight and therefore convenient model of all

# Function to tokenize URLs semantically
def tokenize_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return []
    # Remove protocol
    url = re.sub(r"https?://", "", url)
    # Split by separators
    tokens = re.split(r"[\/\.\-\?\=\_\&]+", url)
    return [t.lower() for t in tokens if t]

# Function to get average embedding for a list of tokens
def get_average_embedding(tokens):
    if not tokens:
        return np.zeros(model.get_sentence_embedding_dimension())
    embeddings = model.encode(tokens)
    return np.mean(embeddings, axis=0)

# Function to compute semantic cosine similarity between two URLs
def compute_token_based_similarity(ref_url, tgt_url):
    ref_tokens = tokenize_url(ref_url)
    tgt_tokens = tokenize_url(tgt_url)
    ref_vec = get_average_embedding(ref_tokens)
    tgt_vec = get_average_embedding(tgt_tokens)
    if np.all(ref_vec == 0) or np.all(tgt_vec == 0):
        return np.nan
    return 1 - cosine(ref_vec, tgt_vec)

# Apply similarity computation to each row
df['Cosine Similarity'] = df.apply(
    lambda row: compute_token_based_similarity(row['Referring page URL'], row['Target URL']),
    axis=1
)

# Optional: Round for readability
df['Cosine Similarity'] = df['Cosine Similarity'].round(3)

df['Domain rating']=df['Domain rating'].astype(int).round(1)
df['Cosine Similarity']=df['Cosine Similarity'].astype(float).round(2)

# Show sample
df[['Referring page URL', 'Target URL','Domain rating', 'Cosine Similarity']].head()

# Save result
df.to_excel("semantic_url_similarity.xlsx", index=False)

df['Domain rating'] = df['Domain rating'].astype(int)  # Already in 0–100 scale

# Scale cosine similarity to 0–100 and convert to integer
df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round().astype(int)
df.head()

##############
#MAke a Scatterplot

fig = px.scatter(df, x='Cosine Similarity', y='Domain rating', title='Domain Rating vs Cosine Similarity')
fig.show()

#############
#Make a Barplot

# Group by 'Referring page URL' and calculate the mean Cosine Similarity
df_agg = df.groupby('Referring page URL')['Cosine Similarity'].mean().reset_index()

# Sort by Cosine Similarity in descending order and get the top 10
top_10_referring_pages = df_agg.sort_values(by='Cosine Similarity', ascending=False).head(10)

# Create a bar chart
fig = px.bar(
    top_10_referring_pages,
    x='Cosine Similarity',
    y='Referring page URL',
    title='Top 10 Backlinks by Cosine Similarity'
)

# Customize the layout for better readability of x-axis labels
fig.update_layout(
    xaxis_tickangle=-45, # Angle the x-axis labels
    xaxis=dict(tickmode='auto', nticks=10) # Adjust tick mode and number
)

fig.show()
