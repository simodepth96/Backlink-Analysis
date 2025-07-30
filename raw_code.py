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

# Load Excel file
file_path = "/content/seodepths.com-backlinks-subdomains_2025-07-30_11-17-30.csv.xlsx" #manually upload in Colab the xlsx file and paste the link here
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

# Convert columns to numeric to ensure proper calculation
df['UR'] = pd.to_numeric(df['UR'], errors='coerce')
df['External links'] = pd.to_numeric(df['External links'], errors='coerce')

# Calculate Contextual Authority Score using specific columns
def calculate_contextual_authority_score(row):
    """
    Calculate Contextual Authority Score = (UR / External links) × Cosine Similarity
    """
    
    url_rating = row['UR']
    external_links = row['External links']
    cosine_sim = row['Cosine Similarity']
    
    # Check if all required values are available and valid
    if (pd.isna(url_rating) or pd.isna(external_links) or pd.isna(cosine_sim) or
        external_links == 0):
        return np.nan
    
    # Calculate the score
    authority_ratio = url_rating / external_links
    contextual_authority_score = authority_ratio * cosine_sim
    
    return contextual_authority_score

# Apply the Contextual Authority Score calculation
df['Contextual Authority Score'] = df.apply(calculate_contextual_authority_score, axis=1)

# Round for readability
df['Contextual Authority Score'] = df['Contextual Authority Score'].round(3)

# Scale cosine similarity to 0–100 and convert to integer
df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round().astype(int)

#Scale Contextual Authority Score to 0–100 and convert to integer
df['Contextual Authority Score'] = (df['Contextual Authority Score'] * 100).round().astype(int)

#output result
df[['Referring page URL','Domain rating', 'UR', 'External links', 'Target URL', 'Cosine Similarity', 'Contextual Authority Score']].head(10)

##############

# Make a Sankey diagram of Backlinks to URLs by Contextual Authority Score


import plotly.graph_objects as go

# Sort by Contextual Authority Score in descending order and get the top 25
df_top25 = df.sort_values(by='Contextual Authority Score', ascending=False).head(25)

# Create a list of unique source and target nodes
all_nodes = pd.concat([df_top25['Referring page URL'], df_top25['Target URL']]).unique()
node_dict = {node: i for i, node in enumerate(all_nodes)}

# Create the links for the Sankey diagram
links = []
for index, row in df_top25.iterrows():
    source_index = node_dict[row['Referring page URL']]
    target_index = node_dict[row['Target URL']]
    # Use a combination of Domain rating and Contextual Authority Score for value
    link_value = row['Domain rating'] + row['Contextual Authority Score'] # Or another combination as needed
    links.append({
        'source': source_index,
        'target': target_index,
        'value': link_value,
        'label': f"DR: {row['Domain rating']}, CAS: {row['Contextual Authority Score']}" # Label for hover
    })

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        label=[link['label'] for link in links] # Add link labels
    )
)])

fig.update_layout(title_text="Top 25 Backlinks to URLs by Contextual Authority Score", font_size=10)
fig.show()

##############

# Make a Scatterplot

fig = px.scatter(df, x='Cosine Similarity', y='Domain rating', title='Domain Rating vs Cosine Similarity')
fig.show()

#############
#Make a Barplot of Cosine Similarity

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

#############
#Make a Barplot of Contextual Authority Score

# Group by 'Referring page URL' and calculate the mean CAS
df_agg = df.groupby('Referring page URL')['Contextual Authority Score'].mean().reset_index()

# Sort by CAS in descending order and get the top 10
top_10_referring_pages = df_agg.sort_values(by='Contextual Authority Score', ascending=False).head(10)

# Create a bar chart
fig = px.bar(
    top_10_referring_pages,
    x='Contextual Authority Score',
    y='Referring page URL',
    title='Top 10 Backlinks by Contextual Authority Score'
)

# Customize the layout for better readability of x-axis labels
fig.update_layout(
    xaxis_tickangle=-45, # Angle the x-axis labels
    xaxis=dict(tickmode='auto', nticks=10) # Adjust tick mode and number
)

fig.show()
