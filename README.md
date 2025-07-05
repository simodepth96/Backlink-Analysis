# Backlink Semantic Analysis

This project enhances your backlink profile analysis by combining **Domain Rating (DR)** with **semantic relevance** between backlinks and your website’s target pages.


### The Python Script

The core of this project is a [Python script](https://github.com/simodepth96/Backlink-Analysis/blob/main/raw_code.py) that:

1. **Generates embeddings** for both backlink URLs and target page URLs using one model of your choice from `sentence-transformers`, which is a library well known for semantic understanding.
2. **Computes cosine similarity** between each backlink and its corresponding target page to quantify semantic closeness.
3. **Visualizes the data** with:
   - A **scatterplot** showing Domain Rating (DR) vs. Cosine Similarity
   - A **bar chart** of the **Top 10 backlinks** ranked by semantic similarity
   - A **downloadable dataset** of the full analysis output

### The Streamlit App

A [Streamlit app](https://semantic-backlink-similarity.streamlit.app/) was built upon the script to get you started with ease. 

Simply upload an Ahrefs backlink export in `.xlsx` format and evaluate your brand's backlink profile using **cosine similarity** as a measure of semantic proximity between backlink URLs and their corresponding target URLs.

---
## What's Cosine Similarity?
Cosine similarity is used as a notion of proximity between the vectors to express a score of similarity between the documents, namely the backlink URLs and the related Target URL. 

---

## Why Combine DR and Semantic Similarity?

While Domain Rating is a widely used authority signal, it has limitations and can be influenced by outliers depending on how third-party providers (Ahrefs, SEMrush) trained their datasets and how machine learning systems were instructed to retrieve the raw data to build up the metric.

Adding cosine similarity between backlinks and target URLs gives your analysis an additional **semantic layer of precision**.

> ✅ **Ideal backlinks**: High DR + High Cosine Similarity (close to 1)  
> ⚠️ **Potential red flags**: Low DR and Low Semantic Similarity  

## How is this Helpful for SEO and Digital PR?

- Uncover **undervalued but contextually strong** backlinks.
- Spot **irrelevant high-DR links** that don’t support your campaign narrative.
- Prioritise link opportunities that align both **authority** and **semantic context**.

---

## Why did I choose Sentence Transformers over Word2Vec or Jina AI?

The script is powered by [SentenceTransformers](https://www.sbert.net/) — chosen for their exceptional semantic accuracy and adaptability across various NLP tasks.

While testing traditional models like **Word2Vec** and some semantic models from **Jina AI**, SentenceTransformers consistently outperformed them in both relevance and robustness of embeddings for cosine similarity computations.

---

## The Process

1. **Export your backlinks**  
   Go to the **Backlinks** section in [Ahrefs](https://ahrefs.com/), and export the report as a **CSV** file.

2. **Clean the file and convert to XLSX**  
   Open the CSV file and **remove any unnecessary columns manually**. Save the cleaned file as an `.xlsx`.  
   The final input should look like this:
   ![Expected input format](https://github.com/user-attachments/assets/f7fbc1fe-56d4-43ba-a20a-202a282df8e0)

3. **Upload your file to the app**  
   Head over to the [Streamlit app](https://semantic-backlink-similarity.streamlit.app/) and upload your `.xlsx` file.

4. **Select a model**  
   Once uploaded, select a model from the dropdown menu.  
   I strongly recommend using **`all-MiniLM-L6-v2`** — it’s lightweight, robust, and offers the best balance between semantic performance and resource efficiency
**TL;DR - it's the most accurate and the quickest one to run especially if you have +100K or rows to process**

