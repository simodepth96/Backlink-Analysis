# Backlink Semantic Analysis

This project enhances **backlink profile analysis** by combining **Domain Rating (DR)** with the **semantic relevance** between backlinks and your site’s target pages.

In my experience, the quality of a backlink often correlates with its position on the target page. This project focuses only on **backlinks placed within the main content area**, ignoring those in static sections like the footer. Footer links are frequently from spammy sources, purchased domains, owned sites, or social media, and are often repeated site-wide.

After measuring each backlink’s semantic relevance to its target URL using **cosine similarity**, the project applies the **Contextual Authority Score (CAS)**. CAS condenses backlink quality into a single, actionable value by combining page authority, link dilution, and semantic relevance — helping SEOs focus on the true value of their backlinks in line with modern ranking signals.

**N.B**- Nobody stops you from inspecting your *brick-and-mortar* backlink profile. The app and the code attached works perfectly fine evne if you export the entire backlink profile from Ahrefs - just expect more outliers

## The Process

1. **Export your backlinks**  
   Go to the **Backlinks** section in [Ahrefs](https://ahrefs.com/), make sure to toggle only **backlinks in content**, and export the report as a **CSV** file.
   <img width="1490" height="797" alt="image" src="https://github.com/user-attachments/assets/31eb6821-82fe-4cab-80fe-44a11297c52f" />


3. **Clean the file and convert to XLSX**  
   Open the CSV file and **remove any unnecessary columns manually**. Save the cleaned file as an `.xlsx`.  
   The expected input file must contain the following headers
   
   <img width="1923" height="25" alt="image" src="https://github.com/user-attachments/assets/511a790d-1290-4520-99f0-9643fd360b58" />

4. **Upload your file to the app**  
   Head over to the [Streamlit app](https://semantic-backlink-similarity.streamlit.app/) and upload your `.xlsx` file.

5. **Select a model**  
   Once uploaded, select a model from the dropdown menu.  
   I strongly recommend using **`all-MiniLM-L6-v2`** — it’s lightweight, robust, and offers the best balance between semantic performance and resource efficiency
**TL;DR - it's the most accurate and the quickest one to run especially if you have +100K or rows to process**

---
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

## Why Combine DR and Semantic Similarity?

While Domain Rating is a widely used authority signal, it has limitations and can be influenced by outliers depending on how third-party providers (Ahrefs, SEMrush) trained their datasets and how machine learning systems were instructed to retrieve the raw data to build up the metric.

Adding cosine similarity between backlinks and target URLs gives your analysis an additional **semantic layer of precision**.

> ✅ **Ideal backlinks**: High DR + High Cosine Similarity (close to 1)  
> ⚠️ **Potential red flags**: Low DR and Low Semantic Similarity  

---
## Cosine Similarity vs. Google’s RankEmbed

**Cosine similarity** is a basic way to measure how similar two pieces of text are. It compares the direction of word meanings (their “angle”) but ignores their "strength" (length). This means it only measures general topic similarity — it doesn’t understand word order or deeper relationships. For example:

> “The dog chased the cat”
> “The cat chased the dog”

Cosine similarity would see these as the same, even though **the meaning is different**

It’s also worth noting that text embeddings (the math behind this) aren’t perfect. They give rough approximations and can miss subtle differences. So, cosine similarity should serve as a *wooden stick* to support your analysis - not be your only decision-making tool.

**Google’s RankEmbed** goes further. It also considers vector length (strength), allowing it to combine meaning with other signals like PageRank, freshness, or click data.

**What this means for SEO**

- For indexation: Google doesn’t rank pages by meaning alone — it blends authority, freshness, and relevance (**DotProduct & RankEmbed** algorithms).
- For audits: Cosine similarity can be a useful early indicator, but it’s not the full picture. Don’t base important SEO decisions on it alone.

---
### Enter the Contextual Authority Score metric

I coined this new metric for the purpose of this project.
**Contextual Authority Score (CAS)** is a number that helps you judge the quality of a backlink.
The measure ponders:

- **Page authority** – how strong the linking page is.
- **Link dilution** – how many other external links are on that page.
- **Relevance** – how closely the linking page’s topic matches your page’s topic.

> CAS = (UR / ExLC) × S

ExLC (External Link Count) = 

S (Semantic Similarity) = 
- UR (URL Rating): strength of the specific linking page (not the whole site).

- ExLC (External Link Count): number of outbound links (pointing to external domains) from the linking page. It serves as a dilution factor, reducing the perceived value of a backlink when it's one of many on a page.

- S (Cosine Similarity): A semantic similarity score that reflects how topically relevant the link is to your page.

📊 Interpretation of CAS Values:
- **✅ High CAS** → A link is both authoritative and topically relevant.
- **❌ Low CAS** → A link may come from a weak page, be overly diluted, or be semantically off-topic.

🧰 Use Cases:
- Backlink Auditing: Identify which links are most beneficial and which are low-quality or off-topic.
- Link Prospecting: Target high CAS opportunities when building new links.
- SEO Reporting: Provide a data-driven score that reflects both authority and relevance.

📌 Important Notes:
Do not use CAS in isolation. Make sure you handle this metric alongside other backlink indicators (e.g., placement, anchor text).
CAS is largely biased by input metrics like UR, which are proprietary and not fully transparent.


| Metric             | Authority | Relevance | Link Equity Awareness |
| ------------------ | --------- | --------- | --------------------- |
| Domain Rating (DR) | ✅         | ❌         | ❌                     |
| URL Rating (UR)    | ✅         | ❌         | ❌                     |
| TF\*IDF Similarity | ❌         | ✅         | ❌                     |
| **CAS**            | ✅         | ✅         | ✅                     |


---

## How is this Helpful for SEO and Digital PR?

- Uncover **undervalued but contextually strong** backlinks.
- Spot **irrelevant high-DR links** that don’t support your campaign narrative.
- Prioritise link opportunities that align both **authority** and **semantic context**.
