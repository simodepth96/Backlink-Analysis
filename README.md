# Backlink Semantic Analysis with Embeddings & Cosine Similarity

The purpose of this script is to pair up Domain Rating (DR) with semantic relevance of backlinks to our website. A streamlit app will companion this script to help you better audit your brand's backlink profile. In a nutshell, a combination of high DR with high cosine similarity (close to 1) is a more robust and accurate indication of a healthy backlink acquisition.

This effort can help Digital PR teams to surface under-valued links and identify placements that might not support the campaign narrative.

The script is powered by sentence-transformers models due to their excellent semantic versatility and robustness in the output. 
After testing out other more traditional models including World2Vec or some from the semantic-oriented Jina's models, I feel pretty confident in concluding that sentence-transformers provide the better performant arsenal of semantic-based models to generate embeddings to compute cosine similarity calculations upon.

##The process
1. Head to Ahrefs and head over to the backlink section. Export the report as a csv
2. Remove unwanted columns manually from the output, and save it as a.xlsx file. Just make sure the input file looks like that:
3. ![image](https://github.com/user-attachments/assets/f7fbc1fe-56d4-43ba-a20a-202a282df8e0)

4. Head over to the link app and upload your file straight in
5. Once uploaded, make sure to choose from the rows of available models. I strongly suggest you use all-MiniLM-L6-v2 because lightweight, robust, pretty accurate and the quickest of all.
6. Backlink URLs are seized against their target destination based on semantic similarity. Cosine similarity is used as a notion of proximity between the vectors to express a score of similarity between the documents, namely the backlink URLs and the related Target URL. 
