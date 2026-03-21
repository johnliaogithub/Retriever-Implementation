#!/usr/bin/env python
# coding: utf-8

# NOTE: this file has been generated from the dense_retrieval.ipynb notebook. 

# ### Load the SciFact Corpus and Queries

# In[1]:


import json
from beir.datasets.data_loader import GenericDataLoader
import os

corpus, queries, qrels = GenericDataLoader(data_folder="datasets/scifact").load(split="test")


# ### Embed the Corpus with a Sentence-Transformers Model

# In[2]:


import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

model = SentenceTransformer(MODEL_NAME)

doc_ids = list(corpus.keys())

doc_texts = [
    (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
    for doc_id in doc_ids
]

corpus_embeddings = model.encode(
    doc_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-normalize for cosine similarity via inner product
)


# ### Build a FAISS Index

# In[3]:


import faiss

embedding_dim = corpus_embeddings.shape[1]

# Create index using IndexFlatIP which performs exact inner product search
index = faiss.IndexFlatIP(embedding_dim)
index.add(corpus_embeddings.astype(np.float32))


# ### Retrieve Top-100 Documents per Query

# In[4]:


TOP_K = 100

query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]

query_embeddings = model.encode(
    query_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# scores shape: (num_queries, TOP_K)
# indices shape: (num_queries, TOP_K)
scores, indices = index.search(query_embeddings.astype(np.float32), TOP_K)

# Build the results dict
results = {}
for i, query_id in enumerate(query_ids):
    results[query_id] = {
        doc_ids[idx]: float(scores[i][rank])
        for rank, idx in enumerate(indices[i])
        if idx != -1   # FAISS returns -1 when fewer results than TOP_K exist
    }


# In[5]:


#print out first query and its top document
sample_qid = next(iter(results))
top_doc = max(results[sample_qid], key=results[sample_qid].get)
print(f"Sample query '{sample_qid}': top doc = {top_doc}, score = {results[sample_qid][top_doc]:.4f}")


# ### Save Results for BEIR Evaluation

# In[6]:


OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dense_results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)


# In[ ]:


import sys

# get size of corpus embeddings and query embeddings in MB
corpus_embeddings_size = sys.getsizeof(corpus_embeddings) / (1024 * 1024)
query_embeddings_size = sys.getsizeof(query_embeddings) / (1024 * 1024)

print(f"Corpus Embeddings Size: {corpus_embeddings_size:.2f} MB")
print(f"Query Embeddings Size: {query_embeddings_size:.2f} MB")

