#!/usr/bin/env python
# coding: utf-8

# NOTE: this file has been generated from the sparse_retrieval.ipynb notebook. 

# ### Load the SciFact Corpus and Queries

# In[3]:


import json
import os
from beir.datasets.data_loader import GenericDataLoader

corpus, queries, qrels = GenericDataLoader(data_folder="datasets/scifact").load(split="test")


# ### Index the Corpus with BM25

# In[4]:


from rank_bm25 import BM25Okapi

def tokenize(text):
    return text.lower().split()

# tokenize both title and text for each doc
doc_ids = list(corpus.keys())
tokenized_corpus = [tokenize(corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]) for doc_id in doc_ids]

# build the index of documents
bm25 = BM25Okapi(tokenized_corpus)


# ### Retrieve Top-100 Documents per Query

# In[5]:


import numpy as np
from tqdm import tqdm

TOP_K = 100

# results format expected by BEIR: {query_id: {doc_id: score, ...}, ...}
bm25_results = {}

for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)

    # Get indices of top-K documents, then sort by score descending
    top_k_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

    bm25_results[query_id] = {
        doc_ids[idx]: float(scores[idx])
        for idx in top_k_indices
    }


# In[6]:


# see first element
sample_qid = next(iter(bm25_results))
top_doc = max(bm25_results[sample_qid], key=bm25_results[sample_qid].get)
print(f"Sample query '{sample_qid}': top doc = {top_doc}, score = {bm25_results[sample_qid][top_doc]:.4f}")


# ### Save Results for BEIR Evaluation

# In[7]:


os.makedirs("results", exist_ok=True)

with open("results/sparse_results.json", "w") as f:
    json.dump(bm25_results, f)


# In[11]:


import pickle

sparse_index_size_mb = len(pickle.dumps(bm25)) / (1024 * 1024)

print(f"Sparse BM25 Index Size: {sparse_index_size_mb:.2f} MB")

