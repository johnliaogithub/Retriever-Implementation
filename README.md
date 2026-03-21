# Retriever Implementation

## Project Structure

```
Retriever-Implementation/
├── 'Literature Review - John Liao.pdf'  # Literature Review for Part 1 of the quiz
├── datasets/                            # Downloaded BEIR/SciFact dataset, created by download_scifact.py
├── results/                             # Output retrieval results (JSON), created by running sparse / dense retrieval scripts
├── download_scifact.py                  # Script to download the SciFact dataset
├── sparse_retrieval.py                  # BM25-based sparse retrieval script, converted from sparse_retrieval.ipynb
├── dense_retrieval.py                   # Dense (embedding-based) retrieval script, converted from dense_retrieval.ipynb
├── requirements.txt                     # Python dependencies
└── notebooks/
    ├── sparse_retrieval.ipynb           # Notebook for sparse retrieval, equivalent to sparse_retrieval.py
    └── dense_retrieval.ipynb            # Notebook for dense retrieval, equivalent to dense_retrieval.py
```

## Setup & Usage

### 1. Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the SciFact dataset

```bash
python3 download_scifact.py
```

This downloads and extracts the dataset into `datasets/scifact/`.

### 4. Run retrieval

Run both scripts for sparse and dense retrieval:

```bash
python3 sparse_retrieval.py
python3 dense_retrieval.py
```
- **Sparse retrieval (BM25):** `sparse_retrieval.py` — outputs `results/sparse_results.json`
- **Dense retrieval (embeddings):** `dense_retrieval.py` — outputs `results/dense_results.json`


### 5. Evaluate results

Results are saved in `results/sparse_results.json` and `results/dense_results.json`, ready for evaluation

---

## Results

### Sparse Retrieval Results

| Metric | @10 | @100 |
|--------|-----|------|
| NDCG | 0.5597 | 0.58389 |
| MAP | 0.51473 | 0.52019 |
| Recall | 0.68617 | 0.79294 |
| Precision | 0.07633 | 0.00890 |

### Dense Retrieval Results

| Metric | @10 | @100 |
|--------|-----|------|
| NDCG | 0.64508 | 0.67665 |
| MAP | 0.59593 | 0.60307 |
| Recall | 0.78333 | 0.92500 |
| Precision | 0.08833 | 0.01053 |

### Discussion

On all metrics, Dense Retrieval outperforms Sparse Retrieval on both Top 10 and Top 100 by a good margin. For instance, according to precision, dense retrieval gives an average of 0.88 relevant documents out of 10, while sparse retrieval gives an average of 0.76.

Dense Retrieval likely performed better due to its advantage of using the embeddings to represent the meaning of the words as opposed to keeping track of the words themselves, like sparse retrieval does. 

Dense Retrieval took significantly more time on embedding the corpus, with its cell taking 4 minutes 40 seconds. Both methods took comparable amounts of time to build the index and retrieve the Top-100 documents. On my machine, dense retrieval took 2.8 seconds to sparse retrieval's 4.3 seconds for Top 100 document retrieval. While the significance of this is debatable, since it was ran on a laptop that was simultaneously running other applications, it does, to some extent support the fact that the usage of dense retrieval does not take much time more than sparse retrieval as suggested by Karpukhin et al. (2020), which is important for real-time use. 

Notice also that the space taken for Dense Retrieval and Sparse Retrieval are comparable, with dense embeddings for the corpus taking up 7.59 MB and the embeddings for the queries taking up 0.44 MB. Meanwhile, the index for sparse retrieval occupies 8.76 MB. 

Thus, one can argue that the time and compute power used to create dense embeddings is the trade-off for a better retrieval performance. One might also point out the need for a pre-trained embedding model. 

### References
* **Karpukhin, V., et al. (2020).** [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/pdf/2004.04906). Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).

