from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_dir = os.path.join(os.getcwd(), "datasets")

data_path = util.download_and_unzip(url, data_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"Dataset downloaded to: {data_path}")