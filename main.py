from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from dataset import PDFDataset
from rag import RAGModel

pdf_paths = [Path("data/aus_womens_softball_team.pdf"), Path("data/renard_r31.pdf")]
# tokenizer_model_name = "bert-base-uncased"
tokenizer_model_name = "facebook/rag-token-nq"
embedding_model_name = "paraphrase-MiniLM-L6-v2"
# embedding_model_name = "all-mpnet-base-v2"
dataset = PDFDataset(pdf_paths, pretrained_model_name=tokenizer_model_name)
corpus = dataset.sentences
model = RAGModel(embedding_model_name="paraphrase-MiniLM-L6-v2")
model.set_corpus(corpus)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

questions = [
    "Which two companies created the R.31 reconnaissance aircraft?",
    "What guns were mounted on the Renard R.31?",
    "Who was the first softball player to represent any country at four World Series of Softball?",
    "Who were the pitchers on the Australian softball team's roster at the 2020 Summer Olympics?",
]

k_value = 5
for query in questions:
    print("###########################################################")
    context = model.retrieve_context(query, top_k=k_value)
    answer = model.generate_answer(query, context)
    print(query)
    print(answer)
    print()
