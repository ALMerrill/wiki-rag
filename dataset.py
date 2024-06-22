from pathlib import Path
from typing import List, Tuple

import nltk
import pdfplumber
import torch
from torch.utils.data import Dataset
from transformers import RagTokenizer
import re


def filter_sentence(sentence):
    patterns = [
        r"https?://\S+",  # URLs
        r"ISBN\s*\d+",  # ISBNs
        r"OCLC\s*\d+",  # OCLC numbers
        r"Retrieved \d{1,2} [A-Z][a-z]+ \d{4}",  # Retrieval dates (e.g., Retrieved 10 November 2023)
    ]

    for pattern in patterns:
        if re.search(pattern, sentence):
            return False

    if len(sentence) < 5 or sentence.strip().startswith("{{cite web}}"):
        return False

    alphanumeric_ratio = sum(c.isalnum() for c in sentence) / len(sentence)
    min_alphanumeric_ratio = 0.5
    if alphanumeric_ratio < min_alphanumeric_ratio:
        return False

    numeric_ratio = sum(c.isdigit() for c in sentence) / len(sentence)
    max_numeric_ratio = 0.6
    if numeric_ratio > max_numeric_ratio:
        return False

    return True


def pdf_to_sentence_chunks(pdf_path: Path) -> List[str]:
    """Parses a PDF file into sentences."""
    nltk.download("punkt")
    text_chunks = []
    removed = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text().replace("\n", " ")
            potential_sentences = nltk.sent_tokenize(text)
            for potential_sentence in potential_sentences:
                for sentence in potential_sentence.split(". "):
                    if filter_sentence(sentence):
                        text_chunks.append(sentence)
                    else:
                        removed.append(sentence)

    moving_window_chunks = []
    window_size = 3
    for i in range(len(text_chunks) - window_size):
        moving_window_chunks.append(" ".join(text_chunks[i : i + window_size]))

    # for chunk in moving_window_chunks:
    #    print(chunk)
    # print()
    print("text_chunks:")
    for text_chunk in text_chunks:
        print(text_chunk)
    print()
    return moving_window_chunks

    # print("removed:")
    # for remove in removed:
    #    print(remove)
    return text_chunks


class PDFDataset(Dataset):
    """Torch Dataset for PDF Sentences."""

    def __init__(self, pdf_paths: List[Path], pretrained_model_name: str):
        self.tokenizer = RagTokenizer.from_pretrained(pretrained_model_name)
        sentences = []
        for path in pdf_paths:
            sentences.extend(pdf_to_sentence_chunks(path))
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        sentence = self.sentences[idx]
        embedding = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        return sentence, embedding


class PDFTextDataset(Dataset):
    """Torch Dataset for PDF Sentences."""

    def __init__(self, pdf_paths: List[Path]):
        sentences = []
        for path in pdf_paths:
            sentences.extend(pdf_to_sentence_chunks(path))
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        sentence = self.sentences[idx]
        return sentence
