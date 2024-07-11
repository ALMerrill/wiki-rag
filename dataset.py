from pathlib import Path
from typing import List, Tuple

import nltk
import pdfplumber
import torch
from torch.utils.data import Dataset
from transformers import RagTokenizer
import re
from bs4 import BeautifulSoup
import requests


def filter_sentence(sentence):
    patterns = [
        r"https?://\S+",  # URLs
        r"ISBN\s*\d+",  # ISBNs
        r"OCLC\s*\d+",  # OCLC numbers
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

    return moving_window_chunks


# Initial setup, using above functions
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


# Used for Doc2Vec
class PDFTextDataset(Dataset):
    """Torch Dataset for PDF Sentences."""

    def __init__(self, pdf_paths: List[Path]):
        sentences = []
        for path in pdf_paths:
            sentences.extend(pdf_to_sentence_chunks(path))
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        sentence = self.sentences[idx]
        return sentence


# Improved Wiki setup
class WikiDataset(Dataset):
    def __init__(self, wiki_page_names: List[str]):
        text = []
        for page_title in wiki_page_names:
            url = f"https://en.wikipedia.org/wiki/{page_title}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            content_div = soup.find("div", id="mw-content-text")
            extracted_text = ""

            for element in content_div.descendants:
                if isinstance(element, str):
                    if "References" in element and "[edit]" in element:
                        break
                    if ".mw-parser-output" in element:
                        continue
                    extracted_text += element.strip() + " "
                elif element.name == "li":
                    extracted_text += "\n- " + element.get_text(
                        separator=" ", strip=True
                    )

            text.extend(extracted_text.strip().split(". "))

        moving_window_chunks = []
        window_size = 3
        for i in range(len(text) - window_size):
            moving_window_chunks.append(" ".join(text[i : i + window_size]))

        self.sentences = moving_window_chunks

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        sentence = self.sentences[idx]
        return sentence
