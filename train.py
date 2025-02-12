from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from dataset import PDFTextDataset

# Tried out Doc2Vec, but didn't get good results


def train():
    pdf_paths = [Path("data/aus_womens_softball_team.pdf"), Path("data/renard_r31.pdf")]
    train_data = PDFTextDataset(pdf_paths)
    tagged_data = [
        TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
        for i, _d in enumerate(train_data.sentences)
    ]
    model = Doc2Vec(vector_size=30, window=3, min_count=2, workers=8, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("models/d2v.model")


if __name__ == "__main__":
    train()
