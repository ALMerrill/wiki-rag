from typing import List
import torch
from sentence_transformers import SentenceTransformer
from llm import GeminiAPI
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize


class RAGModel:
    def __init__(self, embedding_model_name: str = "all-mpnet-base-v2"):
        self.model = GeminiAPI()
        # self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model = Doc2Vec.load("models/d2v.model")
        self.corpus_embeddings: torch.Tensor = torch.Tensor()
        # self.corpus_sentences: List[str] = []

    def embed_sentences(self, sentences: List[str]) -> torch.Tensor:
        return self.embedding_model.encode(sentences, convert_to_tensor=True)

    def set_corpus(self, corpus: List[str]):
        # SentenceTransformer
        self.corpus_sentences = corpus
        # self.corpus_embeddings = self.embed_sentences(corpus)
        pass

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        # SentenceTransformer
        # if len(self.corpus_sentences) == 0:
        #    raise ValueError("Corpus not set. Use `set_corpus` first.")
        # query_embedding = self.embed_sentences([query])[0]
        # similarities = torch.nn.functional.cosine_similarity(
        #    query_embedding, self.corpus_embeddings
        # )
        # top_indices = torch.argsort(similarities, descending=True)[:top_k]

        # Doc2Vec
        # similarities = self.embedding_model.docvecs.most_similar([query], topn=top_k)
        query_tokens = word_tokenize(query)  # Tokenize the query
        query_vector = self.embedding_model.infer_vector(query_tokens)
        similarities = self.embedding_model.docvecs.most_similar(
            [query_vector], topn=top_k
        )
        return [self.corpus_sentences[int(idx)] for idx, _ in similarities]

    def generate_answer(self, query: str, context: List[str]) -> str:
        answer = self.model.query(query, context)
        return answer
