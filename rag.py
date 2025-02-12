from typing import List
import torch
from sentence_transformers import SentenceTransformer
from llm import GeminiAPI


class RAGModel:
    def __init__(self, embedding_model_name: str = "all-mpnet-base-v2"):
        self.model = GeminiAPI()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.corpus_sentences: List[str] = []
        self.corpus_embeddings: torch.Tensor | None = None

    def embed_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Embeds a list of sentences using the SentenceTransformer model."""
        return self.embedding_model.encode(sentences, convert_to_tensor=True)

    def set_corpus(self, corpus: List[str]):
        """Sets the corpus of sentences/embeddings to use for retrieval."""
        self.corpus_sentences = corpus
        self.corpus_embeddings = self.embed_sentences(corpus)

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieves the top-k most similar sentences to the query from the corpus."""
        if len(self.corpus_sentences) == 0:
            raise ValueError("Corpus not set. Use `set_corpus` first.")
        query_embedding = self.embed_sentences([query])[0]
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding, self.corpus_embeddings
        )
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        return [self.corpus_sentences[idx] for idx in top_indices]

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generates an answer to the query given the context sentences."""
        answer = self.model.query(query, context)
        return answer
