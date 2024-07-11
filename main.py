from pathlib import Path

from dataset import WikiDataset
from rag import RAGModel


def main():
    embedding_model_name = "all-mpnet-base-v2"

    wiki_dataset = WikiDataset(
        wiki_page_names=["Australia_women's_national_softball_team", "Renard_R.31"]
    )

    corpus = wiki_dataset.sentences
    model = RAGModel(embedding_model_name=embedding_model_name)
    model.set_corpus(corpus)

    questions = [
        "Which two companies created the R.31 reconnaissance aircraft?",
        "What guns were mounted on the Renard R.31?",
        "Who was the first softball player to represent any country at four World Series of Softball?",
        "Who were the pitchers on the Australian softball team's roster at the 2020 Summer Olympics?",
    ]

    k_value = 5
    for query in questions:
        context = model.retrieve_context(query, top_k=k_value)
        answer = model.generate_answer(query, context)
        print(query)
        print(answer)
        print()


if __name__ == "__main__":
    main()
