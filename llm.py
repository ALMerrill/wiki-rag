import os
import json
from dotenv import load_dotenv
import requests


class GeminiAPI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.prompt = "Given the following sentences and question, generate an answer."

    def query(self, question: str, relevant_sentences: list[str]):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        sentences = "\n\t\t".join(relevant_sentences)
        query_text = f"""
           {self.prompt}

           Sentences:
               {sentences}

           Question:
               {question}
        """
        print("QUERY:", query_text)
        data = {"contents": [{"parts": [{"text": query_text}]}]}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
