from typing import List, Dict
import requests

SYSTEM_PROMPT = """You are an expert tutor in AI, machine learning, and data engineering.
Your answers are based ONLY on the provided book excerpts.
If the context doesn't contain enough information, say so clearly.

Format your answer:
1. Direct answer to the question
2. Key concepts explained simply
3. Reference: mention which page the information came from"""


def build_prompt(query: str, chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Here are relevant excerpts from the book:\n\n{context}\n\n---\n\nBased on the excerpts above, answer this question:\n{query}"


class Generator:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, query: str, chunks: List[Dict]) -> dict:
        prompt = build_prompt(query, chunks)
        answer = self._call_ollama(prompt)
        sources = [
            {"page": c["page"], "source": c["source"], "score": c["score"]}
            for c in chunks
        ]
        return {"answer": answer, "sources": sources}

    def _call_ollama(self, prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False
                },
                timeout=300
            )
            return response.json()["response"]
        except Exception as e:
            return f"Error: {e}"