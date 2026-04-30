import os
import requests
from typing import List

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

class LLMClient:
    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.hf_api_key = os.environ.get("HF_API_KEY") 
        self.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.gen_model = "llama-3.1-8b-instant"

    def is_available(self) -> bool:
        return bool(self.groq_api_key)

    def embed(self, text: str) -> List[float]:
        headers = {}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        try:
            response = requests.post(
                HF_API_URL, 
                headers=headers, 
                json={"inputs": text, "options": {"wait_for_model": True}}, 
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                     if isinstance(result[0], float):
                         return result
                     elif isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], float):
                         # Some HF pipelines return a list of lists of floats
                         return result[0]
            print(f"HF Embed Error: {response.status_code} {response.text}")
            return []
        except Exception as e:
            print(f"HF Request error: {e}")
            return []

    def generate(self, prompt: str) -> str:
        if not self.groq_api_key:
             return "ERROR: Groq configuration missing. Set GROQ_API_KEY environment variable."

        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            completion = client.chat.completions.create(
                model=self.gen_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"ERROR: Groq client failed. Details: {e}"
