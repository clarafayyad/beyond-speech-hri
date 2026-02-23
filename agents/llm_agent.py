import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../config/.env")


class LLMAgent:
    def __init__(
            self,
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=300
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def prompt_llm(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM did not return valid JSON.\nRaw output:\n{content}"
            ) from e
