# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import openai
import os
from pydantic import BaseModel
from typing import List, Optional

# os.environ['OPENAI_API_KEY'] = "xxx"

class ChatLLM(BaseModel):
    model: str = 'gpt-4o-2024-05-13'
    temperature: float = 0.0
    api_key: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the OPENAI_API_KEY environment variable or provide it explicitly.")
        openai.api_key = self.api_key

    def generate(self, prompt: str, stop: List[str] = None):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )
        return response.choices[0].message.content

