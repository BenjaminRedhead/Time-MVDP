# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict
import pandas as pd
import re
import os
import requests
from sklearn.metrics import mean_squared_error

class TimeSeriesTaskIterate:
    def __init__(self, model, prompt_examples: str, engine: str) -> None:
        self.model = model
        self.engine = engine
        self.prompt = self.make_prompt(prompt_examples=prompt_examples)

    def make_prompt(self, prompt_examples: str) -> str:
        header = """Concepts: {concepts}\n"""
        example_template = """Sentence: {sentence}

        What concepts from the concept list are missing from the sentence?

        Concept Feedback: {concept_feedback}

        Any feedback on commonsense?

        Commonsense Feedback: {commonsense_feedback}"""
        instr = "\n\nOkay, improve the sentence using the feedback:\n\n"

        examples_df = pd.read_json(prompt_examples, orient="records", lines=True)
        prompt = []

        for example in examples_df.to_dict(orient="records"):
            single_example = []
            for step in example["sentence_to_feedback"]:
                single_example.append(
                    example_template.format(
                        sentence=step["sentence"],
                        concept_feedback=step["concept_feedback"],
                        commonsense_feedback=step["commonsense_feedback"]
                    )
                )
            prompt.append(header.format(concepts=example["concepts"]) + instr.join(single_example))

        return "\n\n###\n\n".join(prompt) + "\n\n###\n\n"

    def make_one_iterate_example(self, concepts: List[str], sent_to_fb: List[Dict]) -> str:
        header = """Concepts: {concepts}\n"""
        example_template = """Sentence: {sentence}

        What concepts from the concept list are missing from the sentence?

        Concept Feedback: {concept_feedback}

        Any feedback on commonsense?

        Commonsense Feedback: {commonsense_feedback}"""
        instr = "\n\nOkay, improve the sentence using the feedback:\n\n"

        single_example = []
        for example in sent_to_fb:
            single_example.append(
                example_template.format(
                    sentence=example["sentence"],
                    concept_feedback=example["concept_feedback"],
                    commonsense_feedback=example["commonsense_feedback"]
                )
            )

        return header.format(concepts=concepts) + instr.join(single_example)

    def make_query(self, concepts: List[str], sent_to_fb: List[Dict]) -> str:
        query_example = self.make_one_iterate_example(concepts=concepts, sent_to_fb=sent_to_fb)
        return f"{self.prompt}\n\n###\n\n{query_example}\n\n###\n\nOkay, improve the sentence using the feedback:\n\n"

    def call_azure_openai_completion(self, prompt: str) -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = self.engine
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-13")

        url = f"{endpoint}/openai/deployments/{deployment}/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": key
        }

        payload = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "stop": ["###"]
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()

    def __call__(self, concepts: List[str], sent_to_fb: List[Dict]) -> str:
        transfer_query = self.make_query(concepts=concepts, sent_to_fb=sent_to_fb)
        response = self.call_azure_openai_completion(transfer_query)
        match = re.search("Sentence: (.*)", response)
        improved_sentence = match.group(1).strip() if match else response.split("\n")[0].strip()
        return improved_sentence

    def refine_text(self, text_description: str, feedback: Dict) -> str:
        return self.model.refine(text_description, feedback)


if __name__ == "__main__":
    obj = TimeSeriesTaskIterate(
        model="your-model-instance",
        prompt_examples="data/prompt/commongen/iterate.v1.jsonl",
        engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    )

    concepts = ["trend", "increase", "data"]
    sent_to_fb = [
        {"sentence": "The data shows an increasing trend over time.", "concept_feedback": "None", "commonsense_feedback": "The sentence is clear."},
        {"sentence": "Data shows trend.", "concept_feedback": "Missing 'increasing'", "commonsense_feedback": "Incomplete sentence."}
    ]

    refined_sentence = obj(concepts, sent_to_fb)
    print(refined_sentence)
