# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import copy
import requests

from feedback import TimeSeriesFeedback

# === Azure OpenAI Call Wrapper ===
def call_azure_openai(prompt: str, system_msg: str = None, max_tokens: int = 512) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-13")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def auto_refine_text(
    generated_series,
    actual_series,
    initial_text,
    model="gpt-4o",
    iterations=5,
    score_threshold=25
):
    feedback_tool = TimeSeriesFeedback(model=model)

    current_text = initial_text
    best_text = current_text
    best_score = 0
    history = []

    for iteration in range(iterations):
        print(f"\n{'='*30}")
        print(f" Iteration {iteration+1}")
        print(f"{'='*30}")

        feedback_result = feedback_tool.evaluate(
            generated_series=generated_series,
            actual_series=actual_series,
            text_description=current_text
        )

        text_quality_scores = feedback_result["text_quality_scores"]
        suggestions = feedback_result["suggestions"]
        summary_feedback = feedback_result["text_feedback_summary"]

        total_score = 0
        print(f"\n--- Text Quality Scores ---")
        for key, val in text_quality_scores.items():
            score_value = int(val.split("/")[0])
            total_score += score_value
            print(f"{key}: {val} ({suggestions.get(key, 'No suggestion')})")

        if total_score > best_score:
            best_score = total_score
            best_text = current_text

        print(f"\nCurrent Score: {total_score}/25")
        print(f"Summary Feedback: {summary_feedback}")

        if best_score >= score_threshold:
            print(f"\n Score threshold reached ({best_score}/25). Stopping refinement.")
            break

        refined_text = apply_suggestions(current_text, suggestions)

        print(f"\n--- Refined Text ---\n{refined_text}")

        history.append({
            "iteration": iteration + 1,
            "text": current_text,
            "refined_text": refined_text,
            "total_score": total_score,
            "summary_feedback": summary_feedback,
            "suggestions": suggestions
        })

        current_text = refined_text

    print(f"\n{'='*30}")
    print(f" Final Best Text After {iteration+1} Iterations:")
    print(f"{'-'*30}")
    print(best_text)
    print(f"Score: {best_score}/25")
    print(f"{'='*30}")

    return best_text, best_score, history


def apply_suggestions(text, suggestions):
    suggestion_summary = ". ".join([f"{k}: {v}" for k, v in suggestions.items()])
    prompt = f"""
            Original description: {text}

            Suggestions for improvement:
            {suggestion_summary}

            Please revise the original description to address the suggestions, ensuring clarity and completeness.
            """

    return call_azure_openai(
        prompt,
        system_msg="You are an expert technical writer improving time series data descriptions."
    )
