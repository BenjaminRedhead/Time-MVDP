# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import copy
import requests
from feedback import TimeSeriesFeedback
from predictor import LSTPromptPredictor

# ================== CONFIG ==================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-13")

TEAM_ITERATION_LIMIT = 3
GLOBAL_ITERATION_LIMIT = 2

# ================== UTILITY FUNCTION ==================

def call_azure_openai(prompt: str, max_tokens: int = 512) -> str:
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def apply_suggestions(text, suggestions, role="engineer", perspective="macro"):
    suggestion_summary = ". ".join([f"{k}: {v}" for k, v in suggestions.items()])

    focus_instruction = (
        "Focus on MICRO aspects: fine-grained patterns, peaks, dips, local trends, short-term variability."
        if perspective == "micro"
        else "Focus on MACRO aspects: overall trends, long-term cycles, seasonality, external factors."
    )

    prompt = f"""
            You are a {role} revising time series descriptions based on feedback.

            Original description:
            {text}

            Suggestions for improvement:
            {suggestion_summary}

            {focus_instruction}

            Revise the description, ensuring clarity, precision, and completeness.
            """
    return call_azure_openai(prompt)


def critic_review(feedback_summary, score, threshold=24):
    """
    Critic role: decide if the text needs more work.
    """
    if score >= threshold:
        return True, "Accepted by critic (score threshold reached)"
    
    if "unclear" in feedback_summary.lower() or "missing" in feedback_summary.lower():
        return False, "Critic found clarity or completeness issues"
    
    return False, "Critic suggests further refinement"

# ================== TEAM AGENT ==================

class Team:
    def __init__(self, name, generated_series, actual_series, initial_text, perspective):
        self.name = name
        self.perspective = perspective  # 'micro' or 'macro'
        self.generated_series = generated_series
        self.actual_series = actual_series
        self.current_text = initial_text
        self.feedback_tool = TimeSeriesFeedback(model=MODEL_NAME)
        self.iteration = 0
        self.max_iterations = TEAM_ITERATION_LIMIT
        self.logs = []

    def leader_run(self):
        """
        Leader coordinates Scientist, Engineer, Critic in sequence.
        """
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n[{self.name}] Leader: Starting Iteration {self.iteration}")

            # Step 1: Scientist analyzes
            feedback_result = self.scientist_analyze()

            # Step 2: Engineer revises
            revised_text = self.engineer_modify(feedback_result["suggestions"])

            # Step 3: Critic reviews
            is_accepted, reason = self.critic_decide(feedback_result, revised_text)

            # Record the iteration
            self.logs.append({
                "iteration": self.iteration,
                "text": self.current_text,
                "revised_text": revised_text,
                "feedback_result": feedback_result,
                "critic_decision": reason
            })

            print(f"[{self.name}] Critic decision: {reason}")

            # Update the text
            self.current_text = revised_text

            if is_accepted:
                print(f"[{self.name}] Team finished after {self.iteration} iterations!\n")
                break

        return self.current_text, self.logs

    def scientist_analyze(self):
        """
        Scientist performs analysis on current text.
        """
        print(f"[{self.name}] Scientist analyzing text with {self.perspective} focus...")
        feedback_result = self.feedback_tool.evaluate(
            generated_series=self.generated_series,
            actual_series=self.actual_series,
            text_description=self.current_text
        )
        return feedback_result

    def engineer_modify(self, suggestions):
        """
        Engineer applies suggestions to improve text.
        """
        print(f"[{self.name}] Engineer revising text...")
        revised_text = apply_suggestions(self.current_text, suggestions, role="engineer", perspective=self.perspective)
        return revised_text

    def critic_decide(self, feedback_result, revised_text):
        """
        Critic evaluates the revised text to determine if acceptable.
        """
        print(f"[{self.name}] Critic reviewing revision...")
        text_quality_scores = feedback_result["text_quality_scores"]
        summary_feedback = feedback_result["text_feedback_summary"]

        total_score = sum([int(v.split("/")[0]) for v in text_quality_scores.values()])

        decision, reason = critic_review(summary_feedback, total_score)
        return decision, reason


# ================== MANAGER AGENT ==================

class Manager:
    def __init__(self, generated_series, actual_series, initial_text, global_iterations=GLOBAL_ITERATION_LIMIT):
        self.generated_series = generated_series
        self.actual_series = actual_series
        self.initial_text = initial_text
        self.final_text = None
        self.max_global_iterations = global_iterations

    def run(self):
        print("\n===== Manager initializing teams =====\n")

        # Create both teams
        team_micro = Team("Team A (Micro)", self.generated_series, self.actual_series, self.initial_text, perspective="micro")
        team_macro = Team("Team B (Macro)", self.generated_series, self.actual_series, self.initial_text, perspective="macro")

        # Each team runs independently
        text_micro, logs_micro = team_micro.leader_run()
        text_macro, logs_macro = team_macro.leader_run()

        global_iteration = 0

        while global_iteration < self.max_global_iterations:
            global_iteration += 1

            # Teams discuss via leaders + Manager decision
            final_decision = self.manager_decision(text_micro, text_macro)

            # Assume both teams accept manager's decision
            print(f"\n===== Manager Global Iteration {global_iteration} Complete =====\n")

            # Optionally: here you could re-initialize teams with new feedback
            # For now, we break after first merge
            break

        self.final_text = final_decision

        # Save history
        history = {
            "micro_team_logs": logs_micro,
            "macro_team_logs": logs_macro,
            "final_text": self.final_text
        }

        self.save_results(history)
        return self.final_text

    def manager_decision(self, text_micro, text_macro):
        """
        Manager consolidates both leader suggestions.
        """
        print("\n===== Manager coordinating final discussion =====")
        print("\nLeader A presents (Micro Team):")
        print(text_micro)

        print("\nLeader B presents (Macro Team):")
        print(text_macro)

        # Manager compares and resolves
        discussion_prompt = f"""
        You are the Manager of a multi-agent system working on time series descriptions.

        You have two candidate revised descriptions:
        Leader A (Micro Focus): "{text_micro}"
        Leader B (Macro Focus): "{text_macro}"

        Analyze both versions. Produce a final version that combines the best aspects of both, ensuring clarity, precision, inclusion of trend, seasonality, external factors, and completeness.
        """

        print("\n===== Manager Final Decision =====")
        print(all_azure_openai(discussion_prompt))

        return all_azure_openai(discussion_prompt)

    def save_results(self, history):
        """
        Save iteration logs and final text to JSON.
        """
        with open("multi_agent_refinement_result.json", "w") as f:
            json.dump(history, f, indent=4)
        print("\n Results saved to multi_agent_refinement_result.json\n")
