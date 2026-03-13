# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List
from llm_agents.tools.base import ToolInterface
import requests


def _bing_search_results(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Query Bing Web Search API (Azure Cognitive Services) and return results."""

    subscription_key = os.getenv("AZURE_BING_SEARCH_KEY")  # Azure API Key
    endpoint = os.getenv("AZURE_BING_SEARCH_ENDPOINT")     # e.g. https://<your-region>.api.cognitive.microsoft.com

    if not subscription_key or not endpoint:
        raise EnvironmentError("Azure Bing credentials not found in environment variables.")

    search_url = f"{endpoint}/bing/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": max_results}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()

    return response.json().get("webPages", {}).get("value", [])


def search(query: str) -> str:
    """Perform a web search using Azure Bing and return formatted text results."""
    results = _bing_search_results(query)

    if not results:
        return "No relevant Bing Search Result was found."

    toret = []
    for i, result in enumerate(results, 1):
        title = result.get("name")
        url = result.get("url")
        snippet = result.get("snippet")
        toret.append(f"Result {i}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n")

    return "\n".join(toret)


class AzureBingSearchTool(ToolInterface):
    """Tool for web search using Azure Bing Search API."""

    name: str = "Azure Bing Search"
    description: str = (
        "Searches the web using Azure Bing Search API. "
        "Provide a natural language question and receive summarized web information."
    )

    def use(self, input_text: str) -> str:
        return search(input_text)


if __name__ == '__main__':
    tool = AzureBingSearchTool()
    res = tool.use("Who was the pope in 2023?")
    print(res)
