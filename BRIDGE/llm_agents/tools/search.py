# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Based on https://github.com/hwchase17/langchain/blob/master/langchain/utilities/serpapi.py

import os
import sys
from typing import Any

from llm_agents.tools.base import ToolInterface

def search(query: str) -> str:
    # Placeholder implementation.
    # Replace this function with actual Bing Web Search API call using Azure SDK or REST API.
    # Example: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python

    return f"Search results for '{query}' would appear here using Bing Web Search API."
        
def _process_response(res: dict) -> str:
    """Process response from SerpAPI."""
    focus = ['title', 'snippet', 'link']
    get_focused = lambda x: {i: j for i, j in x.items() if i in focus}

    if "error" in res.keys():
        raise ValueError(f"Got error from SerpAPI: {res['error']}")
    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        toret = res["answer_box"]["answer"]
    elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
        toret = res["answer_box"]["snippet"]
    elif (
        "answer_box" in res.keys()
        and "snippet_highlighted_words" in res["answer_box"].keys()
    ):
        toret = res["answer_box"]["snippet_highlighted_words"][0]
    elif (
        "sports_results" in res.keys()
        and "game_spotlight" in res["sports_results"].keys()
    ):
        toret = res["sports_results"]["game_spotlight"]
    elif (
        "knowledge_graph" in res.keys()
        and "description" in res["knowledge_graph"].keys()
    ):
        toret = res["knowledge_graph"]["description"]
    elif "snippet" in res["organic_results"][0].keys():
        toret = res["organic_results"][0]["snippet"]

    else:
        toret = "No good search result found"
    
    toret_l = []
    if res.get("organic_results"):
        for i, result in enumerate(res["organic_results"], 1):
            focused_info = get_focused(result)
            toret_l.append(f"Result {i}: {focused_info.get('title')}\n"
                           f"Link: {focused_info.get('link')}\n"
                           f"Snippet: {focused_info.get('snippet')}\n")
    
    return toret + '\n'.join(toret_l)

    # return str(toret) + '\n' + str(toret_l)

class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class AzureSearchTool(ToolInterface):
    """Tool for performing web search using Microsoft Azure Bing Search."""

    name: str = "Azure Bing Search"
    description: str = (
        "Use this tool to retrieve information from the web based on a natural language query. "
        "Input should be a question like 'How to add numbers in Clojure?'. "
        "The output will be a concise and relevant answer based on Bing Web Search results."
    )

    def use(self, input_text: str) -> str:
        return search(input_text)
