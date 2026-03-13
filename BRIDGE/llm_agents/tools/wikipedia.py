# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any

from llm_agents.tools.base import ToolInterface
import wikipedia


def search(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=1)
        page = wikipedia.page(query)
        return f"Summary: {summary}\nLink: {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:3]  # Limiting to top 3 options
        return f"Disambiguation: {', '.join(options)}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found"
    except wikipedia.exceptions.WikipediaException as e:
        return f"Wikipedia error: {str(e)}"


class WikipediaAPITool(ToolInterface):
    """Tool for Wikipedia search results."""

    name: str = "Wikipedia Search"
    description: str = ("Get summary and link from Wikipedia for a given query. "
                        "Input should be a topic name or question like 'Python programming'.")
    
    def use(self, input_text: str) -> str:
        return search(input_text)


if __name__ == '__main__':
    s = WikipediaAPITool()
    res = s.use("Python programming")
    print(res)
