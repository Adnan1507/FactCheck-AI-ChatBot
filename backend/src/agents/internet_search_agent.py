"""
internet_search_agent.py
────────────────────────
Agent 1 — Internet Search Tool
Uses the Tavily API to search the internet for information
about a claim and return relevant results with sources.
"""

import os
from tavily import TavilyClient
from src.utils.config import TAVILY_API_KEY


def search_internet(query: str, max_results: int = 5) -> dict:
    """
    Searches the internet for the given query using Tavily.

    Args:
        query       : The claim or topic to search for.
        max_results : How many results to return (default 5).

    Returns:
        A dict with:
          - 'answer'  : A short AI-generated answer from Tavily.
          - 'sources' : List of {title, url, content} dicts.
          - 'error'   : Error message if something went wrong.
    """

    # ── Validate API key ─────────────────────────────────────────────────────
    if not TAVILY_API_KEY:
        return {
            "answer": "",
            "sources": [],
            "error": (
                "❌ Tavily API key is missing.\n"
                "👉 Get a free key at https://app.tavily.com/ "
                "and add it to your .env file."
            ),
        }

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Use Tavily's search with source gathering
        response = client.search(
            query=query,
            search_depth="advanced",      # gives deeper, more accurate results
            include_answer=True,          # Tavily generates a short summary
            include_raw_content=False,
            max_results=max_results,
        )

        # Extract sources from the response
        sources = []
        for result in response.get("results", []):
            sources.append({
                "title":   result.get("title", "No Title"),
                "url":     result.get("url", ""),
                "content": result.get("content", "")[:500],   # first 500 chars
            })

        return {
            "answer":  response.get("answer", ""),
            "sources": sources,
            "error":   None,
        }

    except Exception as e:
        error_msg = str(e)

        # ── Friendly API limit message ────────────────────────────────────────
        if "429" in error_msg or "limit" in error_msg.lower():
            return {
                "answer": "",
                "sources": [],
                "error": (
                    "⚠️ Tavily API rate limit reached.\n"
                    "👉 Please wait a moment and try again, "
                    "or check your usage at https://app.tavily.com/"
                ),
            }

        return {
            "answer":  "",
            "sources": [],
            "error":   f"❌ Internet search failed: {error_msg}",
        }
