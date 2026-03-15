"""
google_grounding_agent.py
──────────────────────────
Agent 2 — Grounding with Google Search
Uses Google's Gemini model with built-in Google Search grounding.
This means Gemini looks at real, live Google results to answer,
making it much more accurate and up-to-date than a regular LLM.
"""

import google.generativeai as genai
from src.utils.config import GOOGLE_API_KEY


def ground_with_google(claim: str) -> dict:
    """
    Sends a claim to Gemini with Google Search grounding enabled.
    Gemini will look up live Google results and base its answer on them.

    Args:
        claim : The news claim or statement to verify.

    Returns:
        A dict with:
          - 'response'      : Gemini's grounded answer.
          - 'search_queries': What Google was asked.
          - 'snippets'      : Actual snippets from real web pages.
          - 'error'         : Error message if something went wrong.
    """

    # ── Validate API key ─────────────────────────────────────────────────────
    if not GOOGLE_API_KEY:
        return {
            "response": "",
            "search_queries": [],
            "snippets": [],
            "error": (
                "❌ Google API key is missing.\n"
                "👉 Get a free key at https://aistudio.google.com/app/apikey "
                "and add it to your .env file."
            ),
        }

    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        # Use Gemini 2.0 Flash for grounding (it supports Google Search tool)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools="google_search_retrieval",   # ← this enables real Google grounding
        )

        prompt = f"""
You are a professional fact-checker. Verify the following claim using real Google search results.
Be concise and clear. State whether the claim appears to be TRUE, FALSE, or UNVERIFIED.

Claim to verify:
"{claim}"

Provide:
1. Your verdict (TRUE / FALSE / UNVERIFIED)
2. Brief explanation with evidence from search results
3. Any important context the user should know
"""

        response = model.generate_content(prompt)

        # ── Extract grounding metadata ────────────────────────────────────────
        search_queries = []
        snippets       = []

        try:
            grounding_meta = response.candidates[0].grounding_metadata
            if hasattr(grounding_meta, "search_entry_point"):
                search_queries = [
                    grounding_meta.search_entry_point.rendered_content
                ]
            if hasattr(grounding_meta, "grounding_chunks"):
                for chunk in grounding_meta.grounding_chunks:
                    if hasattr(chunk, "web"):
                        snippets.append({
                            "title": chunk.web.title,
                            "uri":   chunk.web.uri,
                        })
        except Exception:
            pass   # grounding metadata is optional, don't crash

        return {
            "response":       response.text,
            "search_queries": search_queries,
            "snippets":       snippets,
            "error":          None,
        }

    except Exception as e:
        error_msg = str(e)

        # ── Friendly API limit message ────────────────────────────────────────
        if "429" in error_msg or "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return {
                "response": "",
                "search_queries": [],
                "snippets": [],
                "error": (
                    "⚠️ Google API rate limit reached.\n"
                    "👉 Please wait a moment and try again, or switch to a Groq model.\n"
                    "   Check your quota at https://aistudio.google.com/"
                ),
            }

        return {
            "response":       "",
            "search_queries": [],
            "snippets":       [],
            "error":          f"❌ Google grounding failed: {error_msg}",
        }
