"""
orchestrator.py
───────────────
The main "brain" of the FactCheck AI chatbot.
Coordinates all 4 agents:
  1. RAG Agent        — checks if we already know the answer
  2. Internet Search  — searches the web for fresh info
  3. Google Grounding — verifies using live Google results
  4. OCR Agent        — (called before this, text passed in)

Then sends everything to the selected LLM to produce a
final verdict: REAL / FAKE / UNVERIFIED + explanation.
All steps are automatically traced in LangSmith.
"""

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.internet_search_agent import search_internet
from src.agents.google_grounding_agent import ground_with_google
from src.agents.rag_agent import retrieve, store_text
from src.agents.ocr_agent import extract_text_from_image, extract_text_from_pdf
from src.utils.config import GROQ_API_KEY, GOOGLE_API_KEY, MODELS


# ── Load the selected LLM ─────────────────────────────────────────────────────
def _load_llm(model_name: str):
    """
    Creates and returns the LangChain LLM object for the chosen model.
    LangSmith automatically traces every call made through these objects.
    """
    model_info = MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}")

    if model_info["provider"] == "groq":
        return ChatGroq(
            model=model_info["model_id"],
            api_key=GROQ_API_KEY,
            temperature=0.1,    # low temp = more factual, less creative
        )
    else:   # google
        return ChatGoogleGenerativeAI(
            model=model_info["model_id"],
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
        )


# ── Main Fact-Check Function ──────────────────────────────────────────────────
def fact_check(claim: str, model_name: str, file_bytes: bytes = None, file_name: str = None) -> dict:
    """
    Runs the full fact-checking pipeline on a claim.

    Steps:
      1. OCR/PDF Extraction — if a file is provided, extract text and add to claim
      2. Check knowledge base (RAG) for existing info
      3. Search the internet for current information
      4. Ground with Google for authoritative sources
      5. Ask the LLM to reason over all gathered evidence
      6. Store the claim + evidence back into knowledge base

    Args:
        claim      : The text claim to verify.
        model_name : Which of the models to use.
        file_bytes : Optional raw bytes of an uploaded file.
        file_name  : Optional name of the uploaded file (for extension check).

    Returns:
        A dict with:
          - 'verdict'         : "REAL" / "FAKE" / "UNVERIFIED"
          - 'explanation'     : Detailed reasoning from the LLM
          - 'internet_sources': Sources from Tavily search
          - 'google_snippets' : Sources from Google grounding
          - 'rag_used'        : True if knowledge base had relevant info
          - 'error'           : Error message if something went wrong
    """
    
    # ── Step 0: OCR / PDF Extraction ──────────────────────────────────────────
    extracted_text = ""
    if file_bytes and file_name:
        ext = file_name.lower().split('.')[-1]
        
        if ext in ['png', 'jpg', 'jpeg', 'webp']:
            ocr_res = extract_text_from_image(file_bytes)
            if ocr_res.get("error"):
                return {"error": ocr_res["error"], "verdict": "ERROR"}
            extracted_text = ocr_res["text"]
        elif ext == 'pdf':
            pdf_res = extract_text_from_pdf(file_bytes)
            if pdf_res.get("error"):
                return {"error": pdf_res["error"], "verdict": "ERROR"}
            extracted_text = pdf_res["text"]

    # Combine claim and extracted text
    full_claim = claim
    if extracted_text:
        if claim:
            full_claim = f"{claim}\n\n[TEXT FROM UPLOADED FILE]:\n{extracted_text}"
        else:
            full_claim = extracted_text

    if not full_claim.strip():
        return {"error": "No claim or text provided for fact-checking.", "verdict": "ERROR"}

    # ── Step 1: RAG — Check knowledge base ───────────────────────────────────
    rag_result   = retrieve(full_claim)
    rag_context  = ""
    rag_used     = False

    if rag_result["found"] and not rag_result["error"]:
        rag_used = True
        snippets = [r["content"] for r in rag_result["results"]]
        rag_context = "PAST KNOWLEDGE BASE:\n" + "\n---\n".join(snippets)

    # ── Step 2: Internet Search ───────────────────────────────────────────────
    search_result   = search_internet(full_claim)
    search_context  = ""
    internet_sources = []

    if not search_result["error"]:
        internet_sources = search_result["sources"]
        if search_result["answer"]:
            search_context = f"INTERNET SEARCH ANSWER:\n{search_result['answer']}\n"
        for s in internet_sources:
            search_context += f"\nSource: {s['title']} ({s['url']})\n{s['content']}\n"

    # ── Step 3: Google Grounding ──────────────────────────────────────────────
    grounding_result  = ground_with_google(full_claim)
    grounding_context = ""
    google_snippets   = []

    if not grounding_result["error"]:
        google_snippets   = grounding_result["snippets"]
        grounding_context = f"GOOGLE GROUNDED RESPONSE:\n{grounding_result['response']}\n"

    # ── Step 4: LLM Reasoning ─────────────────────────────────────────────────
    try:
        llm = _load_llm(model_name)

        system_prompt = """You are a highly skilled professional Fact-Checker AI.
Your goal is to provide a definitive verdict on the claim based ONLY on the evidence provided.

STRUCTURE YOUR RESPONSE EXACTLY LIKE THIS:

VERDICT: [REAL / FAKE / UNVERIFIED]

EXPLANATION:
- Provide a clear, detailed breakdown of WHY you reached this verdict.
- Cite specific sources or evidence provided in the context.
- Address any contradictions in the evidence.

CONFIDENCE: [HIGH / MEDIUM / LOW]

SOURCES USED:
- List the titles and URLs of the key sources used.

DO NOT include any other text outside these sections."""

        user_message = f"""
EXAMINE THE FOLLOWING CLAIM AND GATHERED EVIDENCE:

CLAIM: "{full_claim}"

EVIDENCE GATHERED:
{rag_context if rag_context else "No past knowledge found."}

### INTERNET SEARCH RESULTS:
{search_context if search_context else "No internet search results found."}

### GOOGLE GROUNDING:
{grounding_context if grounding_context else "No Google grounding snippets found."}

INSTRUCTIONS:
1. Analyze the evidence. 
2. If the evidence confirms the claim, verdict is REAL.
3. If the evidence refutes the claim, verdict is FAKE.
4. If the evidence is insufficient or contradictory, verdict is UNVERIFIED.
5. Provide a detailed EXPLANATION.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        answer   = response.content
        
        # DEBUG LOGGING
        print(f"--- ORCHESTRATOR DEBUG ---")
        print(f"Claim: {str(full_claim)[:100]}...")
        print(f"Internet Sources: {len(internet_sources)}")
        print(f"Google Snippets: {len(google_snippets)}")
        print(f"Raw LLM Answer: {str(answer)[:200]}...")
        print(f"--------------------------")

        # Parse verdict from response
        res_upper = answer.upper()
        verdict = "UNVERIFIED"
        if any(x in res_upper for x in ["VERDICT: REAL", "VERDICT: TRUE", "VERDICT: CORRECT"]):
            verdict = "REAL"
        elif any(x in res_upper for x in ["VERDICT: FAKE", "VERDICT: FALSE", "VERDICT: INCORRECT"]):
            verdict = "FAKE"
        
        # Parse confidence
        confidence = "LOW"
        if "CONFIDENCE: HIGH" in res_upper:
            confidence = "HIGH"
        elif "CONFIDENCE: MEDIUM" in res_upper:
            confidence = "MEDIUM"

        # ── Step 5: Combine Sources for Frontend ──────────────────────────────
        all_sources = []
        for s in internet_sources:
            all_sources.append({
                "title": s.get("title", "Internet Source"),
                "url":   s.get("url", ""),
                "content": s.get("content", "")
            })
        for s in google_snippets:
            all_sources.append({
                "title": s.get("title", "Google Source"),
                "url":   s.get("uri", ""), # Google uses 'uri'
                "content": ""
            })

        # ── Step 6: Store into knowledge base ─────────────────────────────────
        knowledge_to_store = f"Claim: {full_claim}\n\nVerdict: {verdict}\n\n{answer}"
        store_text(knowledge_to_store, source=f"fact_check_{verdict.lower()}")

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "full_response":    answer,
            "sources":          all_sources,
            "internet_sources": internet_sources, # legacy
            "google_snippets":  google_snippets,   # legacy
            "rag_used":         rag_used,
            "error":            None,
        }

    except Exception as e:
        error_msg = str(e)

        # ── Friendly API limit messages ───────────────────────────────────────
        if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
            model_info   = MODELS.get(model_name, {})
            provider     = model_info.get("provider", "")
            alternatives = [
                name for name, m in MODELS.items()
                if m["provider"] != provider
            ]
            suggestion = alternatives[0] if alternatives else "another model"
            return {
                "verdict":          "ERROR",
                "confidence":       "ERROR",
                "full_response":    "",
                "sources":          [],
                "internet_sources": internet_sources,
                "google_snippets":  google_snippets,
                "rag_used":         rag_used,
                "error": (
                    f"⚠️ API rate limit reached for **{model_name}**.\n\n"
                    f"👉 Try switching to **{suggestion}** in the sidebar.\n"
                    f"   Or wait a minute and try again."
                ),
            }

        return {
            "verdict":          "ERROR",
            "confidence":       "ERROR",
            "full_response":    "",
            "sources":          [],
            "internet_sources": internet_sources,
            "google_snippets":  google_snippets,
            "rag_used":         rag_used,
            "error":            f"❌ Analysis failed: {error_msg}",
        }


# ── Claim Extractor ───────────────────────────────────────────────────────────
def extract_claims(text: str, model_name: str) -> list[str]:
    """
    Uses the LLM to extract up to 5 distinct verifiable factual claims
    from a block of text (e.g. a PDF page).

    Returns a list of claim strings, or an empty list on failure.
    """
    try:
        llm = _load_llm(model_name)

        system_prompt = """You are an expert claim extractor.
Given a block of text, identify up to 5 distinct, specific, verifiable factual claims.
Output ONLY a numbered list in this exact format:
1. <claim>
2. <claim>
...
Do NOT include explanations, commentary, or anything else."""

        user_message = f"""Extract up to 5 verifiable factual claims from this text:

\"\"\"{text[:4000]}\"\"\"

List only the claims, one per line, numbered 1-5."""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])

        lines = response.content.strip().splitlines()
        claims = []
        for line in lines:
            line = line.strip()
            # Strip leading "1. " / "2. " etc.
            import re
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                claims.append(cleaned)
            if len(claims) >= 5:
                break

        return claims

    except Exception as e:
        print(f"[extract_claims] Error: {e}")
        return []


# ── Multi-Claim Fact-Check ────────────────────────────────────────────────────
def fact_check_multi(text: str, model_name: str) -> dict:
    """
    Extracts up to 5 claims from `text` and fact-checks each one
    independently using the full pipeline.

    Returns:
        {
          "claims": [
            {
              "claim":       str,
              "verdict":     str,
              "confidence":  str,
              "full_response": str,
              "sources":     list,
              "error":       str | None
            },
            ...
          ],
          "error": None
        }
    """
    import time

    # Step 1: Extract claims
    claims = extract_claims(text, model_name)

    if not claims:
        # Fall back to single-claim on the whole text
        result = fact_check(claim=text, model_name=model_name)
        return {
            "claims": [{
                "claim":         text[:120] + "...",
                "verdict":       result.get("verdict", "UNVERIFIED"),
                "confidence":    result.get("confidence", "LOW"),
                "full_response": result.get("full_response", ""),
                "sources":       result.get("sources", []),
                "error":         result.get("error"),
            }],
            "error": None,
        }

    # Step 2: Fact-check each claim with a rate-limit guard
    results = []
    for i, claim in enumerate(claims):
        print(f"[fact_check_multi] Checking claim {i+1}/{len(claims)}: {claim[:80]}...")

        result = fact_check(claim=claim, model_name=model_name)

        results.append({
            "claim":         claim,
            "verdict":       result.get("verdict", "UNVERIFIED"),
            "confidence":    result.get("confidence", "LOW"),
            "full_response": result.get("full_response", ""),
            "sources":       result.get("sources", []),
            "error":         result.get("error"),
        })

        # Rate-limit guard: pause between calls (except after the last one)
        if i < len(claims) - 1:
            time.sleep(1.5)

    return {"claims": results, "error": None}
