import streamlit as st
import base64
from pathlib import Path
import sys

# Add project root and backend to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from frontend.components.sidebar import render_sidebar
from frontend.components.chat_ui import render_user_message, render_assistant_message, render_multi_claim_results
from frontend.utils.chat_storage import generate_chat_id, save_chat

# Import backend orchestrator (tries two paths for compatibility)
BACKEND_AVAILABLE = False
backend_error = None

try:
    from src.orchestrator import fact_check, fact_check_multi
    BACKEND_AVAILABLE = True
except Exception as e:
    backend_error = str(e)
    try:
        from backend.src.orchestrator import fact_check, fact_check_multi
        BACKEND_AVAILABLE = True
    except Exception as e2:
        backend_error = f"Primary: {e}\nAlternate: {e2}"

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="FactCheck AI — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Theme */
    .stApp { background-color: #212121; color: #ECECEC; }

    /* Empty-state header */
    .main-header { text-align: center; padding: 40px 0 20px 0; }
    .main-header h1 { font-size: 36px; font-weight: 600; color: #FFFFFF; margin: 0; }
    .main-header p  { color: #9E9E9E; font-size: 16px; margin-top: 8px; }

    /* Chat input */
    .stChatInput > div {
        border-radius: 16px;
        border: 1px solid #424242;
        background-color: #2F2F2F;
    }
    .stChatInput > div:focus-within { border-color: #565869; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { display: none !important; }

    /* ── Permanent sticky sidebar ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="true"],
    section[data-testid="stSidebar"][aria-expanded="false"] {
        background-color: #1a1a1a !important;
        position: fixed !important;
        left: 0; top: 0;
        height: 100vh !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        z-index: 100 !important;
        border-right: 1px solid #2e2e2e !important;
        pointer-events: all !important;
    }

    /* Hide all collapse/toggle buttons */
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNavToggleButton"],
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarNavToggleButton"],
    button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
        width: 0 !important; height: 0 !important;
    }

    /* Main content — always offset by sidebar */
    main[data-testid="stMain"] {
        margin-left: 280px !important;
        width: calc(100% - 280px) !important;
        background-color: #212121 !important;
    }
    [data-testid="stAppViewContainer"] { margin-left: 280px !important; }

    .stMainBlockContainer {
        max-width: 860px !important;
        margin: 0 auto !important;
        padding-top: 24px !important;
        padding-bottom: 100px !important;
    }

    /* ── Alignment for Sidebar Delete Buttons ── */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] div[data-testid="column"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        height: 38px !important; /* Matches standard Streamlit button height */
        line-height: 1 !important;
    }

    /* ── Sidebar chat history buttons: fixed size, no expansion ── */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="column"]:first-child button {
        height: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        display: block !important;
        line-height: 36px !important;
        padding: 0 10px !important;
    }

    /* ── LangSmith / Link Button Highlight ── */
    .stLinkButton > a {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .stLinkButton > a:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: #ffaa00 !important;
        box-shadow: 0 0 12px rgba(255, 170, 0, 0.3) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
if "chat_id"       not in st.session_state: st.session_state.chat_id       = generate_chat_id()
if "messages"      not in st.session_state: st.session_state.messages      = []
if "selected_model" not in st.session_state: st.session_state.selected_model = "⚡ Llama 3.3 70B (Groq)"
if "suggestion_clicked" not in st.session_state: st.session_state.suggestion_clicked = None

# ── Sidebar ───────────────────────────────────────────────────
render_sidebar()

# ── Suggestion Card CSS ───────────────────────────────────────
st.markdown("""
<style>
    /* Suggestion cards — target by unique button keys */
    div[data-testid="stButton"][key="sug_0"] > button,
    div[data-testid="stButton"][key="sug_1"] > button,
    div[data-testid="stButton"][key="sug_2"] > button,
    div[data-testid="stButton"][key="sug_3"] > button,
    button[data-testid="sug_0"],
    button[data-testid="sug_1"],
    button[data-testid="sug_2"],
    button[data-testid="sug_3"],
    /* Fallback: all main-area buttons (not sidebar) during empty state */
    div[data-testid="stMain"] div[data-testid="stColumns"] button {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)) !important;
        border: 1px solid #444 !important;
        border-left: 3px solid #ffaa00 !important;
        border-radius: 14px !important;
        padding: 22px 20px !important;
        height: auto !important;
        min-height: 72px !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        color: #ECECEC !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stMain"] div[data-testid="stColumns"] button:hover {
        background: linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,255,255,0.04)) !important;
        border-color: #ffaa00 !important;
        border-left-color: #ffaa00 !important;
        box-shadow: 0 6px 28px rgba(255, 170, 0, 0.22) !important;
        transform: translateY(-3px) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Empty-state header ────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class='main-header'>
        <h1>FactCheck AI</h1>
        <p>What claim would you like to verify today?</p>
    </div>
    """, unsafe_allow_html=True)

    # Only show suggestion cards if no claim is being processed yet
    if not st.session_state.suggestion_clicked:
        suggestions = [
            {"icon": "🦠", "title": "Drinking bleach cures COVID-19", "desc": "Health & Safety"},
            {"icon": "🚀", "title": "NASA faked the moon landing", "desc": "Space & Science"},
            {"icon": "🏛️", "title": "Great Wall of China is visible from space", "desc": "Geography"},
            {"icon": "🎭", "title": "Deepfake video of the president", "desc": "Media Context"}
        ]

        cols = st.columns(2)
        for i, item in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"{item['icon']} {item['title']}\n{item['desc']}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.suggestion_clicked = item['title']
                    st.rerun()
    else:
        # Show a "searching" indicator while the AI processes the suggestion
        st.markdown(
            f"<div style='text-align:center; color:#888; font-size:15px; padding: 40px 0;'>"
            f"🔍 Investigating: <b style='color:#ECECEC'>{st.session_state.suggestion_clicked}</b>..."
            f"</div>",
            unsafe_allow_html=True
        )

# ── Chat History ──────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "user":
        render_user_message(message["content"])
    elif message.get("type") == "multi_claim":
        render_multi_claim_results(message["claims"])
    else:
        render_assistant_message(
            full_response=message.get("full_response") or message.get("explanation", ""),
            verdict=message.get("verdict", "UNVERIFIED"),
            confidence=message.get("confidence") or "LOW",
            sources=message.get("sources", []),
            langsmith_url=message.get("langsmith_url"),
        )

# ── Chat Input ───────────────────────────────────────────────
result = st.chat_input(
    "Type a news claim to fact-check... e.g. '5G towers cause cancer'",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp", "pdf"],
)

# Handle both direct input AND suggestive card clicks
suggested_claim = st.session_state.suggestion_clicked
if result or suggested_claim:
    # Get user text and uploaded file if any
    user_text = result.text if result else suggested_claim
    uploaded  = result.files[0] if (result and result.files) else None
    
    # Clear recommendation state once used
    st.session_state.suggestion_clicked = None

    if not user_text and uploaded:
        user_text = "Please fact-check the claim shown in this image."

    display_text = user_text + (f"  📎 *{uploaded.name}*" if uploaded else "")

    render_user_message(display_text)
    st.session_state.messages.append({"role": "user", "content": display_text})
    save_chat(st.session_state.chat_id, st.session_state.messages)

    file_bytes = uploaded.read() if uploaded else None

    if BACKEND_AVAILABLE:
        is_pdf = uploaded and uploaded.name.lower().endswith(".pdf")

        if is_pdf:
            # ── Multi-Claim PDF Path ───────────────────────────────────────
            with st.spinner("📄 Extracting claims from PDF... this may take 1–2 minutes"):
                try:
                    from src.agents.ocr_agent import extract_text_from_pdf
                except Exception:
                    from backend.src.agents.ocr_agent import extract_text_from_pdf

            try:
                pdf_result = extract_text_from_pdf(file_bytes)
                if pdf_result.get("error"):
                    st.error(f"❌ {pdf_result['error']}")
                else:
                    pdf_text = pdf_result["text"]
                    with st.spinner(f"🔍 Identifying and verifying each claim in the PDF..."):
                        multi_result = fact_check_multi(
                            text=pdf_text,
                            model_name=st.session_state.selected_model,
                        )

                    if multi_result.get("error"):
                        st.error(f"❌ {multi_result['error']}")
                    else:
                        claims = multi_result["claims"]
                        # Save a single assistant message that holds all claims
                        assistant_message = {
                            "role":       "assistant",
                            "type":       "multi_claim",
                            "claims":     claims,
                        }
                        st.session_state.messages.append(assistant_message)
                        save_chat(st.session_state.chat_id, st.session_state.messages)
                        render_multi_claim_results(claims)

            except Exception as e:
                import traceback
                st.error(f"❌ PDF multi-claim analysis failed: {e}")
                with st.expander("View error details"):
                    st.code(traceback.format_exc())

        else:
            # ── Single-Claim Path (text / image) ──────────────────────────
            with st.spinner("🔍 AI is searching the web and analysing sources..."):
                try:
                    result_data = fact_check(
                        claim=user_text,
                        model_name=st.session_state.selected_model,
                        file_bytes=file_bytes,
                        file_name=uploaded.name if uploaded else None,
                    )

                    if result_data.get("error"):
                        st.error(f"❌ {result_data['error']}")
                    else:
                        assistant_message = {
                            "role":          "assistant",
                            "verdict":       result_data.get("verdict") or "UNVERIFIED",
                            "confidence":    result_data.get("confidence") or "LOW",
                            "full_response": result_data.get("full_response") or result_data.get("explanation", ""),
                            "sources":       result_data.get("sources") or [],
                            "langsmith_url": result_data.get("langsmith_url"),
                        }
                        st.session_state.messages.append(assistant_message)
                        save_chat(st.session_state.chat_id, st.session_state.messages)

                        render_assistant_message(
                            full_response=assistant_message["full_response"],
                            verdict=assistant_message["verdict"],
                            confidence=assistant_message["confidence"],
                            sources=assistant_message["sources"],
                            langsmith_url=assistant_message["langsmith_url"],
                        )

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error during fact-checking: {e}")
                    with st.expander("View error details"):
                        st.code(traceback.format_exc())
    else:
        st.error(f"""
⚠️ **Backend not available.**

**Import Error:**
```
{backend_error}
```

**Fix:** `pip install -r backend/requirement.txt` then restart Streamlit.
""")
