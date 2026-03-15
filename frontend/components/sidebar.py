# ============================================================
# sidebar.py — Streamlit Sidebar Component
# ============================================================
# The sidebar shows app information, settings, model selection,
# and chat history.
# ============================================================

import streamlit as st
from utils.chat_storage import get_all_chats, load_chat, delete_chat, generate_chat_id


def render_sidebar():
    """
    Renders a comprehensive sidebar with:
    - New Chat button
    - Model selection (4 LLMs)
    - Recent chat history
    - App info
    """
    with st.sidebar:
        # ── Logo ───────────────────────────────────────────────
        st.markdown("<h2 style='margin: 0; font-size: 30px;'>FactCheck AI 🔍</h2>", unsafe_allow_html=True)
        st.divider()
        
        # ── New Chat Button (Prominent at top) ─────────────────
        if st.button("✏️ New Chat", use_container_width=True):
            st.session_state.chat_id = generate_chat_id()
            st.session_state.messages = []
            st.session_state.suggestion_clicked = None
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Model Selection ────────────────────────────────────
        st.markdown("<div style='font-size: 15px; font-weight: 600; color: #aaa; margin-bottom: 8px;'>Model</div>", unsafe_allow_html=True)
        
        model_options = {
            "⚡ Llama 3.3 70B (Groq)": "⚡ Llama 3.3 70B (Groq)",
            "🚀 Llama 3.1 8B (Groq)": "🚀 Llama 3.1 8B Instant (Groq)",
            "🌟 Gemini 2.0 Flash": "🌟 Gemini 2.0 Flash (Google)",
            "🧠 Gemini 1.5 Pro": "🧠 Gemini 1.5 Pro (Google)",
        }
        
        selected_display = st.selectbox(
            "Choose model",
            list(model_options.keys()),
            label_visibility="collapsed",
            key="model_selector"
        )
        
        # Store the full model name for backend
        st.session_state.selected_model = model_options[selected_display]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        
        # ── Chat History List ──────────────────────────────────
        st.markdown("<div style='font-size: 15px; font-weight: 600; color: #aaa; margin-bottom: 8px;'>Recent Chats</div>", unsafe_allow_html=True)
        
        chats = get_all_chats()
        if not chats:
            st.markdown("<div style='font-size: 15px; color: #666;'>No past chats yet.</div>", unsafe_allow_html=True)
        else:
            for chat in chats[:10]:  # Show last 10 chats
                col1, col2 = st.columns([0.82, 0.18])
                
                # Truncate title for button
                display_title = chat['title']
                if len(display_title) > 15:
                    display_title = display_title[:12] + "..."
                    
                # The chat selection button
                with col1:
                    is_active = (st.session_state.get("chat_id") == chat["id"])
                    
                    if st.button(f"💬 {display_title}", key=f"chat_{chat['id']}", use_container_width=True):
                        st.session_state.chat_id = chat["id"]
                        st.session_state.messages = load_chat(chat["id"])
                        st.session_state.suggestion_clicked = None
                        st.rerun()
                        
                # The delete button
                with col2:
                    if st.button("🗑️", key=f"del_{chat['id']}", help="Delete chat"):
                        delete_chat(chat["id"])
                        # If we deleted the active chat, reset to new chat
                        if st.session_state.get("chat_id") == chat["id"]:
                            st.session_state.chat_id = generate_chat_id()
                            st.session_state.messages = []
                        st.rerun()

        st.markdown("<br><hr style='margin: 10px 0; border-color: #333;'>", unsafe_allow_html=True)
        
        # ── Minimalist Info ───────────────────────────────────
        st.markdown(
            "<div style='color: #888; font-size: 15px; line-height: 1.6;'>"
            "<b>Powered by:</b><br>"
            "• Live Web Search<br>"
            "• Google Grounding<br>"
            "• Vector RAG<br>"
            "• OCR Vision"
            "</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        st.link_button("📡 View LangSmith Traces", "https://smith.langchain.com", use_container_width=True)
