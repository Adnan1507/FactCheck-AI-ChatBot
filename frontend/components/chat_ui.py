# ============================================================
# chat_ui.py — Chat Message Display Components
# ============================================================

import streamlit as st


def render_user_message(message: str):
    """Renders a user chat bubble."""
    with st.chat_message("user", avatar="👤"):
        st.markdown(message)


def render_assistant_message(
    full_response: str,
    verdict: str = "UNVERIFIED",
    confidence: str = "LOW",
    sources: list = None,
    langsmith_url: str = None
):
    """
    Renders the assistant response with a clear verdict card, full reasoning,
    and collapsible sources.
    """
    if sources is None:
        sources = []

    with st.chat_message("assistant", avatar="✨"):

        # ── VERDICT BANNER ───────────────────────────────────────────────────
        verdict_upper = (verdict or "UNVERIFIED").upper()
        confidence_upper = (confidence or "LOW").upper()

        if verdict_upper == "REAL":
            st.success(f"✅ VERDICT: **{verdict_upper}** — Confidence: **{confidence_upper}**")
        elif verdict_upper == "FAKE":
            st.error(f"❌ VERDICT: **{verdict_upper}** — Confidence: **{confidence_upper}**")
        elif verdict_upper == "ERROR":
            st.warning(f"⚠️ VERDICT: **{verdict_upper}** — Something went wrong.")
        else:
            st.warning(f"❓ VERDICT: **{verdict_upper}** — Confidence: **{confidence_upper}**")

        # ── FULL REASONING ───────────────────────────────────────────────────
        if full_response and full_response.strip():
            st.divider()
            st.markdown("#### 📋 Full Analysis")
            st.markdown(full_response)
        else:
            st.warning("⚠️ No detailed reasoning was returned from the AI. Try again.")

        # ── SOURCES ─────────────────────────────────────────────────────────
        if sources:
            st.divider()
            with st.expander(f"📚 View {len(sources)} Analyzed Source(s)"):
                for i, s in enumerate(sources, 1):
                    title = s.get("title", f"Source {i}")
                    url   = s.get("url", "")
                    content = s.get("content", "")
                    if url:
                        st.markdown(f"**{i}. [{title}]({url})**")
                    else:
                        st.markdown(f"**{i}. {title}**")
                    if content:
                        st.caption(content[:250])
        else:
            st.caption("ℹ️ No web sources were retrieved for this claim.")

        # ── TRACING ─────────────────────────────────────────────────────────
        if langsmith_url:
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            st.link_button("🔍 View Trace on LangSmith", langsmith_url)


def render_multi_claim_results(claims: list):
    """
    Renders per-claim verdict cards for multi-claim PDF analysis.
    Each card shows: claim text, verdict badge, confidence, and expandable details.
    """
    with st.chat_message("assistant", avatar="✨"):
        st.markdown(f"### 📄 Multi-Claim Analysis — {len(claims)} claim(s) found")
        st.divider()

        # ── Summary table ────────────────────────────────────────────────────
        rows = []
        for i, c in enumerate(claims, 1):
            v = (c.get("verdict") or "UNVERIFIED").upper()
            badge = "✅ REAL" if v == "REAL" else "❌ FAKE" if v == "FAKE" else "❓ UNVERIFIED"
            conf  = (c.get("confidence") or "LOW").upper()
            rows.append(f"| {i} | {c.get('claim','')[:80]} | {badge} | {conf} |")

        table = "| # | Claim | Verdict | Confidence |\n|---|-------|---------|------------|\n"
        table += "\n".join(rows)
        st.markdown(table)
        st.divider()

        # ── Per-claim detail cards ───────────────────────────────────────────
        for i, c in enumerate(claims, 1):
            verdict    = (c.get("verdict") or "UNVERIFIED").upper()
            confidence = (c.get("confidence") or "LOW").upper()
            claim_text = c.get("claim", f"Claim {i}")
            full_resp  = c.get("full_response", "")
            sources    = c.get("sources") or []
            error      = c.get("error")

            st.markdown(f"**Claim {i}:** {claim_text}")

            if error:
                st.warning(f"⚠️ Could not verify: {error}")
            elif verdict == "REAL":
                st.success(f"✅ **REAL** — Confidence: **{confidence}**")
            elif verdict == "FAKE":
                st.error(f"❌ **FAKE** — Confidence: **{confidence}**")
            else:
                st.warning(f"❓ **UNVERIFIED** — Confidence: **{confidence}**")

            if full_resp:
                with st.expander(f"📋 Full Analysis for Claim {i}"):
                    st.markdown(full_resp)
                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources[:3]:
                            title = s.get("title", "Source")
                            url   = s.get("url", "")
                            if url:
                                st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown(f"- {title}")

            if i < len(claims):
                st.divider()
