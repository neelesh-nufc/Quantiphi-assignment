import streamlit as st

from src.auth import get_allowed_companies, is_valid_user
from src.chat import answer_question
from src.config import COMPANY_NAMES

st.set_page_config(page_title="Earnings Call Q&A", page_icon="📊")


def _company_label(company: str) -> str:
    return COMPANY_NAMES.get(company, company)


def _md(text: str) -> str:
    """Escape $ so Streamlit doesn't render amounts like $7,624 ... $5 as LaTeX math."""
    return text.replace("$", "\\$")


def login_page() -> None:
    st.title("📊 Earnings Call Q&A")
    st.caption("Sign in with your email to query the documents you have access to.")

    with st.form("login"):
        email = st.text_input("Email", placeholder="alice@email.com")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if is_valid_user(email):
            st.session_state.user = email.strip().lower()
            st.session_state.companies = get_allowed_companies(email)
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Unknown user. Try one of the demo accounts listed in the README.")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {_company_label(s['company'])} · {s['quarter']} · page {s['page']}**")
            text = s["text"]
            st.caption(_md(text[:600] + ("…" if len(text) > 600 else "")))


def render_assistant(msg: dict) -> None:
    if msg.get("searched"):
        st.caption(f"🔎 Searched for: {msg['searched']}")
    st.markdown(_md(msg["content"]))
    render_sources(msg.get("sources", []))


def chat_page() -> None:
    with st.sidebar:
        st.markdown(f"**Signed in as**  \n{st.session_state.user}")
        st.markdown("**Access to:**")
        for c in st.session_state.companies:
            st.markdown(f"- {_company_label(c)}")
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.title("📊 Earnings Call Q&A")

    # Replay the conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant(msg)
            else:
                st.markdown(_md(msg["content"]))

    question = st.chat_input("Ask about the earnings calls…")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(_md(question))

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            try:
                result = answer_question(
                    question, st.session_state.messages, st.session_state.companies
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        reply = {
            "role": "assistant",
            "content": result["answer"],
            "sources": [{"text": d.page_content, **d.metadata} for d in result["sources"]],
            "searched": result["standalone_question"] if result["standalone_question"] != question else None,
        }
        render_assistant(reply)

    st.session_state.messages += [{"role": "user", "content": question}, reply]


if "user" in st.session_state:
    chat_page()
else:
    login_page()
