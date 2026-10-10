"""Streamlit chat UI for the copilot.  Run:  streamlit run app.py   (API must be running)."""

import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"x-api-key": os.getenv("API_KEY", "")} if os.getenv("API_KEY") else {}

st.set_page_config(page_title="Knowledge Copilot", page_icon="💬", layout="wide")
st.title("💬 Multimodal Knowledge Copilot")
st.caption("Ask questions over your PDFs, images, audio and video. Answers cite their sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "fb_open" not in st.session_state:
    st.session_state.fb_open = None

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", ["gemini", "openai", "anthropic"])
    k = st.slider("Passages to retrieve (k)", 1, 10, 4)
    st.header("Add documents")
    uploads = st.file_uploader(
        "PDF, text, image, audio or video",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "mp4", "mkv", "mov"],
    )
    if uploads and st.button("Index files"):
        with st.spinner("Transcribing / captioning / embedding..."):
            r = requests.post(
                f"{API}/ingest",
                headers=HEADERS,
                files=[("files", (u.name, u.getvalue())) for u in uploads],
                timeout=1800,
            )
        if r.ok:
            st.success(f"Indexed {r.json()['chunks_indexed']} chunks")
            if r.json()["errors"]:
                st.warning(r.json()["errors"])
        else:
            st.error(r.text)


def send_feedback(interaction_id: int, rating: int, corrected: str | None = None) -> None:
    r = requests.post(
        f"{API}/feedback",
        headers=HEADERS,
        json={"interaction_id": interaction_id, "rating": rating, "corrected_answer": corrected},
        timeout=30,
    )
    st.toast("Thanks for the feedback!" if r.ok else f"Feedback failed: {r.text}")


for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("interaction_id", -1) >= 0:
            with st.expander("Sources"):
                for s in m.get("sources", []):
                    st.markdown(
                        f"**[{s['n']}] {s['source']}** ({s['modality']}, {s['locator']})  \n> {s['snippet']}..."
                    )
            c1, c2, _ = st.columns([1, 1, 8])
            if c1.button("👍", key=f"up{idx}"):
                send_feedback(m["interaction_id"], 1)
            if c2.button("👎", key=f"down{idx}"):
                st.session_state.fb_open = idx
            if st.session_state.fb_open == idx:
                corrected = st.text_area(
                    "Write the correct answer (optional but powers preference tuning)", key=f"fix{idx}"
                )
                if st.button("Submit correction", key=f"sub{idx}"):
                    send_feedback(m["interaction_id"], -1, corrected)
                    st.session_state.fb_open = None

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        r = requests.post(
            f"{API}/ask", headers=HEADERS, json={"question": question, "provider": provider, "k": k}, timeout=300
        )
    if r.ok:
        d = r.json()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": d["answer"],
                "sources": d["sources"],
                "interaction_id": d["interaction_id"],
            }
        )
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Error: {r.text}", "sources": [], "interaction_id": -1}
        )
    st.rerun()
