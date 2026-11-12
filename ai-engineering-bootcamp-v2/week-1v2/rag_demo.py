"""Streamlit front end for the Session 2 RAG API: ingest a document, ask a question.

This page only calls the FastAPI service (POST /ingest, POST /ask). All chunking,
embedding, retrieval and generation happen in the API; nothing here talks to
OpenAI or Pinecone, and no secrets are needed.

Run (point it at your live Render service):
  API_BASE_URL=https://YOUR-SERVICE.onrender.com streamlit run rag_demo.py
Or run `streamlit run rag_demo.py` and paste the URL into the sidebar.
"""

import os

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 120  # Render's free tier can take ~a minute to wake up.


def call_api(method: str, url: str, payload: dict | None = None) -> tuple[int, dict | str]:
    """Return (status_code, body). status_code 0 means the request never got a reply."""

    try:
        response = httpx.request(method, url, json=payload, timeout=TIMEOUT_S)
    except httpx.HTTPError as exc:
        return 0, f"Could not reach {url}: {exc}"
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def show_error(status: int, data: dict | str) -> None:
    detail = data.get("detail", data) if isinstance(data, dict) else data
    st.error(f"HTTP {status}: {detail}" if status else str(detail))


def show_answer(data: dict) -> None:
    answer = data["answer"]["answer"]
    if data.get("refused"):
        st.warning(f"**Refused - not in the documents**\n\n{answer}")
        st.caption("No citations: none of the retrieved chunks contained the answer.")
    else:
        st.success(answer)
        st.markdown("**Citations**")
        st.dataframe(
            [
                {
                    "document_id": c["document_id"],
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "score": c["score"],
                }
                for c in data.get("citations", [])
            ],
            hide_index=True,
            use_container_width=True,
        )

    cols = st.columns(4)
    cols[0].metric("Model", data.get("model", "-"))
    cols[1].metric("Tokens", data.get("tokens_used", "-"))
    cols[2].metric("Cost", f"${data.get('cost_usd', 0):.6f}")
    cols[3].metric("Latency", f"{data.get('latency_ms', '-')} ms")

    with st.expander("Retrieved chunks (what the model was given)"):
        st.write(", ".join(data.get("retrieved_chunk_ids", [])) or "none")
    with st.expander("Raw JSON from /ask"):
        st.json(data)


st.set_page_config(page_title="Northwind RAG Demo", layout="centered")
st.title("RAG Demo: ingest + ask")
st.caption("A front end for the FastAPI service. Answers come only from ingested documents.")

with st.sidebar:
    base_url = st.text_input(
        "API base URL", DEFAULT_BASE_URL, help="Your Render URL, or set API_BASE_URL"
    ).rstrip("/")
    if st.button("Check connection"):
        with st.spinner("Calling /health/pinecone (may take a minute if Render is asleep)..."):
            status, data = call_api("GET", f"{base_url}/health/pinecone")
        if status == 200 and isinstance(data, dict):
            st.success(f"Connected. {data.get('total_vector_count', '?')} chunks in Pinecone.")
        else:
            show_error(status, data)

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ask_tab:
    with st.form("ask_form"):
        question = st.text_input("Question", "What is the remote work policy?")
        st.caption("Try one that is not in the docs too, e.g. \"What is the parental leave policy?\"")
        ask_clicked = st.form_submit_button("Ask", type="primary")
    if ask_clicked:
        if not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Calling /ask..."):
                status, data = call_api("POST", f"{base_url}/ask", {"question": question})
            if status == 200 and isinstance(data, dict):
                show_answer(data)
            else:
                show_error(status, data)

with ingest_tab:
    with st.form("ingest_form"):
        document_id = st.text_input(
            "document_id", placeholder="e.g. handbook", help="Letters, digits, '.', '_' and '-'"
        )
        source = st.text_input("Source (optional)", placeholder="e.g. handbook.md")
        text = st.text_area("Document text", height=220, placeholder="Paste the document here")
        ingest_clicked = st.form_submit_button("Ingest", type="primary")
    if ingest_clicked:
        payload = {"text": text, "document_id": document_id}
        if source.strip():
            payload["source"] = source.strip()
        with st.spinner("Calling /ingest (chunk, embed, store)..."):
            status, data = call_api("POST", f"{base_url}/ingest", payload)
        if status == 200 and isinstance(data, dict):
            st.success(
                f"Indexed **{data['document_id']}**: {data['chunks_indexed']} chunk(s) stored."
            )
            if data.get("chunks_removed"):
                st.caption(f"Replaced an earlier version ({data['chunks_removed']} old chunks removed).")
            with st.expander("Raw JSON from /ingest"):
                st.json(data)
        else:
            show_error(status, data)
