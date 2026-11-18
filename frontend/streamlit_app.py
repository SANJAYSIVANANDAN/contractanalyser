"""DocuPilot AI — minimal Streamlit UI.

Streamlit re-runs this whole script from top to bottom on every click or message.
Anything that must survive between runs lives in st.session_state.
"""
import json
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8001")

st.set_page_config(page_title="DocuPilot AI", layout="wide")


# ------------------------------------------------------------ API helpers
def api_get(path):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def stream_chat(question, document_ids, conversation_id, result):
    """Generator: yields answer tokens one by one (for st.write_stream).
    Reads the SSE stream line by line. When the 'citations'/'done'/'error' events
    arrive, it stores them in the `result` dict so the caller can show them afterwards."""
    body = {"question": question, "document_ids": document_ids, "conversation_id": conversation_id}
    with requests.post(f"{API_URL}/chat/stream", json=body, stream=True, timeout=120) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                if event == "token":
                    yield data["text"]
                else:                      # citations, done, error
                    result[event] = data


def show_clauses(clauses):
    """One expander row per extracted clause: title/type, found/missing, risk, text."""
    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    for clause in clauses:
        label = clause["title"] or clause["clause_type"].replace("_", " ").title()
        if not clause["found"]:
            label += " — not found"
        elif clause["risk_level"]:
            label = f"{risk_colors.get(clause['risk_level'], '')} {label} ({clause['risk_level']} risk)"
        with st.expander(label):
            if clause["found"]:
                st.markdown(clause["summary"] or "")
                if clause["supporting_text"]:
                    st.text(clause["supporting_text"])
                if clause["page_number"] is not None:
                    st.caption(f"Page {clause['page_number']}")
            st.caption(clause["explanation"] or "")


def show_citations(citations):
    """One block per source, exactly the fields the backend validated."""
    for c in citations:
        with st.container(border=True):
            st.markdown(f"**Source {c['source_number']}**")
            st.markdown(f"Filename: `{c['filename']}`")
            st.markdown(f"Page: **{c['page_number'] if c['page_number'] is not None else 'unknown'}**")
            st.markdown(f"Section: {c['section_title'] or 'n/a'}")
            st.markdown("Supporting excerpt:")
            st.text(c["supporting_excerpt"])


# ------------------------------------------------------------ session state
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role", "content", "citations", "warning"}]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# ------------------------------------------------------------ sidebar: upload + documents
with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Upload a contract (PDF)", type=["pdf"])
    if uploaded is not None and st.button("Upload"):
        with st.spinner("Uploading..."):
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=60,
                )
            except requests.RequestException:
                response = None
        if response is None:
            st.error("Could not reach the API")
        elif response.ok:
            st.success("Uploaded. Processing takes about a minute — this page refreshes itself.")
            st.rerun()
        else:
            st.error(response.json().get("detail", "Upload failed"))

    st.button("Refresh status")   # any click re-runs the script, which re-fetches the list

    documents = api_get("/documents")
    if documents is None:
        st.error("API unreachable")
        documents = []

    icons = {"PENDING": "⏳", "PROCESSING": "⚙️", "COMPLETED": "✅", "FAILED": "❌"}
    analysis_icons = {None: "⚪ not analysed", "PENDING": "⏳ queued", "PROCESSING": "⚙️ analysing",
                       "COMPLETED": "📋 analysed", "FAILED": "❌ analysis failed"}

    still_working = any(d["status"] in ("PENDING", "PROCESSING") for d in documents) or any(
        d["analysis_status"] in ("PENDING", "PROCESSING") for d in documents
    )
    if still_working:
        st.caption("⏳ Ingestion/analysis is running — this can take a minute or two. Updating automatically...")

    for doc in documents:
        st.write(f"{icons.get(doc['status'], '?')} {doc['filename']} — {doc['status']}")
        if doc["status"] == "FAILED" and doc["error_message"]:
            st.caption(doc["error_message"][:150])
        if doc["status"] == "COMPLETED":
            col1, col2 = st.columns([2, 1])
            col1.caption(analysis_icons.get(doc["analysis_status"], doc["analysis_status"]))
            if doc["analysis_status"] not in ("PENDING", "PROCESSING"):
                if col2.button("Analyse", key=f"analyze-{doc['id']}"):
                    try:
                        response = requests.post(f"{API_URL}/documents/{doc['id']}/analyze", timeout=30)
                        if response.ok:
                            st.success("Analysis started. Click 'Refresh status' to check progress.")
                        else:
                            st.error(response.json().get("detail", "Could not start analysis"))
                    except requests.RequestException:
                        st.error("Could not reach the API")
            if doc["analysis_status"] == "FAILED" and doc["analysis_error_message"]:
                st.caption(doc["analysis_error_message"][:150])

    completed = {d["filename"]: d["id"] for d in documents if d["status"] == "COMPLETED"}
    # Stable key so this widget keeps the user's selection across reruns — without it,
    # Streamlit derives the key from `options`, so a newly-completed document changes the
    # key, the widget is treated as new, and `default` silently re-selects everything.
    if "search_selection" not in st.session_state:
        st.session_state.search_selection = list(completed)
    else:
        # drop names no longer in `completed` (e.g. the document was deleted) so the
        # widget's stored value never falls outside its current `options`
        st.session_state.search_selection = [n for n in st.session_state.search_selection if n in completed]
    selected_names = st.multiselect(
        "Search in", list(completed), default=st.session_state.search_selection, key="search_selection"
    )
    selected_ids = [completed[name] for name in selected_names]

    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

# ------------------------------------------------------------ main: chat
st.title("DocuPilot AI — Contract Q&A")
st.caption("Answers come only from the selected contracts and cite their source pages. "
           "This is an informational aid, not legal advice.")

if len(selected_ids) == 1:
    selected_doc = next((d for d in documents if d["id"] == selected_ids[0]), None)
    if selected_doc and selected_doc["analysis_status"] == "COMPLETED":
        clauses = api_get(f"/documents/{selected_doc['id']}/clauses")
        with st.expander("📋 Contract Analysis", expanded=True):
            show_clauses(clauses or [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        show_citations(message.get("citations", []))

question = st.chat_input("Ask about the selected contracts...")
if st.button("Summarize selected agreements"):
    question = "Summarize this agreement."

if question:
    if not selected_ids:
        st.warning("Upload a contract and wait for it to show ✅ COMPLETED, then select it.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result = {}
            try:
                answer = st.write_stream(stream_chat(question, selected_ids, st.session_state.conversation_id, result))
            except requests.RequestException as error:
                answer = ""
                result["error"] = {"message": f"Connection problem: {type(error).__name__}"}

            citations_event = result.get("citations", {})
            citations = citations_event.get("sources", [])
            warning = None
            if "error" in result:
                warning = result["error"]["message"]
            elif citations_event.get("insufficient_evidence"):
                warning = ("The available passages did not provide sufficient evidence "
                           "(or the answer could not be backed by a valid citation).")
            if citations_event.get("rejected_sources"):
                warning = (warning or "") + f" Ignored invalid source references: {citations_event['rejected_sources']}."

            if warning:
                st.warning(warning)
            show_citations(citations)

            if "done" in result:
                st.session_state.conversation_id = result["done"]["conversation_id"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations, "warning": warning}
        )

# Auto-refresh last, after everything (chat included) has rendered, so a background
# ingestion/analysis job never preempts an in-progress chat interaction.
if still_working:
    time.sleep(3)
    st.rerun()
