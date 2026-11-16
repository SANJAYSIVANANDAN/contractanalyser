"""DocuPilot AI — minimal Streamlit UI.

Streamlit re-runs this whole script from top to bottom on every click or message.
Anything that must survive between runs lives in st.session_state.
"""
import json
import os

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
        try:
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=60,
            )
            if response.ok:
                st.success("Uploaded. Processing in the background...")
            else:
                st.error(response.json().get("detail", "Upload failed"))
        except requests.RequestException:
            st.error("Could not reach the API")

    st.button("Refresh status")   # any click re-runs the script, which re-fetches the list

    documents = api_get("/documents")
    if documents is None:
        st.error("API unreachable")
        documents = []

    icons = {"PENDING": "⏳", "PROCESSING": "⚙️", "COMPLETED": "✅", "FAILED": "❌"}
    for doc in documents:
        st.write(f"{icons.get(doc['status'], '?')} {doc['filename']} — {doc['status']}")
        if doc["status"] == "FAILED" and doc["error_message"]:
            st.caption(doc["error_message"][:150])

    completed = {d["filename"]: d["id"] for d in documents if d["status"] == "COMPLETED"}
    selected_names = st.multiselect("Search in", list(completed), default=list(completed))
    selected_ids = [completed[name] for name in selected_names]

    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

# ------------------------------------------------------------ main: chat
st.title("DocuPilot AI — Contract Q&A")
st.caption("Answers come only from the selected contracts and cite their source pages. "
           "This is an informational aid, not legal advice.")

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
