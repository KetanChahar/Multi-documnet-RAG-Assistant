"""
app.py
------
Streamlit UI. Built LAST, on top of ingest.py and rag_chain.py, which should
already work standalone before you run this.

Run with:
    streamlit run app.py
"""

import os
import shutil
import streamlit as st

from ingest import ingest_files, VECTORSTORE_DIR

st.set_page_config(page_title="Chat with your Documents", page_icon="📄")
st.title("📄 Chat with your Documents")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Session state setup -----------------------------------------------
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists(VECTORSTORE_DIR)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()  # filenames already embedded, avoid re-ingesting

# --- Sidebar: upload documents (multiple at once) ------------------------
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Only ingest files we haven't already embedded in this session
        new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_files]

        if new_files:
            new_paths = []
            for f in new_files:
                save_path = os.path.join(UPLOAD_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                new_paths.append(save_path)

            with st.spinner(f"Reading, chunking, and embedding {len(new_paths)} document(s)..."):
                ingest_files(new_paths)
                st.session_state.vectorstore_ready = True
                st.session_state.ingested_files.update(f.name for f in new_files)

            st.success(f"{len(new_files)} document(s) ready to query!")

    if st.session_state.ingested_files:
        st.caption("Currently indexed: " + ", ".join(sorted(st.session_state.ingested_files)))

    # --- Reset button: wipe all memory ---
    st.divider()
    if st.button("🗑️ Reset Everything", use_container_width=True):
        # 1. Clear chat history
        st.session_state.chat_history = []
        st.session_state.ingested_files = set()
        st.session_state.vectorstore_ready = False

        # 2. Delete the FAISS vectorstore from disk
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)

        # 3. Delete uploaded files
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 4. Reset the cached retriever so it doesn't serve stale data
        import rag_chain
        rag_chain.retriever = None

        st.success("All history and documents have been cleared!")
        st.rerun()

# --- Main: ask questions --------------------------------------------------
if not st.session_state.vectorstore_ready:
    st.info("Upload a document (.pdf or .txt) from the sidebar to get started.")
else:
    # Import here, not at top of file: rag_chain.py loads the vectorstore at
    # import time, which only exists once a document has been ingested.
    from rag_chain import ask

    # Show conversation history (oldest first, like a chat)
    for q, a, sources, context in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            with st.expander("Show retrieved chunks (sources)"):
                for src, chunk in zip(sources, context):
                    st.markdown(f"**{src}**")
                    st.text(chunk)

    # chat_input auto-clears after the user presses Enter
    question = st.chat_input("Ask a question about your document(s)...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = ask(question)
            st.markdown(result["answer"])
            with st.expander("Show retrieved chunks (sources)"):
                for src, chunk in zip(result["sources"], result["context"]):
                    st.markdown(f"**{src}**")
                    st.text(chunk)

        st.session_state.chat_history.append(
            (question, result["answer"], result["sources"], result["context"])
        )
