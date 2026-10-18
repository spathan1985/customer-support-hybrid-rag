import time
from pathlib import Path

import streamlit as st

from generate_answer import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHome Support Assistant",
    page_icon="💬",
    layout="wide"
)

st.title("💬 SmartHome Support Assistant")

st.caption(
    "Hybrid RAG powered by Nebius, Pinecone, "
    "BM25, RRF, and reranking"
)

st.info(
    "Demo application using fictional products and "
    "synthetic support documentation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    enable_threshold = st.toggle(
        "Enable relevance threshold",
        value=True
    )

    threshold = st.slider(
        "Minimum reranker score",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        disabled=not enable_threshold
    )

    enable_error_check = st.toggle(
        "Validate error codes",
        value=True
    )

    st.divider()

    st.subheader("Retrieval Pipeline")

    st.markdown(
        """
        **Embeddings:** Nebius Qwen3-Embedding-8B

        **Vector Search:** Pinecone

        **Keyword Search:** BM25

        **Fusion:** Reciprocal Rank Fusion

        **Reranker:** Pinecone BGE

        **Generation:** Nebius Qwen3-30B-A3B-Instruct
        """
    )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HELPER: DISPLAY SOURCES
# ============================================================

def display_sources(sources):

    if not sources:
        return

    with st.expander(
        f"📚 View retrieved sources ({len(sources)})"
    ):

        for i, doc in enumerate(sources, start=1):

            source_name = Path(
                doc.get("source", "Unknown")
            ).name

            st.markdown(
                f"**[Source {i}] {source_name}**"
            )

            st.caption(
                f"Document type: "
                f"{doc.get('document_type', 'Unknown')}"
            )

            score = doc.get("rerank_score")

            if score is not None:
                st.caption(
                    f"Reranker score: {score:.4f}"
                )

            st.text(doc.get("text", ""))

            st.divider()


# ============================================================
# HELPER: DISPLAY ASSISTANT RESPONSE
# ============================================================

def display_response(message):

    if message["status"] == "escalated":

        st.warning(message["answer"])

        if message.get("reason"):
            st.caption(
                f"Escalation reason: {message['reason']}"
            )

    else:
        st.markdown(message["answer"])

    if message.get("latency") is not None:
        st.caption(
            f"Response time: {message['latency']:.2f}s"
        )

    display_sources(message.get("sources", []))


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":
            st.markdown(message["content"])

        else:
            display_response(message)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Describe your issue or ask a product question..."
)

if question:

    # Save and display customer question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching support documentation..."
        ):

            start = time.perf_counter()

            try:

                result = generate_answer(
                    question=question,
                    enable_threshold=enable_threshold,
                    min_rerank_score=threshold,
                    enable_error_code_check=enable_error_check
                )

                latency = (
                    time.perf_counter() - start
                )

                assistant_message = {
                    "role": "assistant",
                    "answer": result["answer"],
                    "status": result["status"],
                    "reason": result.get("reason"),
                    "sources": result.get("sources", []),
                    "latency": latency
                }

                st.session_state.messages.append(
                    assistant_message
                )

                display_response(assistant_message)

            except Exception as e:

                st.error(
                    "Unable to process your request. "
                    "Please try again."
                )

                # Keep details out of the customer UI.
                # In local development, inspect the
                # terminal traceback for debugging.
                import traceback
                traceback.print_exc()