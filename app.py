import os
import json

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Configuration
# -----------------------------

INDEX_FOLDER = "faiss_index"
INDEX_PATH = os.path.join(INDEX_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(INDEX_FOLDER, "metadata.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

TOP_K = 4  # number of chunks to retrieve per question


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Daraz AI Assistant",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Daraz AI Assistant")
st.caption("Ask about returns, delivery, refunds, sellers, payments, or customer support.")


# -----------------------------
# Load resources (cached so they only load once)
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None
    return faiss.read_index(INDEX_PATH)


@st.cache_resource
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


embedding_model = load_embedding_model()
faiss_index = load_faiss_index()
metadata = load_metadata()
groq_client = load_groq_client()


# -----------------------------
# Startup checks
# -----------------------------

if faiss_index is None or not metadata:
    st.error(
        "FAISS index or metadata not found. "
        "Make sure `faiss_index/index.faiss` and `faiss_index/metadata.json` "
        "exist in your repository (run ingest.py first)."
    )
    st.stop()

if groq_client is None:
    st.error(
        "GROQ_API_KEY not found. Add it to Streamlit Cloud Secrets as:\n\n"
        'GROQ_API_KEY = "your_api_key_here"'
    )
    st.stop()


# -----------------------------
# Sidebar: department filter (optional)
# -----------------------------

departments = sorted(set(doc["department"] for doc in metadata))

with st.sidebar:
    st.header("Filters")
    selected_departments = st.multiselect(
        "Search only in these departments",
        options=departments,
        default=departments
    )
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=TOP_K)


# -----------------------------
# Retrieval function
# -----------------------------

def retrieve_chunks(question, k=TOP_K, allowed_departments=None):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding, dtype="float32")

    # Over-fetch a bit so filtering by department still leaves enough results
    search_k = min(len(metadata), max(k * 4, k))
    scores, indices = faiss_index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        doc = metadata[idx]

        if allowed_departments is not None and doc["department"] not in allowed_departments:
            continue

        results.append({
            "score": float(score),
            "department": doc["department"],
            "source_file": doc["source_file"],
            "chunk_number": doc["chunk_number"],
            "text": doc["text"]
        })

        if len(results) >= k:
            break

    return results


# -----------------------------
# Answer generation
# -----------------------------

def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} | Department: {chunk['department']} | File: {chunk['source_file']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(question, chunks):
    context = build_context(chunks)

    system_prompt = (
        "You are the Daraz customer support AI assistant. "
        "Answer the user's question using ONLY the information in the provided context. "
        "If the context does not contain the answer, say you don't have that information "
        "and suggest contacting Daraz customer support. "
        "Be concise and clear. Mention the relevant department when helpful."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input("Ask a question, e.g. 'What is the return policy?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(
                question,
                k=top_k,
                allowed_departments=selected_departments if selected_departments else None
            )

        if not chunks:
            answer = (
                "I couldn't find relevant information for that question in the "
                "selected departments. Please try rephrasing or selecting more departments."
            )
            st.markdown(answer)
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks)

            st.markdown(answer)

            with st.expander("Sources used"):
                for c in chunks:
                    st.markdown(
                        f"**Department:** {c['department']} &nbsp;|&nbsp; "
                        f"**File:** {c['source_file']} &nbsp;|&nbsp; "
                        f"**Chunk:** {c['chunk_number']} &nbsp;|&nbsp; "
                        f"**Score:** {c['score']:.3f}"
                    )
                    st.caption(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
