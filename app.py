import json
import io
from typing import List, Tuple

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def load_model():
    # Lightweight, fast, high-quality sentence embedding model (free)
    return SentenceTransformer("all-MiniLM-L6-v2")

def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise for cosine similarity with FAISS (IndexFlatIP)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def build_index(model: SentenceTransformer, questions: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build a FAISS inner-product (cosine) index from questions."""
    q_emb = model.encode(questions, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
    q_emb = normalize(q_emb.astype("float32"))
    dim = q_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if inputs are normalized
    index.add(q_emb)
    return index, q_emb

def search(index: faiss.IndexFlatIP, model: SentenceTransformer, query: str, k: int = 3):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    q = normalize(q)
    scores, idxs = index.search(q, k)
    return scores[0], idxs[0]

def default_qa() -> List[Tuple[str, str]]:
    """Built-in demo KB. Replace/extend later as needed."""
    return [
        # Workspace / onboarding
        ("How do I set up my workspace?",
         "Follow the below doc"),

        # Processes / ticketing
        ("How do I raise a ticket for data patching?",
         "Send a mail to CatalystDB team"),

        # kubectl
        ("Common kubectl commands",
         "List pods: `kubectl get pods -n <ns>`\nDescribe pod: `kubectl describe pod <name> -n <ns>`\n"),

        # git
        ("Common git commands",
         "`git clone <url>` → clone\n`git checkout -b feature/x` → new branch\n"
         "`git add . && git commit -m \"msg\" && git push` → commit & push\n"),

        # Troubleshooting
        ("Service is not connecting to the interface",
         "Check service/pod status, container logs. "
         "Confirm readiness/liveness probes issue."),
    ]


def export_kb_json(pairs: List[Tuple[str, str]]) -> bytes:
    data = [{"question": q, "answer": a} for q, a in pairs]
    return json.dumps(data, indent=2).encode("utf-8")

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Q&A Chatbot – Knowledge Companion", page_icon="💬")
st.title("💬 Q&A Chatbot – Knowledge Companion (Prototype)")

# Sidebar: info + KB management
with st.sidebar:
    st.subheader("About this prototype")
    st.write(
        "- 100% free: Sentence-Transformers + FAISS\n"
        "- Retrieval-based (no external APIs)\n"
        "- Export KB to JSON"
    )

    # Initialize state
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = default_qa()
    if "model" not in st.session_state:
        st.session_state.model = load_model()
    if "index" not in st.session_state:
        # build initial index
        qs = [q for q, _ in st.session_state.qa_pairs]
        st.session_state.index, st.session_state._emb = build_index(st.session_state.model, qs)

    st.divider()
    st.subheader("➕ Add a Q&A")
    with st.form("add_qa_form", clear_on_submit=True):
        new_q = st.text_input("Question")
        new_a = st.text_area("Answer", height=120)
        submitted = st.form_submit_button("Add to knowledge base")
        if submitted:
            if new_q.strip() and new_a.strip():
                st.session_state.qa_pairs.append((new_q.strip(), new_a.strip()))
                # rebuild index
                qs2 = [q for q, _ in st.session_state.qa_pairs]
                st.session_state.index, st.session_state._emb = build_index(st.session_state.model, qs2)
                st.success("Added! Index updated.")
            else:
                st.warning("Please provide both a question and an answer.")

    st.divider()
    st.subheader("⬇️ Export KB")
    kb_bytes = export_kb_json(st.session_state.qa_pairs)
    st.download_button("Download knowledge_base.json", data=kb_bytes, file_name="knowledge_base.json", mime="application/json")

st.write("Ask anything about workspace setup, ticketing, kubectl, git, or common troubleshooting:")

# Search settings
col_a, col_b = st.columns([1, 1])
with col_a:
    k = st.slider("Number of suggestions (k)", min_value=1, max_value=5, value=3)
with col_b:
    threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05,
                          help="Only show answers with cosine similarity ≥ threshold")

query = st.text_input("Your question")

if query:
    scores, idxs = search(st.session_state.index, st.session_state.model, query, k=k)
    results_shown = False

    for rank, (s, i) in enumerate(zip(scores, idxs), start=1):
        if i < 0:
            continue
        if s < threshold:
            continue
        q, a = st.session_state.qa_pairs[i]
        with st.container(border=True):
            st.markdown(f"**Top match #{rank}**  \n"
                        f"**Matched Question:** {q}  \n"
                        f"**Confidence:** {s:.2f}")
            st.markdown("**Answer:**")
            st.write(a)
        results_shown = True

    if not results_shown:
        st.info(
            "I couldn't find a confident match. "
            "Try rephrasing your question or lower the confidence threshold."
        )
