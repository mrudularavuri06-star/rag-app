import fitz
import streamlit as st
import torch
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "scholarshipdata111.pdf")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# PDF LOADER (AUTO)
# -----------------------------
def load_pdf():
    docs = []

    if not os.path.exists(PDF_PATH):
        st.error(f"❌ File not found at: {PDF_PATH}")
        return docs

    try:
        pdf = fitz.open(PDF_PATH)
    except Exception as e:
        st.error(f"❌ Error opening PDF: {e}")
        return docs

    for i, page in enumerate(pdf):
        text = page.get_text()
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": f"Page {i+1}"}
                )
            )

    return docs

# -----------------------------
# WEBSITE LOADER
# -----------------------------
def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = url

    return docs

# -----------------------------
# SPLIT + VECTOR DB
# -----------------------------
def create_db(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# ANSWER ENGINE
# -----------------------------
def generate_answer(context, query, mode):
    q = query.lower()

    # 📄 PDF LOGIC
    if mode == "📄 PDF":
        if "scholarship" in q:
            lines = context.split("\n")
            results = []

            for line in lines:
                parts = [p.strip() for p in line.split(",")]

                if len(parts) >= 8:
                    try:
                        name = parts[1]
                        category = parts[2]
                        level = parts[3]
                        amount = parts[6]
                        deadline = parts[7]

                        if "sc" in q and "sc" not in category.lower():
                            continue
                        if "st" in q and "st" not in category.lower():
                            continue
                        if "obc" in q and "obc" not in category.lower():
                            continue

                        results.append(
                            f"{name} (Category: {category}, Level: {level}, Amount: ₹{amount}, Deadline: {deadline})"
                        )
                    except:
                        continue

            return "\n\n".join(results[:5]) if results else "Not available"

    # 🌐 WEBSITE QUICK ANSWER
    if mode == "🌐 Website":
        if "criteria" in q:
            return "Scholarships are awarded based on merit, financial need, athletic ability, or other criteria."

    # 🤖 FALLBACK MODEL
    prompt = f"""
Context:
{context}

Question:
{query}

Answer briefly using only the context.
If not found, say: Not available.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return result if result else "Not available"

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RAG System", layout="wide")

st.title("🔍 RAG System (PDF + Website)")

mode = st.radio("Select Source:", ["📄 PDF", "🌐 Website"])

documents = None

# -----------------------------
# PDF MODE
# -----------------------------
if mode == "📄 PDF":
    st.info("Using preloaded PDF")

    documents = load_pdf()

    if not documents:
        st.error("❌ PDF loaded but no readable text found. Use a text-based PDF.")
        st.stop()

# -----------------------------
# WEBSITE MODE
# -----------------------------
if mode == "🌐 Website":
    url = st.text_input("Enter Website URL:")

    if url:
        with st.spinner("Loading website..."):
            try:
                documents = load_website(url)
            except:
                st.error("❌ Failed to load website")
                st.stop()

# -----------------------------
# QUERY
# -----------------------------
if documents:
    db = create_db(documents)

    st.success("Ready! Ask your question")

    query = st.text_input("Ask question:")

    if query:
        with st.spinner("Searching..."):
            results = db.similarity_search(query, k=5)

            if not results:
                st.write("Not available")
                st.stop()

            context = "\n".join([r.page_content for r in results])
            answer = generate_answer(context, query, mode)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Source:")
            sources = list(set([r.metadata["source"] for r in results]))
            for s in sources:
                st.write(f"- {s}")

