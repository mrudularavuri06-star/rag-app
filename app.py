import fitz  # PyMuPDF
import streamlit as st
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

# -----------------------------
# LOAD MODEL (CACHE)
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Streamlit Cloud uses CPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# -----------------------------
# EMBEDDINGS (CACHE)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# LOAD PDF
# -----------------------------
def load_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    documents = []

    for i, page in enumerate(pdf):
        text = page.get_text()

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1}
                )
            )

    return documents

# -----------------------------
# SPLIT TEXT
# -----------------------------
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# -----------------------------
# CREATE VECTOR DB (CACHED)
# -----------------------------
@st.cache_resource
def create_db_cached(docs):
    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# GENERATE ANSWER (FIXED)
# -----------------------------
def generate_answer(context, query):
    prompt = f"""
Extract information from the scholarship data below.

DATA:
{context}

QUESTION:
{query}

Rules:
- If listing, return only names
- If answer not found, say "Not available"
- Keep answer short and clear

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result.strip() if result.strip() else "⚠️ No answer generated"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Scholarship RAG", layout="wide")

st.title("🎓 Scholarship RAG System")
st.write("Upload a PDF and ask questions")

uploaded_file = st.file_uploader("Upload Scholarship PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded!")

    with st.spinner("Processing PDF..."):
        documents = load_pdf(uploaded_file)

        if len(documents) == 0:
            st.error("No text found in PDF")
        else:
            docs = split_docs(documents)
            db = create_db_cached(docs)

            st.success("Ready! Ask questions below 👇")

            query = st.text_input("Ask your question:")

            if query:
                results = db.similarity_search(query, k=5)
                context = "\n".join([r.page_content for r in results])

                # -----------------------------
                # 🔥 SMART FIX FOR LIST QUERIES
                # -----------------------------
                if "list" in query.lower():
                    lines = context.split("\n")
                    names = []

                    for line in lines:
                        parts = line.split(",")
                        if len(parts) > 1:
                            names.append(parts[1].strip())

                    answer = "\n".join(set(names)) if names else "No scholarships found"

                else:
                    answer = generate_answer(context, query)

                st.subheader("📌 Answer:")
                st.write(answer)

                # 🔍 DEBUG (optional)
                # st.write("Context:", context)