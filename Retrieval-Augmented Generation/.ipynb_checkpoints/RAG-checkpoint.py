import streamlit as st
import PyPDF2
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def make_chunks(text, chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk,model="text-embedding-3-small")
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_chunks(question, chunks, index, k=2):
    q_embed = client.embeddings.create(input=question,model="text-embedding-3-small").data[0].embedding
    q_embed = np.array([q_embed]).astype("float32")
    distance, indices = index.search(q_embed, k)
    return [chunks[i] for i in indices[0]]

def ask_question(chunks, question):
    context = "\n\n".join(chunks)
    prompt = f"Answer the question based on this text:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="PDF Q&A", layout="centered")

st.title("📄 PDF Question Answering System")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = read_pdf(uploaded_file)
        chunks = make_chunks(text)
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Searching for answer..."):
            best_chunks = search_chunks(question, chunks, index)
            answer = ask_question(best_chunks, question)

        st.subheader("🧠 Answer")
        st.write(answer)