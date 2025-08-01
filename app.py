from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import normalize
import re

# ------------------ Setup ------------------
load_dotenv()
app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEAM_TOKEN = "d1b791fa0ef5092d9cd051b2b09df2473d1e2ea07e09fe6c61abb5722dfbc7d3"
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ PDF Processing ------------------
def extract_pages_from_pdf_url(url):
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    return [page.get_text() for page in doc]

# ------------------ Chunking ------------------
def chunk_text_with_pages(text_by_page):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    for page_num, page_text in enumerate(text_by_page, 1):
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "page": page_num})
    return all_chunks

# ------------------ Embedding + FAISS ------------------
def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    norm_embeddings = normalize(np.array(embeddings), axis=1)
    index = faiss.IndexFlatIP(norm_embeddings.shape[1])
    index.add(norm_embeddings)
    return index, chunks, norm_embeddings

# ------------------ Retrieval ------------------
def retrieve_top_chunks(question, index, chunks, k=7):
    q_vec = normalize(model.encode([question]), axis=1)
    scores, indices = index.search(q_vec, k)
    top_matches = [{"text": chunks[i]["text"], "page": chunks[i]["page"], "score": scores[0][j]}
                   for j, i in enumerate(indices[0])]
    
    keywords = set(question.lower().split())
    filtered = [chunk for chunk in top_matches if any(word in chunk["text"].lower() for word in keywords)]
    return filtered or top_matches[:5]

# ------------------ Prompting ------------------
def build_prompt(question, context_chunks):
    context = "\n---\n".join([c["text"] for c in context_chunks])
    return f"""
You are an expert insurance assistant. Use only the facts from the provided context to answer the question. Do not infer or assume anything.

For financial or legal clauses, always provide the complete clause wording if available.
For financial clauses, always extract any limits, exceptions, or conditions if mentioned.

EXAMPLE:
Question: What is the grace period for premium payment?
Answer: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

CONTEXT:
{context}

QUESTION:
{question}

Respond in this exact JSON format:
{{ "answer": "..." }}
"""

# ------------------ LLM Call ------------------
import re
import json

def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")

    content = response.json()["choices"][0]["message"]["content"]
    print("LLM raw response:", content)  # Debug

    # Match only the first full JSON object using balanced braces
    brace_stack = []
    start_idx = end_idx = -1

    for i, ch in enumerate(content):
        if ch == '{':
            if not brace_stack:
                start_idx = i
            brace_stack.append(ch)
        elif ch == '}':
            brace_stack.pop()
            if not brace_stack:
                end_idx = i + 1
                break

    if start_idx != -1 and end_idx != -1:
        json_str = content[start_idx:end_idx].replace('\n', ' ').replace('\r', ' ')
        try:
            return json.loads(json_str).get("answer", "Not specified in the document.")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding failed: {e}")
    else:
        raise ValueError("No valid JSON object found in LLM response.")


# ------------------ Main Endpoint ------------------
@app.route("/api/v1/hackrx/run", methods=["POST"])
def run_submission():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != TEAM_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        document_url = data.get("documents")
        questions = data.get("questions")

        if not document_url or not questions:
            return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

        text_by_page = extract_pages_from_pdf_url(document_url)
        chunk_dicts = chunk_text_with_pages(text_by_page)
        index, chunks, _ = embed_chunks(chunk_dicts)

        def process(q):
            top_chunks = retrieve_top_chunks(q, index, chunks)
            prompt = build_prompt(q, top_chunks)
            return call_groq(prompt)

        with ThreadPoolExecutor(max_workers=5) as executor:
            answers = list(executor.map(process, questions))

        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
