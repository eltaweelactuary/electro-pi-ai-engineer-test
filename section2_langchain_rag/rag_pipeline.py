"""
rag_pipeline.py — RAG with Gemini embeddings + Gemini LLM + FAISS
"""
import os, glob
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
SIM_THRESHOLD = 0.3

def load_docs(docs_dir="documents"):
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    if not md_files:
        sd = os.path.dirname(os.path.abspath(__file__))
        md_files = glob.glob(os.path.join(sd, docs_dir, "*.md"))
    docs = []
    for p in md_files:
        with open(p, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(p)}))
    print(f"loaded {len(docs)} docs")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks): c.metadata["chunk_id"] = i
    print(f"{len(chunks)} chunks")
    return chunks

def make_vectorstore(chunks):
    api_key = os.environ.get("GOOGLE_API_KEY")
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    store = FAISS.from_documents(chunks, emb)
    print("FAISS ready")
    return store

RAG_PROMPT = """Answer ONLY from context. If insufficient say:
"I don't have that info. Contact support@quickbite.com."
Cite the source doc name.

Context:
{context}

Question: {question}

Answer:"""

class RAGPipeline:
    def __init__(self, docs_dir="documents"):
        self.docs_dir = docs_dir
        self.vectorstore = None
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def build(self):
        docs = load_docs(self.docs_dir)
        chunks = chunk_docs(docs)
        self.vectorstore = make_vectorstore(chunks)
        print("pipeline ready\n")

    def query(self, question):
        raw = self.vectorstore.similarity_search_with_score(question, k=TOP_K)
        sources = []
        for doc, dist in raw:
            sim = 1.0 / (1.0 + dist)
            if sim >= SIM_THRESHOLD:
                sources.append({"text": doc.page_content, "source": doc.metadata["source"],
                                "chunk_id": doc.metadata["chunk_id"], "sim": round(sim, 3)})
        context = "\n---\n".join(f"[{s['source']}]\n{s['text']}" for s in sources) if sources else "NO RELEVANT CONTEXT."
        resp = self.llm.invoke(self.prompt.format(context=context, question=question))
        return {"question": question, "answer": resp.content, "sources": sources}

    def print_result(self, r):
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")
        if r['sources']:
            print(f"   (from: {', '.join(s['source'] for s in r['sources'])})")
        else:
            print("   (no sources)")
        print()

if __name__ == "__main__":
    print("Run `python run_examples.py`")
