"""3 example queries through the RAG pipeline."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from rag_pipeline import RAGPipeline

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set."); sys.exit(1)
    pipe = RAGPipeline(docs_dir="documents")
    pipe.build()
    questions = [
        "My order came 45 minutes late. Can I get a refund?",
        "What's the delivery fee for Maadi and when do you operate?",
        "What tech stack is QuickBite built on?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"{'='*50}\n  Example {i}\n{'='*50}")
        pipe.print_result(pipe.query(q))

if __name__ == "__main__":
    main()
