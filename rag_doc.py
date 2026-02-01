import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# 1. Load Environment Variables
load_dotenv()

# --- Configuration ---
PDF_PATH = "2509.03680v1.pdf"  # Ensure this file is in your project folder
INDEX_NAME = "luxdit-paper-index"
EMBEDDING_MODEL = "models/text-embedding-004" # Google's model (768 dimensions)

def main():
    # 2. Load the PDF
    print(f"📄 Loading PDF: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")

    # 3. Chunking Strategy (Type 5: Recursive)
    # We use a large chunk size with overlap to maintain context across page breaks
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""] # Try paragraphs first, then lines, then words
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks.")

    # 4. Initialize Embeddings (Google Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 5. Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # Check if index exists, create if not
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        print(f"🏗️  Creating Pinecone Index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # Matches Google's embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(2) # Wait for index initialization
    else:
        print(f"✅ Found existing index: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)

    # 6. Store in Vector Database
    # We use .from_documents to automatically handle the text and metadata
    print("💾 Upserting vectors to Pinecone (this may take a moment)...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("✅ Upload complete!")

    # 7. The Interface (Simple Loop)
    print("\n" + "="*50)
    print("🤖 LUXDIT PAPER AI ASSISTANT")
    print("Ask a question about the paper (or type 'exit' to quit)")
    print("="*50 + "\n")

    while True:
        user_query = input("User: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break

        # Search Pinecone
        results = vector_store.similarity_search_with_score(user_query, k=3)

        print("\n--- Top Relevant Snippets ---")
        for i, (doc, score) in enumerate(results):
            # Show the score and the page number from metadata
            page_num = doc.metadata.get('page', 'Unknown')
            print(f"[Result {i+1}] (Score: {score:.3f} | Page: {page_num})")
            print(f"\"{doc.page_content[:250]}...\"\n") # Show first 250 chars
        print("-" * 50)

if __name__ == "__main__":
    main()