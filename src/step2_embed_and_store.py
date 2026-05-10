import json
import os
from pathlib import Path
from typing import List, Dict
 
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
   
    try:
        from sentence_transformers import SentenceTransformer
        print(f" Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"   Vector size: {model.get_sentence_embedding_dimension()}")
        return model
    except ImportError:
        print(" sentence-transformers not installed")
        print(" Run: pip install sentence-transformers")
        raise
 
def get_vector_store(persist_dir: str = "data/chroma_db",
                     collection_name: str = "ai_books"):

    try:
        import chromadb
        os.makedirs(persist_dir, exist_ok=True)
 
        client = chromadb.PersistentClient(path=persist_dir)
 
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # نوع المسافة: cosine similarity
        )
        print(f"✅ ChromaDB ready → {persist_dir}")
        print(f"   Collection: {collection_name}")
        print(f"   Existing docs: {collection.count()}")
 
        return collection
    except ImportError:
        print(" chromadb not installed")
        print("   Run: pip install chromadb")
        raise
 
def index_chunks(chunks: List[Dict],
                 model,
                 collection,
                 batch_size: int = 64) -> None:
   
    total = len(chunks)
    indexed = 0
 
    print(f"\n Indexing {total} chunks in batches of {batch_size}...")
 
    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
 
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "page": c["page"],
                "source": c["source"],
                "word_count": c["word_count"]
            }
            for c in batch
        ]
 
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
 
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,      # النص الأصلي
            metadatas=metadatas   # page, source, word_count
        )
        indexed += len(batch)
        percent = (indexed / total) * 100
        print(f"   [{percent:5.1f}%] {indexed}/{total} chunks indexed")
 
    print(f"\n Done! Total in DB: {collection.count()}")
 
def build_index(chunks_file: str,
                chroma_dir: str = "data/chroma_db") -> None:
   
    print(f" Loading chunks from: {chunks_file}")
    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Found {len(chunks)} chunks")
 
    model = get_embedding_model()
 
    collection = get_vector_store(persist_dir=chroma_dir)
 
    existing_ids = set()
    if collection.count() > 0:
        # نتجنب الـ duplicate
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
        print(f"   Skipping {len(existing_ids)} already-indexed chunks")
 
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
 
    if not new_chunks:
        print(" All chunks already indexed, nothing to do!")
        return
    print(f"   New chunks to index: {len(new_chunks)}")
   
    index_chunks(new_chunks, model, collection)

if __name__ == "__main__":
    CHUNKS_FILE = "data/Data_rag_chunks.json"
    if os.path.exists(CHUNKS_FILE):
        build_index(CHUNKS_FILE)
    else:
        print(f" Chunks file not found: {CHUNKS_FILE}")
        print("   Run step1_ingest.py first!")