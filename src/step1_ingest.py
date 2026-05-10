import fitz  # PyMuPDF
import json
import os
from pathlib import Path
from typing import List, Dict
 
def load_pdf(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    book_name = Path(pdf_path).stem  
    print(f"Loading: {book_name}")
    print(f"Total pages: {len(doc)}")
 
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") 
        if len(text.strip()) < 50:
            continue
        pages.append({
            "page": page_num + 1,     
            "text": text.strip(),
            "source": book_name
        })
 
    doc.close()
    print(f"   Extracted {len(pages)} non-empty pages")
    return pages
 
def chunk_pages(pages: List[Dict],
                chunk_size: int = 500,
                overlap: int = 50) -> List[Dict]:

    chunks = []
    chunk_id = 0
 
    for page_data in pages:
        words = page_data["text"].split()
        if len(words) <= chunk_size:
            chunks.append({
                "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_id}",
                "text": page_data["text"],
                "page": page_data["page"],
                "source": page_data["source"],
                "word_count": len(words)
            })
            chunk_id += 1
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
 
            chunks.append({
                "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_id}",
                "text": chunk_text,
                "page": page_data["page"],
                "source": page_data["source"],
                "word_count": len(chunk_words)
            })
 
            chunk_id += 1
            start += (chunk_size - overlap)
            if end == len(words):
                break
 
    return chunks
 
def process_book(pdf_path: str, output_dir: str = "data") -> str:
    # Load
    pages = load_pdf(pdf_path)
    # Chunk
    print(f"\nChunking (500 words, 50 overlap)...")
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    print(f"   Created {len(chunks)} chunks")
 
    os.makedirs(output_dir, exist_ok=True)
    book_name = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{book_name}_chunks.json")
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
 
    print(f"\n Saved {len(chunks)} chunks → {output_path}")
 
    # Stats
    avg_words = sum(c["word_count"] for c in chunks) / len(chunks)
    print(f"   Avg chunk size: {avg_words:.0f} words")
 
    return output_path
 
if __name__ == "__main__":
    PDF_PATH = "data/books/Data_rag.pdf"

    if os.path.exists(PDF_PATH):
        output_file = process_book(PDF_PATH, output_dir="data")
 
        with open(output_file, encoding="utf-8") as f:
            chunks = json.load(f)
 
        print("\nExample chunk:")
        print("-" * 60)
        first = chunks[0]
        print(f"  ID:     {first['chunk_id']}")
        print(f"  Page:   {first['page']}")
        print(f"  Words:  {first['word_count']}")
        print(f"  Text:   {first['text'][:200]}...")
    else:
        print(f"File not found: {PDF_PATH}")
        
 