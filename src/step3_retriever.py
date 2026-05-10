"""
STEP 3 — Retriever
====================
المهمة: تاخد سؤال وترجع أقرب chunks
 
كيف بيشتغل الـ Retrieval؟
1. المستخدم يكتب سؤال
2. نحول السؤال لـ embedding (نفس الـ model اللي استخدمناه في الـ indexing)
3. نحسب الـ cosine similarity بين embedding السؤال وكل chunk
4. نرجع الـ top-k chunks الأقرب
 
Cosine Similarity:
- 1.0 = متطابقان تماماً
- 0.0 = لا علاقة بينهم
- -1.0 = معاكسان
 
مثال على الـ top-k:
query: "How does B-Tree work?"
top-3:
  0.92 - "B-Trees store data sorted by key..."         ← page 83
  0.87 - "B-Tree nodes split when they overflow..."    ← page 84
  0.81 - "Log-structured merge trees vs B-Trees..."   ← page 91
"""
 
from typing import List, Dict, Optional
 
 
# ==============================================================
# الـ Retriever Class
# ==============================================================
 
class Retriever:
    def __init__(self,
                 chroma_dir: str = "data/chroma_db",
                 collection_name: str = "ai_books",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        يحمّل الـ model والـ collection مرة واحدة عند الإنشاء
        (مش بيعيد التحميل عند كل query)
        """
        # Lazy import عشان ما يفشلش لو المكتبات مش متاحة
        from sentence_transformers import SentenceTransformer
        import chromadb
 
        print("🔍 Loading retriever...")
        self.model = SentenceTransformer(model_name)
        print("model  loaded !")
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
 
        print(f"   ✅ Ready — {self.collection.count()} chunks in index")
 
    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 source_filter: Optional[str] = None) -> List[Dict]:
        """
        يبحث بالمعنى ويرجع أقرب chunks
 
        Args:
            query:         السؤال بالإنجليزي (أو عربي، المودل بيفهم)
            top_k:         كام نتيجة ترجعلنا (عادةً 3-7)
            source_filter: لو عايز تفلتر كتاب معين بالاسم
 
        Returns:
            [{"text": ..., "page": ..., "source": ..., "score": ...}, ...]
        """
        # 1. حول السؤال لـ embedding
        query_embedding = self.model.encode(query).tolist()
 
        # 2. ابني الـ where filter لو محتاج
        where = None
        if source_filter:
            where = {"source": source_filter}
 
        # 3. ابحث في ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
 
        # 4. حوّل النتائج لـ format نظيف
        chunks = []
        for i in range(len(results["ids"][0])):
            # ChromaDB بيرجع distance مش similarity
            # cosine distance = 1 - cosine similarity
            distance = results["distances"][0][i]
            similarity = 1 - distance  # نحوّله لـ similarity (أكبر = أحسن)
 
            chunks.append({
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"],
                "source": results["metadatas"][0][i]["source"],
                "score": round(similarity, 4)
            })
 
        return chunks
 
    def print_results(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Helper: يجيب النتائج ويطبعها بشكل قابل للقراءة
        """
        print(f"\n🔎 Query: \"{query}\"")
        print("=" * 60)
 
        results = self.retrieve(query, top_k=top_k)
 
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] Score: {chunk['score']:.4f}")
            print(f"    Source: {chunk['source']} | Page: {chunk['page']}")
            print(f"    Text: {chunk['text'][:300]}...")
 
        return results
 
 
# ==============================================================
# تجربة مباشرة
# ==============================================================
if __name__ == "__main__":
    retriever = Retriever()
 
    # أمثلة على queries من DDIA
    test_queries = [
        "How does B-Tree indexing work?",
        "What is the difference between OLTP and OLAP?",
        "How does Kafka handle fault tolerance?",
        "Explain CAP theorem",
        "What are the challenges of distributed transactions?",
    ]
 
    for query in test_queries:
        retriever.print_results(query, top_k=3)
        print("\n" + "─" * 60)