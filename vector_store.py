"""
vector_store.py

Simple RAG implementation using sentence-transformers + FAISS.
- Loads policies from policies.jsonl (one Q&A per line)
- Embeds QUESTIONS (for question-to-question matching)
- Returns ANSWERS (the actual policy information)
- Caches embeddings and index to disk for fast reload

Usage:
    store = PolicyVectorStore()
    store.build_index()  # First time: creates embeddings
    results = store.retrieve(user_query, top_k=3)
"""
import os
import json
import numpy as np
import time

# Optional imports with fallback
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _HAVE_RAG = True
except Exception as e:
    _HAVE_RAG = False
    _RAG_IMPORT_ERROR = e


def _ensure_rag_available():
    if not _HAVE_RAG:
        raise ImportError(
            "RAG dependencies not available. Install with: pip install sentence-transformers faiss-cpu\n"
            f"Underlying error: {_RAG_IMPORT_ERROR}"
        )

# Helper: Load JSONL file
def load_jsonl(path):
    """Load JSONL file - one JSON object per line"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class PolicyVectorStore:
    """
    Simple RAG store using JSONL format
    - Embeds QUESTIONS for similarity search
    - Returns ANSWERS as results
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        _ensure_rag_available()
        self.model_name = model_name
        self.model = None  # Lazy load
        self.index = None
        self.questions = []  # List of questions (what we search)
        self.answers = []    # List of answers (what we return)
        self.sections = []   # List of section names (metadata)
        self.embeddings = None
        
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded")
    
    def build_index(self, policies_file: str = 'policies.jsonl', cache_dir: str = 'vector_index'):
        """Build FAISS index from policies.jsonl - embeds QUESTIONS, returns ANSWERS"""
        _ensure_rag_available()
        
        if not os.path.exists(policies_file):
            raise FileNotFoundError(f"Policies file not found: {policies_file}")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Paths for cached data
        embeddings_path = os.path.join(cache_dir, 'embeddings.npz')
        index_path = os.path.join(cache_dir, 'policy_index.index')
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        
        # Check if cached version exists
        if os.path.exists(embeddings_path) and os.path.exists(index_path) and os.path.exists(metadata_path):
            print("Found cached embeddings and index, loading...")
            self.embeddings = np.load(embeddings_path)['embeddings']
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.questions = metadata['questions']
                self.answers = metadata['answers']
                self.sections = metadata['sections']
            print(f"✅ Loaded cached index with {len(self.questions)} Q&A pairs")
            return
        
        # Load policies from JSONL
        print(f"Loading policies from {policies_file}...")
        policy_data = load_jsonl(policies_file)
        
        self.questions = []
        self.answers = []
        self.sections = []
        
        for item in policy_data:
            if 'question' in item and 'answer' in item:
                self.questions.append(item['question'])
                self.answers.append(item['answer'])
                self.sections.append(item.get('section', 'Unknown'))
        
        if not self.questions:
            raise ValueError("No Q&A pairs found in policies file")
        
        print(f"Found {len(self.questions)} Q&A pairs")
        
        # Create embeddings for QUESTIONS (not answers!)
        # This enables question-to-question similarity matching
        self._load_model()
        print("Creating embeddings for questions...")
        start_time = time.time()
        
        # Add "passage:" prefix for better embedding quality
        question_passages = [f"passage: {q}" for q in self.questions]
        self.embeddings = self.model.encode(
            question_passages,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity
        self.embeddings = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
        
        elapsed = time.time() - start_time
        print(f"✅ Created embeddings with shape {self.embeddings.shape} in {elapsed:.2f}s")
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings.astype('float32'))
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        
        # Save to cache
        np.savez_compressed(embeddings_path, embeddings=self.embeddings)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'questions': self.questions,
                'answers': self.answers,
                'sections': self.sections
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Index cached to {cache_dir}/")
    
    def load_index(self, cache_dir: str = 'vector_index'):
        """Load pre-built index from cache"""
        _ensure_rag_available()
        
        embeddings_path = os.path.join(cache_dir, 'embeddings.npz')
        index_path = os.path.join(cache_dir, 'policy_index.index')
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        
        if not all(os.path.exists(p) for p in [embeddings_path, index_path, metadata_path]):
            raise FileNotFoundError(f"Index files not found in {cache_dir}/. Run build_index() first.")
        
        print(f"Loading index from {cache_dir}/...")
        self.embeddings = np.load(embeddings_path)['embeddings']
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.questions = metadata['questions']
            self.answers = metadata['answers']
            self.sections = metadata['sections']
        print(f"✅ Loaded index with {len(self.questions)} Q&A pairs")
    
    def save_index(self, cache_dir: str = 'vector_index'):
        """Alias for compatibility - build_index already saves"""
        print(f"Index is already saved to {cache_dir}/")
    
    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve top_k most relevant answers by matching query to questions"""
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        self._load_model()
        
        # Encode query with "query:" prefix for better search
        query_with_prefix = f"query: {query}"
        query_vec = self.model.encode([query_with_prefix], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vec.astype('float32'), top_k)
        
        # Format results - return ANSWERS not questions
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.questions):  # Valid index
                results.append({
                    'question': self.questions[idx],
                    'answer': self.answers[idx],
                    'section': self.sections[idx],
                    'distance': float(dist),
                    'rank': rank + 1
                })
        
        return results
    
    def get_context_for_query(self, query: str, max_context_length: int = 1500, top_k: int = 3) -> str:
        """Get formatted context string for a query"""
        try:
            results = self.retrieve(query, top_k=top_k)
        except Exception as e:
            print(f"Warning: retrieval failed: {e}")
            return ''
        
        # Build context string from Q&A pairs
        context_parts = []
        total_length = 0
        
        for r in results:
            # Format: Question → Answer (from section)
            section_text = f"Q: {r['question']}\nA: {r['answer']}\n[From: {r['section']}]"
            if total_length + len(section_text) > max_context_length:
                break
            context_parts.append(section_text)
            total_length += len(section_text)
        
        return '\n\n'.join(context_parts)


# Backward compatibility
def search(self, query: str, top_k: int = 5, max_distance: float = 2.0):
    """Backward compatible search method"""
    results = self.retrieve(query, top_k=top_k)
    # Filter by distance threshold (lower is better)
    filtered = [r for r in results if r['distance'] <= max_distance]
    return [{'text': r['answer'], 'score': r['distance'], 'metadata': {'section': r['section'], 'question': r['question']}} 
            for r in filtered]

PolicyVectorStore.search = search



if __name__ == '__main__':
    print("=" * 60)
    print("PolicyVectorStore Test - JSONL-based RAG")
    print("=" * 60)
    
    if not _HAVE_RAG:
        print(f"❌ RAG dependencies not available: {_RAG_IMPORT_ERROR}")
        print("Install with: pip install sentence-transformers faiss-cpu")
    else:
        # Check if policies.jsonl exists
        if not os.path.exists('policies.jsonl'):
            print("❌ policies.jsonl not found. Run convert_to_jsonl.py first!")
            print("   python convert_to_jsonl.py")
        else:
            # Build index
            store = PolicyVectorStore(model_name='all-MiniLM-L6-v2')
            print("\n1. Building index from policies.jsonl...")
            store.build_index()
            
            # Test retrieval
            print("\n2. Testing retrieval...")
            test_queries = [
                "Can I bring my pet on the flight?",
                "What are the cancellation fees?",
                "baggage allowance for checked bags",
                "seat selection policy"
            ]
            
            for query in test_queries:
                print(f"\n--- Query: '{query}' ---")
                results = store.retrieve(query, top_k=2)
                for r in results:
                    print(f"  [{r['section']}] Distance: {r['distance']:.3f}")
                    print(f"  Matched Q: {r['question'][:100]}...")
                    print(f"  Answer: {r['answer'][:150]}...")
            
            # Test get_context_for_query
            print("\n3. Testing get_context_for_query...")
            context = store.get_context_for_query("pet travel requirements", max_context_length=500)
            print(f"Context length: {len(context)} chars")
            print(f"Context preview:\n{context[:300]}...")
            
            print("\n✅ All tests passed!")
            print(f"Index saved to: vector_index/")

