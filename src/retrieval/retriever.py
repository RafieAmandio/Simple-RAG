"""Simple Top-K RAG: Remove similarity threshold complexity, just take best results."""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import re

class Retriever:
    """
    Ultra-simple approach: No thresholds, just take top-k best results.
    Let the vector search do its job without second-guessing.
    """
    
    def __init__(self, vector_store: Chroma, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config
        
        # Only essential parameters
        self.top_k = config["retrieval"].get("top_k", 10)
        self.max_docs_final = config["retrieval"].get("max_docs_final", 5)
        
        # Optional simple features
        self.enable_deduplication = config["retrieval"].get("enable_deduplication", True)
        self.enable_quality_filter = config["retrieval"].get("enable_quality_filter", True)
    
    def _is_obviously_bad_content(self, content: str) -> bool:
        """Only filter content that's obviously useless."""
        content = content.strip()
        
        # Too short
        if len(content) < 20:
            return True
        
        # Just a figure caption
        if re.match(r'^\s*(Fig\.|Figure|Table)\s*\d+\s*\.?\s*$', content):
            return True
        
        # Just numbers or symbols
        if re.match(r'^\s*[\d\s\.\,\-\+\=]+\s*$', content):
            return True
        
        return False
    
    def _simple_retrieve(self, query: str) -> List[Document]:
        """Dead simple retrieval: search + filter + take top-k."""
        print(f"🎯 Simple Top-K retrieval for: {query}")
        
        # Step 1: Get candidates from vector search
        print(f"🔍 Getting top {self.top_k} candidates")
        candidates = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        
        # Step 2: Optional deduplication
        if self.enable_deduplication:
            deduplicated = []
            seen_content = set()
            
            for doc, score in candidates:
                # Use first 80 chars as signature
                signature = doc.page_content[:80].strip().lower()
                signature = re.sub(r'\s+', ' ', signature)
                
                if signature not in seen_content:
                    seen_content.add(signature)
                    deduplicated.append((doc, score))
                else:
                    print(f"  🔄 Skipped duplicate: {doc.page_content[:40]}...")
            
            candidates = deduplicated
        
        # Step 3: Optional obvious quality filtering
        if self.enable_quality_filter:
            quality_filtered = []
            
            for doc, score in candidates:
                if not self._is_obviously_bad_content(doc.page_content):
                    quality_filtered.append((doc, score))
                else:
                    print(f"  ❌ Filtered bad content: {doc.page_content[:40]}...")
            
            candidates = quality_filtered
        
        # Step 4: Sort by score and take top documents
        candidates.sort(key=lambda x: x[1])  # Lower score = better
        final_docs = [doc for doc, score in candidates[:self.max_docs_final]]
        
        # Step 5: Show what we got
        print(f"📊 Top-K selection results:")
        for i, (doc, score) in enumerate(candidates[:6], 1):
            source = doc.metadata.get('source', 'Unknown')
            source_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
            preview = doc.page_content[:50].replace('\n', ' ')
            
            is_selected = "📌" if doc in final_docs else "  "
            print(f"  {is_selected}Doc {i}: score={score:.3f}")
            print(f"    {source_name} | {preview}...")
        
        print(f"✅ Selected {len(final_docs)} top documents (no threshold used)")
        return final_docs
    
    def retrieve(self, query: str) -> List[Document]:
        """Main retrieval method - as simple as it gets."""
        # Minimal query cleaning
        clean_query = re.sub(r'\s+', ' ', query.strip())
        
        # Core retrieval
        documents = self._simple_retrieve(clean_query)
        
        # Final output
        print(f"📄 Final top-k documents:")
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  📄 Doc {i} ({source_name}): {preview}...")
        
        return documents