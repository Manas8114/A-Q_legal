"""
Advanced RAG (Retrieval-Augmented Generation) System for A-Qlegal 2.0
Uses FAISS, ChromaDB, and hybrid retrieval strategies
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from collections import defaultdict


try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Using FAISS only.")


class AdvancedRAGSystem:
    """
    Advanced RAG system with multiple retrieval strategies:
    1. Dense retrieval with FAISS
    2. Sparse retrieval with BM25
    3. Hybrid combining both
    4. ChromaDB for vector database
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "models/faiss_index",
        chroma_path: str = "data/chroma_db",
        use_gpu: bool = False
    ):
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.chroma_path = Path(chroma_path)
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Initialize embedding model
        logger.info(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize storage
        self.documents = []
        self.document_texts = []
        self.document_embeddings = None
        
        # FAISS index
        self.faiss_index = None
        
        # BM25
        self.bm25 = None
        self.bm25_corpus = []
        
        # ChromaDB
        self.chroma_client = None
        self.chroma_collection = None
        
        # Create directories
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ RAG System initialized")
    
    def create_faiss_index(self, use_ivf: bool = False, nlist: int = 100) -> faiss.Index:
        """Create FAISS index for dense retrieval"""
        logger.info("🔨 Creating FAISS index...")
        
        if use_ivf and len(self.documents) > 1000:
            # Use IVF (Inverted File Index) for large datasets
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            if self.document_embeddings is not None:
                logger.info("Training IVF index...")
                index.train(self.document_embeddings)
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Move to GPU if available
        if self.use_gpu:
            logger.info("🚀 Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def initialize_chromadb(self, collection_name: str = "legal_documents"):
        """Initialize ChromaDB for vector storage"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, skipping initialization")
            return
        
        try:
            logger.info("🔄 Initializing ChromaDB...")
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            try:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ Created new ChromaDB collection: {collection_name}")
            except:
                self.chroma_collection = self.chroma_client.get_collection(name=collection_name)
                logger.info(f"✅ Loaded existing ChromaDB collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ChromaDB: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
        use_chromadb: bool = True
    ):
        """Add documents to the RAG system"""
        logger.info(f"📚 Adding {len(documents)} documents to RAG system...")
        
        self.documents = documents
        self.document_texts = [doc.get('text', doc.get('legal_text', '')) for doc in documents]
        
        # Generate embeddings
        logger.info("🔄 Generating embeddings...")
        self.document_embeddings = self.embedding_model.encode(
            self.document_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        self.faiss_index = self.create_faiss_index(use_ivf=len(documents) > 1000)
        self.faiss_index.add(self.document_embeddings)
        logger.info(f"✅ FAISS index created with {self.faiss_index.ntotal} vectors")
        
        # Create BM25 index
        logger.info("🔄 Creating BM25 index...")
        tokenized_corpus = [text.lower().split() for text in self.document_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = tokenized_corpus
        logger.info("✅ BM25 index created")
        
        # Add to ChromaDB if available
        if use_chromadb and CHROMADB_AVAILABLE:
            self.initialize_chromadb()
            if self.chroma_collection is not None:
                logger.info("🔄 Adding documents to ChromaDB...")
                
                # Prepare data for ChromaDB
                ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(documents)]
                metadatas = [
                    {
                        'title': doc.get('title', ''),
                        'category': doc.get('category', ''),
                        'source': doc.get('source', ''),
                        'section': doc.get('section', '')
                    }
                    for doc in documents
                ]
                
                # Add in batches
                for i in range(0, len(documents), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_texts = self.document_texts[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    
                    try:
                        self.chroma_collection.add(
                            ids=batch_ids,
                            documents=batch_texts,
                            metadatas=batch_metadatas
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Error adding batch to ChromaDB: {e}")
                
                logger.info(f"✅ Added documents to ChromaDB")
        
        logger.info("✅ All documents indexed successfully")
    
    def search_faiss(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using FAISS dense retrieval"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Return indices and scores (convert distance to similarity)
        results = [(int(idx), 1.0 / (1.0 + float(dist))) for idx, dist in zip(indices[0], distances[0])]
        return results
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25 sparse retrieval"""
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Normalize scores to 0-1 range
        max_score = scores.max() if scores.max() > 0 else 1.0
        results = [(int(idx), float(scores[idx] / max_score)) for idx in top_indices]
        
        return results
    
    def search_chromadb(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using ChromaDB"""
        if not self.chroma_collection:
            return []
        
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Convert to (index, score) format
            doc_results = []
            if results and 'ids' in results and results['ids']:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Find document index
                    for idx, doc in enumerate(self.documents):
                        if doc.get('id', f'doc_{idx}') == doc_id:
                            distance = results['distances'][0][i] if 'distances' in results else 0.5
                            score = 1.0 / (1.0 + distance)
                            doc_results.append((idx, score))
                            break
            
            return doc_results
            
        except Exception as e:
            logger.warning(f"⚠️  ChromaDB search error: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        faiss_weight: float = 0.4,
        bm25_weight: float = 0.3,
        chroma_weight: float = 0.3,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining FAISS, BM25, and ChromaDB
        
        Args:
            query: Search query
            top_k: Number of results to return
            faiss_weight: Weight for FAISS scores
            bm25_weight: Weight for BM25 scores
            chroma_weight: Weight for ChromaDB scores
            rerank: Whether to rerank results
        """
        logger.debug(f"🔍 Hybrid search for: {query[:50]}...")
        
        # Get results from each retriever
        faiss_results = self.search_faiss(query, top_k * 2)
        bm25_results = self.search_bm25(query, top_k * 2)
        
        # Optionally get ChromaDB results
        chroma_results = []
        if self.chroma_collection and chroma_weight > 0:
            chroma_results = self.search_chromadb(query, top_k * 2)
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        # Add FAISS scores
        for idx, score in faiss_results:
            combined_scores[idx] += faiss_weight * score
        
        # Add BM25 scores
        for idx, score in bm25_results:
            combined_scores[idx] += bm25_weight * score
        
        # Add ChromaDB scores
        for idx, score in chroma_results:
            combined_scores[idx] += chroma_weight * score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for idx, score in sorted_indices[:top_k]:
            if 0 <= idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'score': float(score),
                    'text': self.document_texts[idx],
                    'index': int(idx)
                }
                results.append(result)
        
        # Optionally rerank
        if rerank and results:
            results = self.rerank_results(query, results)
        
        return results
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder or other method"""
        # For now, use simple query-document overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            text = result['text'].lower()
            text_terms = set(text.split())
            
            # Calculate overlap
            overlap = len(query_terms & text_terms) / len(query_terms) if query_terms else 0
            
            # Boost score based on overlap
            result['score'] = result['score'] * (1 + overlap * 0.2)
        
        # Re-sort
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def save_index(self, path: str = None):
        """Save FAISS index and metadata"""
        if path is None:
            path = self.index_path
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.faiss_index is not None:
            # Convert GPU index to CPU before saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                faiss.write_index(cpu_index, str(path / "faiss_index.bin"))
            else:
                faiss.write_index(self.faiss_index, str(path / "faiss_index.bin"))
            
            logger.info(f"💾 FAISS index saved to {path / 'faiss_index.bin'}")
        
        # Save documents
        with open(path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'num_documents': len(self.documents),
            'use_gpu': self.use_gpu
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ RAG index saved to {path}")
    
    def load_index(self, path: str = None):
        """Load FAISS index and metadata"""
        if path is None:
            path = self.index_path
        else:
            path = Path(path)
        
        logger.info(f"📂 Loading RAG index from {path}")
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load documents
        with open(path / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        self.document_texts = [doc.get('text', doc.get('legal_text', '')) for doc in self.documents]
        
        # Load FAISS index
        cpu_index = faiss.read_index(str(path / "faiss_index.bin"))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.faiss_index = cpu_index
        
        # Recreate BM25 index
        tokenized_corpus = [text.lower().split() for text in self.document_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = tokenized_corpus
        
        logger.info(f"✅ RAG index loaded: {len(self.documents)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            'num_documents': len(self.documents),
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'use_gpu': self.use_gpu,
            'chromadb_enabled': self.chroma_collection is not None,
            'bm25_enabled': self.bm25 is not None
        }
        
        if self.chroma_collection:
            try:
                stats['chromadb_count'] = self.chroma_collection.count()
            except:
                stats['chromadb_count'] = 0
        
        return stats


def main():
    """Test RAG system"""
    # Initialize RAG
    rag = AdvancedRAGSystem(use_gpu=False)
    
    # Load sample documents
    dataset_file = Path("data/expanded_legal_dataset.json")
    
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"📚 Loaded {len(documents)} documents")
        
        # Add documents
        rag.add_documents(documents[:1000], use_chromadb=True)  # Test with first 1000
        
        # Test search
        query = "What is the punishment for murder?"
        logger.info(f"\n🔍 Query: {query}")
        
        results = rag.hybrid_search(query, top_k=5)
        
        logger.info(f"\n📊 Top {len(results)} Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Score: {result['score']:.3f}")
            logger.info(f"   Title: {result['document'].get('title', 'N/A')}")
            logger.info(f"   Text: {result['text'][:200]}...")
        
        # Save index
        rag.save_index()
        
        # Print stats
        stats = rag.get_stats()
        logger.info(f"\n📊 RAG Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
    
    else:
        logger.error(f"❌ Dataset not found: {dataset_file}")


if __name__ == "__main__":
    main()

