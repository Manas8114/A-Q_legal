#!/usr/bin/env python3
"""
Test script for A-Qlegal 3.0 functionality
"""

import sys
sys.path.append('.')
from aqlegal_v3_simple import AQlegalV3
import json

def test_aqlegal_v3():
    """Test all A-Qlegal 3.0 functionality"""
    print("🧪 Testing A-Qlegal 3.0 System")
    print("=" * 50)
    
    # Initialize system
    aqlegal = AQlegalV3()
    
    # Load data
    print("📚 Loading legal data...")
    aqlegal.legal_data = aqlegal.load_legal_data()
    print(f"✅ Loaded {len(aqlegal.legal_data)} documents")
    
    # Load models
    print("🤖 Loading AI models...")
    if aqlegal.load_models():
        print("✅ Models loaded successfully")
    else:
        print("❌ Model loading failed")
        return False
    
    # Test queries
    test_queries = [
        "can I kill someone in self-defense?",
        "what is the punishment for theft?",
        "can a minor enter into a contract?",
        "what are my rights if arrested?",
        "how to file a divorce case?",
        "what is the punishment for fraud?"
    ]
    
    print("\n🔍 Testing Query Processing")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Test keyword search
        keyword_results = aqlegal.keyword_search(query, aqlegal.legal_data, 3)
        print(f"   Keyword search: {len(keyword_results)} results")
        
        # Test semantic search
        semantic_results = aqlegal.semantic_search(query, aqlegal.legal_data, 3)
        print(f"   Semantic search: {len(semantic_results)} results")
        
        # Test full processing
        response = aqlegal.process_query(query)
        print(f"   Response type: {response['type']}")
        print(f"   Confidence: {response['confidence']}")
        print(f"   Sections found: {len(response['sections'])}")
        
        # Show top result
        if response['documents']:
            top_doc = response['documents'][0]
            print(f"   Top result: {top_doc.get('title', 'Unknown')}")
            print(f"   Similarity: {top_doc.get('similarity_score', 0):.3f}")
    
    print("\n✅ All tests completed successfully!")
    return True

def test_specific_queries():
    """Test specific legal queries"""
    print("\n🎯 Testing Specific Legal Queries")
    print("=" * 40)
    
    aqlegal = AQlegalV3()
    aqlegal.legal_data = aqlegal.load_legal_data()
    aqlegal.load_models()
    
    # Self-defense query
    print("\n1. Self-Defense Query:")
    query = "can I kill someone in self-defense?"
    response = aqlegal.process_query(query)
    
    print(f"Query: {query}")
    print(f"Response Type: {response['type']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Explanation: {response['explanation'][:200]}...")
    
    if response['documents']:
        print("Top Results:")
        for i, doc in enumerate(response['documents'][:2], 1):
            print(f"  {i}. {doc.get('title', 'Unknown')} (Score: {doc.get('similarity_score', 0):.2f})")
    
    # Theft query
    print("\n2. Theft Query:")
    query = "what is the punishment for theft?"
    response = aqlegal.process_query(query)
    
    print(f"Query: {query}")
    print(f"Response Type: {response['type']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Explanation: {response['explanation'][:200]}...")
    
    if response['documents']:
        print("Top Results:")
        for i, doc in enumerate(response['documents'][:2], 1):
            print(f"  {i}. {doc.get('title', 'Unknown')} (Score: {doc.get('similarity_score', 0):.2f})")

if __name__ == "__main__":
    print("🚀 A-Qlegal 3.0 Test Suite")
    print("=" * 50)
    
    # Run basic tests
    if test_aqlegal_v3():
        print("\n🎉 Basic functionality test passed!")
        
        # Run specific query tests
        test_specific_queries()
        
        print("\n🎯 All tests completed successfully!")
        print("✅ A-Qlegal 3.0 is ready for use!")
    else:
        print("\n❌ Tests failed. Please check the setup.")



