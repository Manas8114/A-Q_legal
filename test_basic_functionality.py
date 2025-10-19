#!/usr/bin/env python3
"""
Basic Functionality Test for A-Qlegal 4.0
Tests core features without heavy dependencies
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing Basic Imports...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"  ❌ Streamlit import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✅ Pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ Pandas import failed: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("  ✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"  ❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_legal_components():
    """Test legal components without heavy dependencies"""
    print("\n⚖️ Testing Legal Components...")
    
    try:
        # Test LegalDocumentGenerator
        from aqlegal_v4_enhanced import LegalDocumentGenerator
        doc_generator = LegalDocumentGenerator()
        print("  ✅ LegalDocumentGenerator initialized")
        
        # Test document generation
        test_data = {
            "recipient_name": "Jane Smith",
            "recipient_address": "123 Test Street, New Delhi",
            "subject": "Test Contract Dispute",
            "issue_description": "This is a test issue description for the legal notice.",
            "demand": "Please resolve this matter immediately.",
            "time_limit": "15",
            "sender_name": "John Doe",
            "advocate_name": "Test Advocate",
            "date": "2024-01-15",
            "place": "New Delhi"
        }
        
        document = doc_generator.generate_document("legal_notice", test_data)
        if document and len(document) > 100:
            print("  ✅ Document generation working")
        else:
            print("  ⚠️ Document generation seems short")
        
    except Exception as e:
        print(f"  ❌ LegalDocumentGenerator failed: {e}")
        return False
    
    try:
        # Test CaseLawIntegration
        from aqlegal_v4_enhanced import CaseLawIntegration
        case_law = CaseLawIntegration()
        print("  ✅ CaseLawIntegration initialized")
        
        # Test case law search
        results = case_law.search_case_law("fundamental rights")
        if results:
            print(f"  ✅ Case law search working - {len(results)} results")
        else:
            print("  ⚠️ No case law results found")
        
    except Exception as e:
        print(f"  ❌ CaseLawIntegration failed: {e}")
        return False
    
    try:
        # Test LegalGlossary
        from aqlegal_v4_enhanced import LegalGlossary
        glossary = LegalGlossary()
        print("  ✅ LegalGlossary initialized")
        
        # Test glossary search
        results = glossary.search_glossary("affidavit")
        if results:
            print(f"  ✅ Glossary search working - {len(results)} results")
        else:
            print("  ⚠️ No glossary results found")
        
    except Exception as e:
        print(f"  ❌ LegalGlossary failed: {e}")
        return False
    
    try:
        # Test LegalCalendar
        from aqlegal_v4_enhanced import LegalCalendar
        calendar = LegalCalendar()
        print("  ✅ LegalCalendar initialized")
        
        # Test upcoming events
        upcoming = calendar.get_upcoming_events(30)
        if upcoming:
            print(f"  ✅ Calendar working - {len(upcoming)} upcoming events")
        else:
            print("  ⚠️ No upcoming events found")
        
    except Exception as e:
        print(f"  ❌ LegalCalendar failed: {e}")
        return False
    
    return True

def test_pdf_export():
    """Test PDF export functionality"""
    print("\n📄 Testing PDF Export...")
    
    try:
        from aqlegal_v4_enhanced import PDFExporter
        pdf_exporter = PDFExporter()
        print("  ✅ PDFExporter initialized")
        
        # Test legal answer export
        test_answer_data = {
            "explanation": "This is a test legal explanation.",
            "sections": ["Section 1", "Section 2"],
            "example": "This is a test example.",
            "punishment": "This is a test punishment.",
            "source": "Test Source"
        }
        
        pdf_bytes = pdf_exporter.export_legal_answer("Test Question", test_answer_data)
        if pdf_bytes and len(pdf_bytes) > 1000:
            print(f"  ✅ PDF export working - {len(pdf_bytes)} bytes")
        else:
            print("  ⚠️ PDF export seems small")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PDF export failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        from aqlegal_v4_enhanced import AQlegalV4
        aqlegal = AQlegalV4()
        print("  ✅ AQlegalV4 initialized")
        
        # Test data loading
        legal_data = aqlegal.load_legal_data()
        if legal_data:
            print(f"  ✅ Legal data loaded - {len(legal_data)} documents")
            
            # Show sample data structure
            if legal_data:
                sample = legal_data[0]
                print(f"  📋 Sample document keys: {list(sample.keys())}")
        else:
            print("  ⚠️ No legal data loaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        from aqlegal_v4_enhanced import AQlegalV4
        aqlegal = AQlegalV4()
        
        # Test model loading
        models_loaded = aqlegal.load_models()
        if models_loaded:
            print("  ✅ Models loaded successfully")
        else:
            print("  ⚠️ Models not loaded (may be expected if not trained)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def test_query_processing():
    """Test query processing functionality"""
    print("\n❓ Testing Query Processing...")
    
    try:
        from aqlegal_v4_enhanced import AQlegalV4
        aqlegal = AQlegalV4()
        
        # Load data first
        legal_data = aqlegal.load_legal_data()
        if not legal_data:
            print("  ⚠️ No legal data available for testing")
            return True
        
        # Test a simple query
        test_query = "What is theft?"
        response = aqlegal.process_query(test_query)
        
        if response:
            print("  ✅ Query processing working")
            print(f"  📝 Response keys: {list(response.keys())}")
        else:
            print("  ⚠️ No response generated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Query processing failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting Basic A-Qlegal 4.0 Functionality Test")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("Legal Components", test_legal_components()))
    test_results.append(("PDF Export", test_pdf_export()))
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Model Loading", test_model_loading()))
    test_results.append(("Query Processing", test_query_processing()))
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 TEST REPORT SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for _, success in test_results if success)
    failed_tests = total_tests - successful_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Successful: {successful_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, success in test_results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    if failed_tests == 0:
        print("\n🎉 All basic tests passed! A-Qlegal 4.0 core functionality is working.")
    else:
        print(f"\n⚠️ {failed_tests} tests failed. Please review the issues above.")
    
    # Save report
    report = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": (successful_tests/total_tests)*100,
        "test_results": [{"name": name, "success": success} for name, success in test_results]
    }
    
    with open("basic_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: basic_test_report.json")

if __name__ == "__main__":
    main()
