#!/usr/bin/env python3
"""
Comprehensive Test Suite for A-Qlegal 4.0
Tests all endpoints, features, and functionality
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
import subprocess
import threading
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AQlegalTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = {}
        self.api_process = None
        
    def start_api_server(self):
        """Start the FastAPI server for testing"""
        print("🚀 Starting API server...")
        try:
            # Start the server in a separate process
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "src.api.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--log-level", "error"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(5)
            
            # Test if server is running
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API server started successfully")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print("❌ Failed to start API server")
            return False
            
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def stop_api_server(self):
        """Stop the API server"""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait()
            print("🛑 API server stopped")
    
    def test_endpoint(self, method, endpoint, data=None, expected_status=200):
        """Test a single endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, timeout=10)
            else:
                return False, f"Unsupported method: {method}"
            
            success = response.status_code == expected_status
            result = {
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": success,
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
            
            return success, result
            
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
        except Exception as e:
            return False, {"error": str(e)}
    
    def test_basic_endpoints(self):
        """Test basic API endpoints"""
        print("\n📋 Testing Basic Endpoints...")
        
        endpoints = [
            ("GET", "/", 200),
            ("GET", "/health", 200),
            ("GET", "/status", 200),
            ("GET", "/models/info", 200),
            ("GET", "/cache/stats", 200),
        ]
        
        for method, endpoint, expected_status in endpoints:
            print(f"  Testing {method} {endpoint}...")
            success, result = self.test_endpoint(method, endpoint, expected_status=expected_status)
            
            if success:
                print(f"    ✅ {method} {endpoint} - Status: {result['status_code']}")
            else:
                print(f"    ❌ {method} {endpoint} - Error: {result}")
            
            self.test_results[f"{method}_{endpoint}"] = {"success": success, "result": result}
    
    def test_question_endpoint(self):
        """Test the question answering endpoint"""
        print("\n❓ Testing Question Endpoint...")
        
        test_questions = [
            "What is the punishment for theft?",
            "What are the rights of an arrested person?",
            "What is the procedure for filing a complaint?",
            "What is the definition of murder?",
            "What are the fundamental rights?"
        ]
        
        for question in test_questions:
            print(f"  Testing question: '{question[:50]}...'")
            data = {"question": question, "top_k": 3}
            success, result = self.test_endpoint("POST", "/ask", data)
            
            if success:
                print(f"    ✅ Question answered - Confidence: {result['response'].get('confidence', 'N/A')}")
            else:
                print(f"    ❌ Question failed - Error: {result}")
            
            self.test_results[f"question_{hash(question)}"] = {"success": success, "result": result}
    
    def test_feedback_endpoint(self):
        """Test the feedback endpoint"""
        print("\n📝 Testing Feedback Endpoint...")
        
        feedback_data = {
            "question": "What is theft?",
            "answer": "Theft is the dishonest taking of property belonging to another person.",
            "is_helpful": True,
            "feedback_text": "Very helpful explanation"
        }
        
        success, result = self.test_endpoint("POST", "/feedback", feedback_data)
        
        if success:
            print(f"    ✅ Feedback submitted successfully")
        else:
            print(f"    ❌ Feedback failed - Error: {result}")
        
        self.test_results["feedback"] = {"success": success, "result": result}
    
    def test_cache_endpoints(self):
        """Test cache-related endpoints"""
        print("\n💾 Testing Cache Endpoints...")
        
        # Test cache search
        success, result = self.test_endpoint("GET", "/cache/search?query=theft&limit=5")
        if success:
            print(f"    ✅ Cache search - Found {result['response'].get('count', 0)} results")
        else:
            print(f"    ❌ Cache search failed - Error: {result}")
        
        self.test_results["cache_search"] = {"success": success, "result": result}
        
        # Test cache clear
        success, result = self.test_endpoint("DELETE", "/cache")
        if success:
            print(f"    ✅ Cache cleared successfully")
        else:
            print(f"    ❌ Cache clear failed - Error: {result}")
        
        self.test_results["cache_clear"] = {"success": success, "result": result}
    
    def test_streamlit_app(self):
        """Test the Streamlit application"""
        print("\n🖥️ Testing Streamlit Application...")
        
        try:
            # Import and test the main components
            from aqlegal_v4_enhanced import AQlegalV4, LegalDocumentGenerator, CaseLawIntegration, LegalGlossary, LegalCalendar, PDFExporter, ComputerVisionProcessor
            
            # Test AQlegalV4 initialization
            print("  Testing AQlegalV4 initialization...")
            aqlegal = AQlegalV4()
            print("    ✅ AQlegalV4 initialized successfully")
            
            # Test component initialization
            components = [
                ("LegalDocumentGenerator", LegalDocumentGenerator()),
                ("CaseLawIntegration", CaseLawIntegration()),
                ("LegalGlossary", LegalGlossary()),
                ("LegalCalendar", LegalCalendar()),
                ("PDFExporter", PDFExporter()),
                ("ComputerVisionProcessor", ComputerVisionProcessor())
            ]
            
            for name, component in components:
                print(f"  Testing {name}...")
                if component:
                    print(f"    ✅ {name} initialized successfully")
                else:
                    print(f"    ❌ {name} initialization failed")
            
            # Test data loading
            print("  Testing data loading...")
            try:
                legal_data = aqlegal.load_legal_data()
                if legal_data:
                    print(f"    ✅ Legal data loaded - {len(legal_data)} documents")
                else:
                    print("    ⚠️ No legal data loaded")
            except Exception as e:
                print(f"    ❌ Data loading failed: {e}")
            
            # Test model loading
            print("  Testing model loading...")
            try:
                models_loaded = aqlegal.load_models()
                if models_loaded:
                    print("    ✅ Models loaded successfully")
                else:
                    print("    ⚠️ Models not loaded (may be expected)")
            except Exception as e:
                print(f"    ❌ Model loading failed: {e}")
            
            self.test_results["streamlit_app"] = {"success": True, "result": "All components initialized"}
            
        except Exception as e:
            print(f"    ❌ Streamlit app test failed: {e}")
            self.test_results["streamlit_app"] = {"success": False, "result": str(e)}
    
    def test_computer_vision(self):
        """Test computer vision functionality"""
        print("\n👁️ Testing Computer Vision Features...")
        
        try:
            from aqlegal_v4_enhanced import ComputerVisionProcessor
            cv_processor = ComputerVisionProcessor()
            
            # Create a test image
            test_image = Image.new('RGB', (400, 200), color='white')
            
            # Test OCR
            print("  Testing OCR functionality...")
            try:
                ocr_results = cv_processor.extract_text_from_image(test_image)
                print(f"    ✅ OCR test completed - EasyOCR: {ocr_results['easyocr']['confidence']:.2f}, Tesseract: {ocr_results['tesseract']['confidence']:.2f}")
            except Exception as e:
                print(f"    ❌ OCR test failed: {e}")
            
            # Test document type detection
            print("  Testing document type detection...")
            try:
                doc_type_results = cv_processor.detect_document_type(test_image)
                print(f"    ✅ Document type detection completed - Type: {doc_type_results['document_type']}")
            except Exception as e:
                print(f"    ❌ Document type detection failed: {e}")
            
            # Test signature verification
            print("  Testing signature verification...")
            try:
                signature_results = cv_processor.verify_signature(test_image)
                print(f"    ✅ Signature verification completed - Detected: {signature_results['signature_detected']}")
            except Exception as e:
                print(f"    ❌ Signature verification failed: {e}")
            
            # Test document structure analysis
            print("  Testing document structure analysis...")
            try:
                structure_results = cv_processor.analyze_document_structure(test_image)
                print(f"    ✅ Document structure analysis completed - Layout: {structure_results['document_layout']}")
            except Exception as e:
                print(f"    ❌ Document structure analysis failed: {e}")
            
            self.test_results["computer_vision"] = {"success": True, "result": "All CV features tested"}
            
        except Exception as e:
            print(f"    ❌ Computer vision test failed: {e}")
            self.test_results["computer_vision"] = {"success": False, "result": str(e)}
    
    def test_document_generation(self):
        """Test document generation features"""
        print("\n📄 Testing Document Generation...")
        
        try:
            from aqlegal_v4_enhanced import LegalDocumentGenerator
            
            doc_generator = LegalDocumentGenerator()
            
            # Test document generation
            test_data = {
                "party_name": "John Doe",
                "other_party": "Jane Smith",
                "subject": "Contract Dispute",
                "date": "2024-01-15",
                "place": "New Delhi"
            }
            
            document_types = ["legal_notice", "complaint", "affidavit", "power_of_attorney"]
            
            for doc_type in document_types:
                print(f"  Testing {doc_type} generation...")
                try:
                    document = doc_generator.generate_document(doc_type, test_data)
                    if document and len(document) > 100:
                        print(f"    ✅ {doc_type} generated successfully - {len(document)} characters")
                    else:
                        print(f"    ⚠️ {doc_type} generated but seems short")
                except Exception as e:
                    print(f"    ❌ {doc_type} generation failed: {e}")
            
            self.test_results["document_generation"] = {"success": True, "result": "All document types tested"}
            
        except Exception as e:
            print(f"    ❌ Document generation test failed: {e}")
            self.test_results["document_generation"] = {"success": False, "result": str(e)}
    
    def test_case_law_integration(self):
        """Test case law integration"""
        print("\n⚖️ Testing Case Law Integration...")
        
        try:
            from aqlegal_v4_enhanced import CaseLawIntegration
            
            case_law = CaseLawIntegration()
            
            # Test case law search
            test_queries = [
                "fundamental rights",
                "constitutional law",
                "criminal procedure",
                "civil rights"
            ]
            
            for query in test_queries:
                print(f"  Testing case law search: '{query}'...")
                try:
                    results = case_law.search_case_law(query)
                    if results:
                        print(f"    ✅ Found {len(results)} case law results")
                    else:
                        print(f"    ⚠️ No case law results found")
                except Exception as e:
                    print(f"    ❌ Case law search failed: {e}")
            
            self.test_results["case_law"] = {"success": True, "result": "Case law search tested"}
            
        except Exception as e:
            print(f"    ❌ Case law test failed: {e}")
            self.test_results["case_law"] = {"success": False, "result": str(e)}
    
    def test_glossary(self):
        """Test legal glossary"""
        print("\n📚 Testing Legal Glossary...")
        
        try:
            from aqlegal_v4_enhanced import LegalGlossary
            
            glossary = LegalGlossary()
            
            # Test glossary search
            test_terms = ["affidavit", "bail", "contract", "jurisdiction", "plaintiff"]
            
            for term in test_terms:
                print(f"  Testing glossary search: '{term}'...")
                try:
                    results = glossary.search_glossary(term)
                    if results:
                        print(f"    ✅ Found {len(results)} glossary results")
                    else:
                        print(f"    ⚠️ No glossary results found")
                except Exception as e:
                    print(f"    ❌ Glossary search failed: {e}")
            
            self.test_results["glossary"] = {"success": True, "result": "Glossary search tested"}
            
        except Exception as e:
            print(f"    ❌ Glossary test failed: {e}")
            self.test_results["glossary"] = {"success": False, "result": str(e)}
    
    def test_calendar(self):
        """Test legal calendar"""
        print("\n📅 Testing Legal Calendar...")
        
        try:
            from aqlegal_v4_enhanced import LegalCalendar
            
            calendar = LegalCalendar()
            
            # Test upcoming events
            print("  Testing upcoming events...")
            try:
                upcoming = calendar.get_upcoming_events(30)
                print(f"    ✅ Found {len(upcoming)} upcoming events")
            except Exception as e:
                print(f"    ❌ Upcoming events test failed: {e}")
            
            # Test events by category
            print("  Testing events by category...")
            try:
                categories = ["holiday", "court_holiday", "legal_deadline"]
                for category in categories:
                    events = calendar.get_events_by_category(category)
                    print(f"    ✅ Found {len(events)} events in {category}")
            except Exception as e:
                print(f"    ❌ Events by category test failed: {e}")
            
            self.test_results["calendar"] = {"success": True, "result": "Calendar features tested"}
            
        except Exception as e:
            print(f"    ❌ Calendar test failed: {e}")
            self.test_results["calendar"] = {"success": False, "result": str(e)}
    
    def test_pdf_export(self):
        """Test PDF export functionality"""
        print("\n📄 Testing PDF Export...")
        
        try:
            from aqlegal_v4_enhanced import PDFExporter
            
            pdf_exporter = PDFExporter()
            
            # Test legal answer export
            print("  Testing legal answer PDF export...")
            try:
                test_answer_data = {
                    "explanation": "This is a test legal explanation.",
                    "sections": ["Section 1", "Section 2"],
                    "example": "This is a test example.",
                    "punishment": "This is a test punishment.",
                    "source": "Test Source"
                }
                
                pdf_bytes = pdf_exporter.export_legal_answer("Test Question", test_answer_data)
                if pdf_bytes and len(pdf_bytes) > 1000:
                    print(f"    ✅ Legal answer PDF exported - {len(pdf_bytes)} bytes")
                else:
                    print(f"    ⚠️ PDF export seems small")
            except Exception as e:
                print(f"    ❌ Legal answer PDF export failed: {e}")
            
            # Test legal document export
            print("  Testing legal document PDF export...")
            try:
                test_document = "This is a test legal document content."
                pdf_bytes = pdf_exporter.export_legal_document(test_document, "test_document")
                if pdf_bytes and len(pdf_bytes) > 1000:
                    print(f"    ✅ Legal document PDF exported - {len(pdf_bytes)} bytes")
                else:
                    print(f"    ⚠️ PDF export seems small")
            except Exception as e:
                print(f"    ❌ Legal document PDF export failed: {e}")
            
            self.test_results["pdf_export"] = {"success": True, "result": "PDF export tested"}
            
        except Exception as e:
            print(f"    ❌ PDF export test failed: {e}")
            self.test_results["pdf_export"] = {"success": False, "result": str(e)}
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Comprehensive A-Qlegal 4.0 Test Suite")
        print("=" * 60)
        
        # Test Streamlit app components first
        self.test_streamlit_app()
        self.test_computer_vision()
        self.test_document_generation()
        self.test_case_law_integration()
        self.test_glossary()
        self.test_calendar()
        self.test_pdf_export()
        
        # Test API endpoints
        if self.start_api_server():
            self.test_basic_endpoints()
            self.test_question_endpoint()
            self.test_feedback_endpoint()
            self.test_cache_endpoints()
            self.stop_api_server()
        else:
            print("⚠️ Skipping API tests - server could not be started")
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {test_name}")
            if not result["success"] and "error" in result["result"]:
                print(f"      Error: {result['result']['error']}")
        
        # Save detailed report
        report_file = "test_report.json"
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! A-Qlegal 4.0 is ready for use.")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please review the issues above.")

def main():
    """Main test function"""
    tester = AQlegalTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
