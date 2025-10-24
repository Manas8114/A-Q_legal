#!/usr/bin/env python3
"""
Test Script for A-Qlegal Generative Computer Vision Features
Comprehensive testing of all generative CV capabilities

Author: A-Qlegal Team
Version: 1.0.0
"""

import sys
import os
import traceback
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_generative_cv_imports():
    """Test if all generative CV imports work correctly"""
    print("🔍 Testing Generative CV Imports...")
    
    try:
        from src.generative_cv import (
            GenerativeComputerVision, 
            LegalDocumentLayout, 
            LegalDiagramConfig,
            create_sample_legal_document_data,
            create_sample_diagram_config,
            create_sample_infographic_data
        )
        print("✅ All generative CV imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_generative_cv_initialization():
    """Test generative CV class initialization"""
    print("🔍 Testing Generative CV Initialization...")
    
    try:
        from src.generative_cv import GenerativeComputerVision
        gen_cv = GenerativeComputerVision()
        print("✅ GenerativeComputerVision initialized successfully")
        
        # Test color schemes
        assert hasattr(gen_cv, 'colors')
        assert hasattr(gen_cv, 'legal_colors')
        assert len(gen_cv.colors) > 0
        assert len(gen_cv.legal_colors) > 0
        print("✅ Color schemes loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False

def test_legal_document_template_generation():
    """Test legal document template generation"""
    print("🔍 Testing Legal Document Template Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision, create_sample_legal_document_data
        
        gen_cv = GenerativeComputerVision()
        sample_data = create_sample_legal_document_data()
        
        # Test document template generation
        generated_doc = gen_cv.generate_legal_document_template(
            "contract", sample_data, "professional"
        )
        
        assert generated_doc is not None
        assert hasattr(generated_doc, 'save')
        print("✅ Legal document template generated successfully")
        
        # Test different document types
        doc_types = ["contract", "affidavit", "motion", "brief"]
        for doc_type in doc_types:
            doc = gen_cv.generate_legal_document_template(doc_type, sample_data, "professional")
            assert doc is not None
        print("✅ All document types generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Document template generation error: {e}")
        traceback.print_exc()
        return False

def test_legal_diagram_generation():
    """Test legal diagram generation"""
    print("🔍 Testing Legal Diagram Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision, create_sample_diagram_config
        
        gen_cv = GenerativeComputerVision()
        sample_config = create_sample_diagram_config()
        
        # Test diagram generation
        generated_diagram = gen_cv.generate_legal_diagram(sample_config, "professional")
        
        assert generated_diagram is not None
        assert hasattr(generated_diagram, 'save')
        print("✅ Legal diagram generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Diagram generation error: {e}")
        traceback.print_exc()
        return False

def test_legal_infographic_generation():
    """Test legal infographic generation"""
    print("🔍 Testing Legal Infographic Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision, create_sample_infographic_data
        
        gen_cv = GenerativeComputerVision()
        sample_data = create_sample_infographic_data()
        
        # Test infographic generation
        generated_infographic = gen_cv.generate_legal_infographic(sample_data, "statistics")
        
        assert generated_infographic is not None
        assert hasattr(generated_infographic, 'save')
        print("✅ Legal infographic generated successfully")
        
        # Test different infographic types
        infographic_types = ["statistics", "timeline", "comparison"]
        for infographic_type in infographic_types:
            infographic = gen_cv.generate_legal_infographic(sample_data, infographic_type)
            assert infographic is not None
        print("✅ All infographic types generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Infographic generation error: {e}")
        traceback.print_exc()
        return False

def test_contract_visualization():
    """Test contract visualization generation"""
    print("🔍 Testing Contract Visualization Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision
        
        gen_cv = GenerativeComputerVision()
        
        # Test contract visualization
        contract_data = {
            'title': 'Service Agreement',
            'clauses': [
                {'title': 'Clause 1'},
                {'title': 'Clause 2'},
                {'title': 'Clause 3'}
            ]
        }
        
        generated_viz = gen_cv.generate_contract_visualization(contract_data, "structure")
        
        assert generated_viz is not None
        assert hasattr(generated_viz, 'save')
        print("✅ Contract visualization generated successfully")
        
        # Test different visualization types
        viz_types = ["structure", "timeline", "risk"]
        for viz_type in viz_types:
            viz = gen_cv.generate_contract_visualization(contract_data, viz_type)
            assert viz is not None
        print("✅ All visualization types generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Contract visualization error: {e}")
        traceback.print_exc()
        return False

def test_legal_flowchart():
    """Test legal flowchart generation"""
    print("🔍 Testing Legal Flowchart Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision
        
        gen_cv = GenerativeComputerVision()
        
        # Test flowchart generation
        process_steps = [
            {'label': 'Start', 'type': 'start_end'},
            {'label': 'File Case', 'type': 'process'},
            {'label': 'Serve Notice', 'type': 'process'},
            {'label': 'Discovery', 'type': 'process'},
            {'label': 'Trial', 'type': 'process'},
            {'label': 'Decision', 'type': 'decision'},
            {'label': 'End', 'type': 'start_end'}
        ]
        
        generated_flowchart = gen_cv.generate_legal_flowchart(process_steps, "legal_process")
        
        assert generated_flowchart is not None
        assert hasattr(generated_flowchart, 'save')
        print("✅ Legal flowchart generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Flowchart generation error: {e}")
        traceback.print_exc()
        return False

def test_court_document_template():
    """Test court document template generation"""
    print("🔍 Testing Court Document Template Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision
        
        gen_cv = GenerativeComputerVision()
        
        # Test court document template
        case_info = {
            'court_name': 'UNITED STATES DISTRICT COURT',
            'case_number': 'Case No. 12345',
            'parties': 'Plaintiff vs. Defendant',
            'content': 'This is a sample court document content for testing purposes.'
        }
        
        generated_doc = gen_cv.generate_court_document_template("Order", case_info)
        
        assert generated_doc is not None
        assert hasattr(generated_doc, 'save')
        print("✅ Court document template generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Court document template error: {e}")
        traceback.print_exc()
        return False

def test_legal_presentation_slide():
    """Test legal presentation slide generation"""
    print("🔍 Testing Legal Presentation Slide Generation...")
    
    try:
        from src.generative_cv import GenerativeComputerVision
        
        gen_cv = GenerativeComputerVision()
        
        # Test presentation slide
        slide_data = {
            'title': 'Legal Case Summary',
            'content': [
                {'type': 'heading', 'text': 'Case Overview'},
                {'type': 'bullet', 'text': 'Key point 1'},
                {'type': 'bullet', 'text': 'Key point 2'},
                {'type': 'paragraph', 'text': 'This is a detailed paragraph about the case.'}
            ]
        }
        
        generated_slide = gen_cv.generate_legal_presentation_slide(slide_data, "content")
        
        assert generated_slide is not None
        assert hasattr(generated_slide, 'save')
        print("✅ Legal presentation slide generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Presentation slide error: {e}")
        traceback.print_exc()
        return False

def test_sample_data_generation():
    """Test sample data generation functions"""
    print("🔍 Testing Sample Data Generation...")
    
    try:
        from src.generative_cv import (
            create_sample_legal_document_data,
            create_sample_diagram_config,
            create_sample_infographic_data
        )
        
        # Test sample legal document data
        doc_data = create_sample_legal_document_data()
        assert isinstance(doc_data, dict)
        assert 'title' in doc_data
        assert 'sections' in doc_data
        assert 'signature_blocks' in doc_data
        print("✅ Sample legal document data generated successfully")
        
        # Test sample diagram config
        diagram_config = create_sample_diagram_config()
        assert hasattr(diagram_config, 'title')
        assert hasattr(diagram_config, 'nodes')
        assert hasattr(diagram_config, 'edges')
        print("✅ Sample diagram config generated successfully")
        
        # Test sample infographic data
        infographic_data = create_sample_infographic_data()
        assert isinstance(infographic_data, dict)
        assert 'categories' in infographic_data
        print("✅ Sample infographic data generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Sample data generation error: {e}")
        traceback.print_exc()
        return False

def test_image_save_functionality():
    """Test image save functionality"""
    print("🔍 Testing Image Save Functionality...")
    
    try:
        from src.generative_cv import GenerativeComputerVision, create_sample_legal_document_data
        import tempfile
        import time
        
        gen_cv = GenerativeComputerVision()
        sample_data = create_sample_legal_document_data()
        
        # Generate document
        generated_doc = gen_cv.generate_legal_document_template("contract", sample_data, "professional")
        
        # Test saving to temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_file.close()  # Close the file handle
        
        try:
            generated_doc.save(tmp_file.name)
            assert os.path.exists(tmp_file.name)
            print("✅ Image save functionality works correctly")
            return True
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            except PermissionError:
                # Sometimes Windows needs a moment to release the file
                time.sleep(0.1)
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass  # Ignore if still can't delete
        
    except Exception as e:
        print(f"❌ Image save error: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all generative CV tests"""
    print("🚀 Starting Comprehensive Generative Computer Vision Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Import Test", test_generative_cv_imports),
        ("Initialization Test", test_generative_cv_initialization),
        ("Document Template Test", test_legal_document_template_generation),
        ("Diagram Generation Test", test_legal_diagram_generation),
        ("Infographic Generation Test", test_legal_infographic_generation),
        ("Contract Visualization Test", test_contract_visualization),
        ("Flowchart Test", test_legal_flowchart),
        ("Court Document Template Test", test_court_document_template),
        ("Presentation Slide Test", test_legal_presentation_slide),
        ("Sample Data Test", test_sample_data_generation),
        ("Image Save Test", test_image_save_functionality)
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            test_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    # Save results to file
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': (passed_tests/total_tests)*100,
        'test_results': test_results
    }
    
    with open('generative_cv_test_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Test results saved to: generative_cv_test_results.json")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Generative Computer Vision features are working correctly.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
