#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete feature test for A-Qlegal 3.0"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append('.')
from aqlegal_v3_production import AQlegalV3

def test_query(aqlegal, query, expected_domain=None):
    """Test a single query and display results"""
    print(f'\n{"="*70}')
    print(f'Query: "{query}"')
    print('="*70')
    
    try:
        response = aqlegal.process_query(query)
        
        print(f'✅ Response Type: {response["type"]}')
        print(f'   Confidence: {response["confidence"]}')
        print(f'   Max Score: {response["max_score"]:.2f}')
        
        if response['sections']:
            print(f'   Sections Found: {len(response["sections"])}')
            for i, section in enumerate(response["sections"][:3], 1):
                print(f'      {i}. {section}')
        
        if response['documents']:
            print(f'   Documents: {len(response["documents"])}')
            top_doc = response["documents"][0]
            print(f'   Top Document: {top_doc.get("title", "Unknown")[:60]}...')
            print(f'   Top Score: {top_doc.get("similarity_score", 0):.2f}')
            
            # Display formatted output
            print(f'\n   📝 Summary: {response["documents"][0].get("simplified_summary", "N/A")[:100]}...')
            if response["documents"][0].get("punishment"):
                print(f'   ⚖️ Punishment: {response["documents"][0].get("punishment")}')
            if response["documents"][0].get("keywords"):
                print(f'   🏷️ Keywords: {", ".join(response["documents"][0].get("keywords", [])[:5])}')
        
        return True
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    print('='*70)
    print('A-QLEGAL 3.0 - COMPLETE FEATURE TEST')
    print('='*70)
    
    # Initialize
    aqlegal = AQlegalV3()
    aqlegal.legal_data = aqlegal.load_legal_data()
    aqlegal.load_models()
    
    print(f'\n✅ System Initialized: {len(aqlegal.legal_data)} documents loaded')
    
    # Test cases covering different scenarios
    test_cases = [
        ("Can I kill someone in self-defense?", "self_defense"),
        ("Can a minor sign a contract?", "contract"),
        ("What is the punishment for theft?", "theft"),
        ("Section 420 IPC", "exact_section"),
        ("What are my rights if arrested?", "arrest"),
        ("How to file for divorce?", "marriage"),
        ("Is cheating punishable?", "fraud"),
        ("Can I defend my property?", "self_defense"),
    ]
    
    passed = 0
    failed = 0
    
    for query, domain in test_cases:
        if test_query(aqlegal, query, domain):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f'\n{"="*70}')
    print('TEST SUMMARY')
    print('="*70')
    print(f'Total Tests: {len(test_cases)}')
    print(f'✅ Passed: {passed}')
    print(f'❌ Failed: {failed}')
    print(f'Success Rate: {(passed/len(test_cases)*100):.1f}%')
    print('='*70)

if __name__ == "__main__":
    main()

