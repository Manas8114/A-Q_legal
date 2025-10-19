#!/usr/bin/env python3
"""Comprehensive diagnostic test for A-Qlegal 3.0"""

import sys
sys.path.append('.')

print('='*70)
print('COMPREHENSIVE CODE DIAGNOSTICS')
print('='*70)

# Test 1: Import test
print('\n1. Testing imports...')
try:
    from aqlegal_v3_production import AQlegalV3
    print('   ✅ Imports successful')
except Exception as e:
    print(f'   ❌ Import error: {e}')
    sys.exit(1)

# Test 2: Initialization
print('\n2. Testing initialization...')
try:
    aqlegal = AQlegalV3()
    print('   ✅ Initialization successful')
except Exception as e:
    print(f'   ❌ Init error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Data loading
print('\n3. Testing data loading...')
try:
    aqlegal.legal_data = aqlegal.load_legal_data()
    print(f'   ✅ Loaded {len(aqlegal.legal_data)} documents')
    
    if aqlegal.legal_data:
        sample = aqlegal.legal_data[0]
        print(f'   Sample keys: {list(sample.keys())[:5]}')
except Exception as e:
    print(f'   ❌ Data loading error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model loading
print('\n4. Testing model loading...')
try:
    success = aqlegal.load_models()
    if success:
        print('   ✅ Models loaded successfully')
    else:
        print('   ⚠️ Models failed to load (may be expected)')
except Exception as e:
    print(f'   ❌ Model loading error: {e}')
    import traceback
    traceback.print_exc()

# Test 5: Search functions
print('\n5. Testing search functions...')
test_queries = [
    'Can I kill someone in self-defense?',
    'Can a minor sign a contract?',
    'What is the punishment for theft?'
]

for query in test_queries:
    try:
        results = aqlegal.keyword_search(query, top_k=3)
        print(f'   ✅ "{query[:40]}..." - {len(results)} results')
        if results:
            top = results[0]
            print(f'      Top: {top.get("section", "N/A")} | Score: {top.get("similarity_score", 0):.1f}')
    except Exception as e:
        print(f'   ❌ Search error for "{query}": {e}')
        import traceback
        traceback.print_exc()

# Test 6: Process query
print('\n6. Testing full query processing...')
try:
    response = aqlegal.process_query('Can I kill someone in self-defense?')
    print(f'   ✅ Process query successful')
    print(f'      Type: {response["type"]}')
    print(f'      Confidence: {response["confidence"]}')
    print(f'      Score: {response["max_score"]:.1f}')
    print(f'      Sections: {response["sections"][:2]}')
except Exception as e:
    print(f'   ❌ Process query error: {e}')
    import traceback
    traceback.print_exc()

# Test 7: Edge cases
print('\n7. Testing edge cases...')
edge_cases = [
    '',  # Empty query
    'xyz123',  # Nonsense query
    'Section 96 IPC',  # Exact section
]

for query in edge_cases:
    try:
        response = aqlegal.process_query(query)
        print(f'   ✅ "{query if query else "(empty)"}" - Type: {response["type"]}')
    except Exception as e:
        print(f'   ⚠️ "{query if query else "(empty)"}" - Error: {str(e)[:50]}')

print('\n' + '='*70)
print('DIAGNOSTIC COMPLETE')
print('='*70)




