"""
ICD/CPT Parser
Basic ICD code parsing for demonstration.
"""

def parse_icd_code(code):
    # Example: return code type and description
    if code.startswith('E'): return 'ICD-10', 'Endocrine disease'
    if code.startswith('I'): return 'ICD-10', 'Cardiovascular disease'
    return 'Unknown', 'Unknown disease'
