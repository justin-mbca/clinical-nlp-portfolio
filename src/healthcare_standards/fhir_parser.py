"""
FHIR Parser
Basic FHIR resource parsing for demonstration.
"""

import json

def parse_fhir_resource(resource_json):
    resource = json.loads(resource_json)
    return resource.get('resourceType'), resource.get('id')
