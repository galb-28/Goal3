#!/usr/bin/env python3
"""Debug specific queries."""

from src.agent.graph import create_medical_agent
from src.tools.medical_tools import MedicalTools

def test_tools_directly():
    """Test the tools directly."""
    print("\n" + "="*70)
    print("TESTING TOOLS DIRECTLY")
    print("="*70)
    
    tools = MedicalTools()
    
    # Test search
    print("\n1. Searching for 'Michael':")
    result = tools.search_patients("Michael")
    print(result)
    
    print("\n2. Searching for 'Linda Davis':")
    result = tools.search_patients("Linda Davis")
    print(result)
    
    print("\n3. Getting vital signs for 'Michael':")
    result = tools.get_vital_signs("Michael")
    print(result)
    
    print("\n4. Resolve patient ID for 'Michael':")
    patient_id = tools._resolve_patient_id("Michael")
    print(f"Patient ID: {patient_id}")

def debug_query(query):
    print(f"\n{'='*70}")
    print(f"DEBUGGING AGENT: {query}")
    print('='*70)
    
    agent = create_medical_agent()
    
    # Test extraction
    extracted = agent._extract_patient_identifier(query)
    print(f"\nExtracted identifier: '{extracted}'")
    
    # Check if planner would detect patient search
    q = query.lower()
    patient_related = any(x in q for x in ["patient", "john", "sarah", "michael", "emily", "david", "linda"]) or q.strip().isdigit()
    print(f"Patient related: {patient_related}")
    has_search = any(word in q for word in ["search", "find", "look for", "who is"])
    print(f"Has search keywords: {has_search}")
    print(f"Has 'patient' in query: {'patient' in q}")
    
    # Run the agent
    response = agent.process_message(query)
    print(f"\nResponse:\n{response}")
    print(f"\nResponse length: {len(response)}")

if __name__ == "__main__":
    test_tools_directly()
    debug_query("Show vital signs for Michael")
    debug_query("Find Linda Davis")
