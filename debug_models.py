#!/usr/bin/env python3
"""Debug script to check model name matching issues"""

import ollama

def check_model_availability():
    client = ollama.Client()
    
    # Models expected by the system
    expected_models = {
        'reasoning': ['deepseek-r1:8b', 'smallthinker:latest'],
        'orchestration': ['qwen3:8b', 'llama3:8b-instruct-q4_0'],
        'coding': ['deepseek-coder:6.7b', 'opencoder:8b'],
        'general': ['gemma3:4b', 'gurubot/phi3-mini-abliterated:latest'],
        'vision': ['moondream:latest'],
        'utility': ['mistral-openorca:7b-q4_K_M']
    }
    
    # Get actually available models
    try:
        models_response = client.list()
        available_models = [model['name'] for model in models_response['models']]
        
        print("=== AVAILABLE MODELS ===")
        for model in available_models:
            print(f"  - {model}")
        
        print("\n=== EXPECTED vs AVAILABLE ===")
        all_found = True
        for category, models in expected_models.items():
            print(f"\n{category.upper()}:")
            for model in models:
                if model in available_models:
                    print(f"  ✅ {model} - FOUND")
                else:
                    print(f"  ❌ {model} - NOT FOUND")
                    all_found = False
                    
                    # Check for partial matches
                    partial_matches = [m for m in available_models if model.split(':')[0] in m]
                    if partial_matches:
                        print(f"      Potential matches: {partial_matches}")
        
        print(f"\n=== SUMMARY ===")
        if all_found:
            print("✅ All expected models are available")
        else:
            print("❌ Some expected models are missing")
            print("This explains why DSPy configuration shows 'No models available'")
            
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    check_model_availability()