#!/usr/bin/env python3
import requests
import json
import sys

PROXY_URL = "http://localhost:8000"

def call_llm(model: str, messages: list, temperature: float = 0.7):
    """
    Call LLM through the LiteLLM proxy server
    
    Args:
        model: The model name as configured in the proxy
        messages: List of message dictionaries with role and content
        temperature: Temperature parameter for generation
    
    Returns:
        Response from the LLM
    """
    url = f"{PROXY_URL}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM proxy: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        sys.exit(1)

def get_available_models():
    """Get list of available models from the proxy"""
    try:
        response = requests.get(f"{PROXY_URL}/v1/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting models: {e}")
        sys.exit(1)

def main():
    # Example of listing models
    print("Available models:")
    models = get_available_models()
    for model in models["data"]:
        print(f"- {model['id']}")
    
    # Example conversation
    model = input("\nEnter model to use (e.g., gpt-3.5-turbo, claude-3-sonnet): ")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you introduce yourself?"}
    ]
    
    print(f"\nCalling {model}...")
    response = call_llm(model, messages)
    
    print("\nResponse:")
    print(json.dumps(response, indent=2))
    
    # Extract and print just the assistant's message
    assistant_message = response["choices"][0]["message"]["content"]
    print("\nAssistant:", assistant_message)

if __name__ == "__main__":
    main()
