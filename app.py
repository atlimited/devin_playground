import os
import streamlit as st
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure proxy URL
PROXY_URL = "http://localhost:8000"

# Set API keys for backward compatibility
openai_key = os.getenv("OPENAI_API_KEY", "")
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
cohere_key = os.getenv("COHERE_API_KEY", "")

# Default model for backward compatibility
PROVIDER_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic": "claude-3.5-sonnet",
    "Cohere": "command-nightly",
}

# Determine if we're in test mode
TEST_MODE = (
    openai_key.startswith("test-") or 
    anthropic_key.startswith("test-") or 
    cohere_key.startswith("test-")
)

# Function to get available models from proxy
def get_available_models():
    """Get list of available models from the proxy"""
    try:
        response = requests.get(f"{PROXY_URL}/v1/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting models: {e}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Response: {e.response.text}")
        return {"data": []}

# Function to call LLM via proxy
def call_llm_proxy(model: str, messages: list, temperature: float = 0.7):
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
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Streamlit UI setup
st.title("💬 Multi-Provider Chatbot with LiteLLM Proxy")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_json" not in st.session_state:
    st.session_state.show_json = False
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Get models from proxy
with st.spinner("Loading available models..."):
    proxy_models = get_available_models()
    model_options = [model["id"] for model in proxy_models.get("data", [])]
    
    # If no models found, use default providers as fallback
    if not model_options:
        st.warning("Could not connect to proxy server. Using default provider models.")
        model_options = [PROVIDER_MODELS[provider] for provider in PROVIDER_MODELS]

# Model selection
selected_model = st.selectbox("Choose LLM Model:", model_options)

# Temperature slider
temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Display toggle for raw JSON response
st.session_state.show_json = st.checkbox("Show full JSON response", st.session_state.show_json)

# Chat form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

# Process form submission
if submit and user_input:
    # Add user message to history
    st.session_state.history.append({
        "role": "user", 
        "content": user_input
    })
    
    try:
        if TEST_MODE:
            # Mock response for test mode
            with st.spinner(f"Generating response from {selected_model}..."):
                time.sleep(1)  # Simulate API call delay
                assistant_msg = f"This is a mock response from {selected_model}. You said: {user_input}"
                st.session_state.last_response = {
                    "choices": [{"message": {"content": assistant_msg}}]
                }
        else:
            # Real API call via proxy
            with st.spinner(f"Generating response from {selected_model}..."):
                # Create messages from history
                messages = [{"role": msg["role"], "content": msg["content"]} 
                           for msg in st.session_state.history]
                
                # Call the proxy
                response = call_llm_proxy(
                    model=selected_model,
                    messages=messages,
                    temperature=temperature
                )
                
                # Store full response and extract message
                st.session_state.last_response = response
                assistant_msg = response["choices"][0]["message"]["content"]
        
        # Add assistant message to history
        st.session_state.history.append({
            "role": "assistant", 
            "content": assistant_msg,
            "model": selected_model  # Store which model generated this response
        })
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        if TEST_MODE:
            st.info("Using test API keys? Make sure the proxy server is running.")

# Display chat history
st.subheader("Chat History")
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        model_name = msg.get("model", "Assistant")
        st.markdown(f"**{model_name}:** {msg['content']}")

# Display raw JSON if enabled
if st.session_state.show_json and st.session_state.last_response:
    st.subheader("Raw JSON Response")
    st.code(json.dumps(st.session_state.last_response, indent=2), language="json")
