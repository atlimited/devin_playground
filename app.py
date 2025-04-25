import os
import streamlit as st
import openai
import litellm
import time

openai_key = os.getenv("OPENAI_API_KEY", "")
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
cohere_key = os.getenv("COHERE_API_KEY", "")

PROVIDER_MODELS = {
    "OpenAI": "gpt-3.5-turbo",
    "Anthropic": "claude-instant-1",
    "Cohere": "command-nightly",
}

TEST_MODE = (
    openai_key.startswith("test-") or 
    anthropic_key.startswith("test-") or 
    cohere_key.startswith("test-")
)

st.title("💬 Multi-Provider Chatbot with LiteLLM")

provider = st.selectbox("Choose LLM Provider:", list(PROVIDER_MODELS.keys()))

if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.history.append({
        "role": "user", 
        "content": user_input,
        "provider": None  # User messages don't have a provider
    })
    
    try:
        if provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = openai_key
        elif provider == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        elif provider == "Cohere":
            os.environ["COHERE_API_KEY"] = cohere_key
        
        if TEST_MODE:
            with st.spinner(f"Generating response from {provider}..."):
                time.sleep(1)  # Simulate API call delay
                
            if provider == "OpenAI":
                assistant_msg = f"This is a mock response from OpenAI. You said: {user_input}"
            elif provider == "Anthropic":
                assistant_msg = f"This is a mock response from Anthropic. You said: {user_input}"
            elif provider == "Cohere":
                assistant_msg = f"This is a mock response from Cohere. You said: {user_input}"
        else:
            with st.spinner(f"Generating response from {provider}..."):
                messages = [{"role": msg["role"], "content": msg["content"]} 
                           for msg in st.session_state.history]
                
                response = litellm.completion(
                    model=PROVIDER_MODELS[provider],
                    messages=messages
                )
                assistant_msg = response.choices[0].message.content
        
        st.session_state.history.append({
            "role": "assistant", 
            "content": assistant_msg,
            "provider": provider  # Store which provider generated this response
        })
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Using test API keys? Enable TEST_MODE in the code for mock responses.")

for msg in st.session_state.history:
    if msg["role"] == "user":
        label = "You"
    else:
        label = msg.get("provider", "Assistant")
    
    st.markdown(f"**{label}:** {msg['content']}")
