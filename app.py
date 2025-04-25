import os
import streamlit as st
import openai
import litellm

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

PROVIDER_MODELS = {
    "OpenAI": "gpt-3.5-turbo",
    "Anthropic": "claude-instant-1",
    "Cohere": "command-nightly",
}

st.title("💬 Multi-Provider Chatbot with LiteLLM")

provider = st.selectbox("Choose LLM Provider:", list(PROVIDER_MODELS.keys()))

if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = openai.api_key
    elif provider == "Anthropic":
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    elif provider == "Cohere":
        os.environ["COHERE_API_KEY"] = cohere_key
    
    response = litellm.completion(
        model=PROVIDER_MODELS[provider],
        messages=st.session_state.history
    )
    assistant_msg = response.choices[0].message.content
    st.session_state.history.append({"role": "assistant", "content": assistant_msg})

for msg in st.session_state.history:
    label = "You" if msg["role"] == "user" else provider
    st.markdown(f"**{label}:** {msg['content']}")
