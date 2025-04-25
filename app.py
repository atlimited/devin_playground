import os
import streamlit as st
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("💬 Streamlit Chatbot with OpenAI")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.history,
        temperature=0.7,
    )
    assistant_msg = response.choices[0].message.content
    st.session_state.history.append({"role": "assistant", "content": assistant_msg})

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
