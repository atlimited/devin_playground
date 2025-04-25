import os
import streamlit as st
import requests
import json
import time
import base64
from io import BytesIO
from PIL import Image
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

# Vision-capable models
VISION_MODELS = [
    "gpt-4o-mini",
    "claude-3.7-sonnet",
    "gemini-2.5-pro",
    "Llama-4-Maverick-17B-128E-Instruct",
]

# Audio-capable models
AUDIO_MODELS = [
    "gpt-4o-audio-preview"
]

# Audio transcription models
AUDIO_TRANSCRIPTION_MODELS = {
    "openai": "whisper-1",
    "sambanova": "Whisper-Large-v3"  # SambaNova の正確なモデル名（大文字小文字を維持）
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

# Function to encode image to base64
def encode_image_to_base64(image_file):
    """Convert an uploaded image file to base64 encoding"""
    img = Image.open(image_file)
    
    # Resize large images to reduce size
    max_size = 1024
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    
    # Convert to RGB if it's not already (e.g., for PNG with transparency)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Save to bytes and encode
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Function to encode audio to base64
def encode_audio_to_base64(audio_file):
    """Convert an uploaded audio file to base64 encoding"""
    audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

# Function to transcribe audio
def transcribe_audio(audio_file, model="openai"):
    """
    Transcribe audio using the Whisper model via proxy
    
    Args:
        audio_file: The uploaded audio file
        model: The model to use for transcription (openai or sambanova)
    
    Returns:
        Text transcription
    """
    url = f"{PROXY_URL}/v1/audio/transcriptions"
    
    # ファイルストリームを先頭に戻す
    audio_file.seek(0)
    
    # マルチパートフォームデータ形式でファイルを送信（プロキシサーバーの期待する形式）
    files = {
        "file": (audio_file.name, audio_file, "audio/mpeg")
    }
    
    data = {
        "model": AUDIO_TRANSCRIPTION_MODELS[model],
        "response_format": "text"
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        if "text" in response.json():
            return response.json()["text"]
        else:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error transcribing audio: {e}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Response: {e.response.text}")
        return ""

# Function to call LLM via proxy
def call_llm_proxy(model: str, messages: list, temperature: float = 0.7, **kwargs):
    """
    Call LLM through the LiteLLM proxy server
    
    Args:
        model: The model name as configured in the proxy
        messages: List of message dictionaries with role and content
        temperature: Temperature parameter for generation
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        Response from the LLM
    """
    url = f"{PROXY_URL}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs  # 追加パラメータがあれば追加
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
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "uploaded_audio" not in st.session_state:
    st.session_state.uploaded_audio = None
if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = ""

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

# Set up tabs for different input types
tab1, tab2, tab3 = st.tabs(["Text", "Image", "Audio"])

# Text input tab
with tab1:
    with st.form(key="text_form", clear_on_submit=True):
        user_input = st.text_area("You:", height=100)
        submit_text = st.form_submit_button("Send")
    
    if submit_text and user_input:
        # Add user message to history
        st.session_state.history.append({
            "role": "user", 
            "content": user_input,
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
                "model": selected_model
            })
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if TEST_MODE:
                st.info("Using test API keys? Make sure the proxy server is running.")

# Image input tab
with tab2:
    with st.form(key="image_form", clear_on_submit=True):
        image_prompt = st.text_area("Ask about an image:", height=100, 
                                  placeholder="Please describe this image.")
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        submit_image = st.form_submit_button("Send")
    
    # Display uploaded image
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if submit_image and st.session_state.uploaded_image:
        # Prepare the message content
        message_content = image_prompt if image_prompt else "Please describe this image."
        
        # Check if selected model supports vision
        is_vision_capable = any(vision_model in selected_model for vision_model in VISION_MODELS)
        
        if not is_vision_capable:
            st.warning(f"The selected model ({selected_model}) may not support image understanding. Try using a vision-capable model like gpt-4o-mini, claude-3.7-sonnet, or gemini-2.5-pro.")
        
        # Create message with image
        # Encode the image
        base64_image = encode_image_to_base64(st.session_state.uploaded_image)
        
        # Format message with image
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": message_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
        
        # Add user message to history
        st.session_state.history.append({
            "role": "user", 
            "content": message_content,
            "has_image": True
        })
        
        # Clear the image after adding to history
        image_was_uploaded = True
        temp_image = st.session_state.uploaded_image
        st.session_state.uploaded_image = None
        
        try:
            if TEST_MODE:
                # Mock response for test mode
                with st.spinner(f"Generating response from {selected_model}..."):
                    time.sleep(1)  # Simulate API call delay
                    assistant_msg = f"This is a mock description of your image from {selected_model}. I can see an image that you uploaded. You asked: {message_content}"
                    st.session_state.last_response = {
                        "choices": [{"message": {"content": assistant_msg}}]
                    }
            else:
                # Real API call via proxy
                with st.spinner(f"Generating response from {selected_model}..."):
                    # Create the messages list
                    messages = []
                    for msg in st.session_state.history:
                        if msg["role"] == "user" and msg.get("has_image") and msg == st.session_state.history[-1]:
                            # This is the latest message with image, use the formatted version
                            messages.append(user_message)
                        else:
                            # Regular message
                            messages.append({"role": msg["role"], "content": msg["content"]})
                    
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
                "model": selected_model
            })
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if TEST_MODE:
                st.info("Using test API keys? Make sure the proxy server is running.")
            else:
                st.warning("Error processing the image. The selected model may not support image understanding, or the proxy server configuration needs adjustment.")
        
        # Show the image in the UI for reference
        if temp_image:
            st.image(temp_image, caption="Processed Image", width=250)

# Audio input tab
with tab3:
    with st.form(key="audio_form", clear_on_submit=True):
        st.write("Upload an audio file to send to the model")
        audio_file = st.file_uploader("Upload audio (MP3, WAV, etc.)", type=["mp3", "wav", "m4a", "ogg"])
        audio_question = st.text_area("Question about the audio (optional):", 
                                 placeholder="What's being discussed in this audio?", height=100)
        transcribe_only = st.checkbox("Transcribe only (don't send to chat)", value=False)
        transcription_provider = st.radio("Transcription Provider:", ["openai", "sambanova"], horizontal=True)
        submit_audio = st.form_submit_button("Send to Chat")
    
    # Process audio input
    if submit_audio and audio_file:
        st.session_state.uploaded_audio = audio_file
        
        # Display audio player
        st.audio(audio_file)
        
        # トランスクリプションのみの場合
        if transcribe_only:
            with st.spinner(f"Transcribing audio with {transcription_provider}..."):
                if TEST_MODE:
                    time.sleep(2)  # Simulate API call delay
                    transcript = f"This is a mock transcription of your audio file: {audio_file.name}"
                else:
                    transcript = transcribe_audio(audio_file, model=transcription_provider)
                
                st.session_state.audio_transcript = transcript
                
                # トランスクリプション結果を表示
                st.subheader("Transcription Result:")
                st.write(transcript)
                
                # トランスクリプション結果をクリップボードにコピーするためのボタン
                if st.button("Copy to clipboard"):
                    st.write("Copied to clipboard!")
                    st.session_state.clipboard = transcript
        # 通常の音声処理（チャットに送信）
        else:
            # Add audio to chat
            with st.spinner(f"Processing audio with {selected_model}..."):
                if TEST_MODE:
                    time.sleep(2)  # Simulate API call delay
                    assistant_msg = f"This is a mock response to your audio from {selected_model}."
                    st.session_state.last_response = {
                        "choices": [{"message": {"content": assistant_msg}}]
                    }
                else:
                    # Create the user message with the audio content
                    # Encode the audio file
                    base64_audio = encode_audio_to_base64(audio_file)
                    
                    # Determine if the selected model supports audio
                    is_audio_capable = selected_model in AUDIO_MODELS
                    
                    if not is_audio_capable:
                        st.warning(f"The selected model ({selected_model}) may not support audio input. Best results with gpt-4o-audio-preview.")
                    
                    # Format the message with audio
                    question = audio_question if audio_question else "What's in this audio recording?"
                    
                    # Use correct format based on OpenAI's requirements for audio API
                    if selected_model == "gpt-4o-audio-preview":
                        # 音声ファイルの形式を取得
                        audio_format = audio_file.name.split(".")[-1].lower()
                        
                        # 音声メッセージを作成（LiteLLMのドキュメントに基づく形式）
                        user_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": base64_audio,
                                        "format": audio_format
                                    }
                                }
                            ]
                        }
                        
                        # 音声モデル用の追加パラメータ
                        extra_params = {
                            "modalities": ["text", "audio"],
                            "audio": {
                                "format": audio_format
                            }
                        }
                    else:
                        # Fallback to standard message format for non-audio models
                        user_message = {
                            "role": "user", 
                            "content": question + " (Note: Audio was attached but couldn't be processed by this model)"
                        }
                        extra_params = {}
                        st.warning(f"The model {selected_model} doesn't support direct audio input. Using text-only message.")
                    
                    # Create messages list with history
                    messages = []
                    for msg in st.session_state.history:
                        # Add previous messages as simple text
                        if msg["role"] == "user" and msg == st.session_state.history[-1]:
                            # Skip the last message as we'll add it with audio
                            continue
                        else:
                            # Regular message
                            messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Add the audio message
                    messages.append(user_message)
                    
                    # Call the proxy
                    try:
                        response = call_llm_proxy(
                            model=selected_model,
                            messages=messages,
                            temperature=temperature,
                            **extra_params
                        )
                        
                        # Store full response and extract message
                        st.session_state.last_response = response
                        assistant_msg = response["choices"][0]["message"]["content"]
                        
                        # Add user message to history
                        st.session_state.history.append({
                            "role": "user", 
                            "content": question,
                            "has_audio": True
                        })
                        
                        # Add assistant message to history
                        st.session_state.history.append({
                            "role": "assistant", 
                            "content": assistant_msg,
                            "model": selected_model
                        })
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                        st.warning("The selected model may not support audio input. Try using gpt-4o-audio-preview.")

                    if transcribe_only:
                        # Transcribe the audio
                        transcript = transcribe_audio(audio_file, transcription_provider)
                        st.session_state.audio_transcript = transcript
                        st.write("Transcript:")
                        st.write(transcript)

# Display chat history
st.subheader("Chat History")
for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        source_info = ""
        if msg.get("has_image"):
            source_info = " [From image]"
        elif msg.get("is_transcript"):
            source_info = " [From audio]"
        elif msg.get("has_audio"):
            source_info = " [From audio]"
        
        st.markdown(f"**You{source_info}:** {msg['content']}")
    else:
        model_name = msg.get("model", "Assistant")
        st.markdown(f"**{model_name}:** {msg['content']}")

# Display raw JSON if enabled
if st.session_state.show_json and st.session_state.last_response:
    st.subheader("Raw JSON Response")
    st.code(json.dumps(st.session_state.last_response, indent=2), language="json")
