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
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

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

# Chat form with image upload
with st.form(key="chat_form", clear_on_submit=True):
    # Text input for message
    user_input = st.text_input("You:", "")
    
    # File uploader for images
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    # Submit button
    submit = st.form_submit_button("Send")

# Display uploaded image
if uploaded_file:
    st.session_state.uploaded_image = uploaded_file
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Process form submission
if submit and (user_input or st.session_state.uploaded_image):
    # Prepare the message content
    message_content = user_input if user_input else "Please describe this image."
    
    # Create a basic message object
    user_message = {
        "role": "user",
        "content": message_content
    }
    
    # If there's an image, modify the message to include it
    if st.session_state.uploaded_image:
        # Check if selected model supports vision
        is_vision_capable = any(vision_model in selected_model for vision_model in VISION_MODELS)
        
        if not is_vision_capable:
            st.warning(f"The selected model ({selected_model}) may not support image understanding. Try using a vision-capable model like gpt-4o-mini, claude-3.7-sonnet, or gemini-2.5-pro.")
        
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
        
        # Show the image in chat history
        img_placeholder = st.empty()
        img_placeholder.image(st.session_state.uploaded_image, caption="Uploaded Image", width=250)
    
    # Add user message to history
    st.session_state.history.append({
        "role": "user", 
        "content": message_content,
        "has_image": st.session_state.uploaded_image is not None
    })
    
    # Clear the image after adding to history
    image_was_uploaded = st.session_state.uploaded_image is not None
    st.session_state.uploaded_image = None
    
    try:
        if TEST_MODE:
            # Mock response for test mode
            with st.spinner(f"Generating response from {selected_model}..."):
                time.sleep(1)  # Simulate API call delay
                if image_was_uploaded:
                    assistant_msg = f"This is a mock description of your image from {selected_model}. I can see an image that you uploaded. You asked: {message_content}"
                else:
                    assistant_msg = f"This is a mock response from {selected_model}. You said: {message_content}"
                
                st.session_state.last_response = {
                    "choices": [{"message": {"content": assistant_msg}}]
                }
        else:
            # Real API call via proxy
            with st.spinner(f"Generating response from {selected_model}..."):
                # Create the messages list with appropriate format based on image presence
                if image_was_uploaded:
                    messages = []
                    for msg in st.session_state.history:
                        if msg["role"] == "user" and msg.get("has_image"):
                            # This is the latest message with image, use the formatted version
                            messages.append(user_message)
                        else:
                            # Regular message
                            messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Regular text messages
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
        elif image_was_uploaded:
            st.warning("Error processing the image. The selected model may not support image understanding, or the proxy server configuration needs adjustment.")

# Display chat history
st.subheader("Chat History")
for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
        
        # If this message had an image and it's not the most recent one being displayed
        if msg.get("has_image") and i < len(st.session_state.history) - 2:
            st.markdown("*[Image was uploaded]*")
    else:
        model_name = msg.get("model", "Assistant")
        st.markdown(f"**{model_name}:** {msg['content']}")

# Display raw JSON if enabled
if st.session_state.show_json and st.session_state.last_response:
    st.subheader("Raw JSON Response")
    st.code(json.dumps(st.session_state.last_response, indent=2), language="json")
