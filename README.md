# Multi-Provider Chatbot with LiteLLM Proxy

A chat UI built with Streamlit and LiteLLM's proxy server, supporting multiple LLM providers. This application allows users to select an LLM model, input text, send it to the selected provider's API via a proxy server, and display the response in a chat format.

## Features

- Model selection dropdown with all available models from the proxy
- Text input form for users to enter messages
- Integration with multiple LLM providers via LiteLLM proxy
- Chat-like display of conversation history with model attribution
- Session persistence for message history
- Test mode for when proxy server is unavailable
- Unified API access through proxy server
- Temperature adjustment for response randomness
- Raw JSON response viewing option

## Architecture

This application uses a proxy-based approach to handle LLM provider differences:

1. A LiteLLM proxy server centralizes provider configuration and API key management
2. The Streamlit app communicates with the proxy server via HTTP requests
3. The proxy server routes requests to the appropriate LLM provider
4. Provider-specific details are abstracted away from the client application

Benefits of this approach:
- Centralized API key management (keys stored only on the proxy server)
- Consistent API interface for all providers
- Simplified client code with no provider-specific logic
- Easy addition of new providers without client changes
- Support for fallbacks between models

## Requirements

- Python 3.11 or higher
- Streamlit
- FastAPI and Uvicorn (for proxy server)
- LiteLLM
- Requests
- Python-dotenv
- API keys for supported providers

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your API keys in the `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   COHERE_API_KEY=your-cohere-api-key
   MISTRAL_API_KEY=your-mistral-api-key
   GEMINI_API_KEY=your-gemini-api-key
   ```

## Usage

1. Start the proxy server:
   ```
   ./run_proxy.sh
   ```

2. In another terminal, run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Select your preferred LLM model from the dropdown
4. Adjust the temperature slider if desired
5. Enter your message in the text input field
6. Click the "Send" button to submit your message
7. View the AI's response in the chat history
8. Optionally check "Show full JSON response" to see the raw API response
9. Continue the conversation by sending more messages

## Testing

### Unit Tests
Run unit tests with mocked proxy responses:
```
pytest tests/test_app.py -v
```

### Integration Tests
Run integration tests (requires a running proxy server):
```
pytest tests/test_integration.py -v
```

### Manual Testing
For manual testing:
1. Start the proxy server: `./run_proxy.sh`
2. Run the app: `streamlit run app.py`
3. Select a model from the dropdown
4. Interact with the chat interface in your browser
5. Verify that messages are sent and responses are displayed correctly
6. Try different models to ensure they all work
7. Stop the proxy server to test the fallback to test mode

## Proxy Server Configuration

The proxy server is configured using the `litellm_config.yaml` file, which defines:
- Available models and their mappings to provider models
- Fallback models for each primary model
- API key settings and environment variable references
- Server port and host settings

To customize the proxy server:
1. Edit the `litellm_config.yaml` file
2. Add or modify model mappings
3. Configure fallback options
4. Restart the proxy server to apply changes
