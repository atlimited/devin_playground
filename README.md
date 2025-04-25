# Multi-Provider Chatbot with LiteLLM

A chat UI built with Streamlit and LiteLLM, supporting multiple LLM providers. This application allows users to select an LLM provider, input text, send it to the selected provider's API, and display the response in a chat format.

## Features

- Provider selection dropdown (OpenAI, Anthropic, Cohere)
- Text input form for users to enter messages
- Integration with multiple LLM providers via LiteLLM
- Chat-like display of conversation history
- Session persistence for message history
- Unit tests with mocked API calls
- Integration tests for UI interaction

## Requirements

- Python 3.11 or higher
- Streamlit
- OpenAI Python SDK
- LiteLLM
- API keys for supported providers

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your API keys as environment variables:
   ```
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export COHERE_API_KEY="your-cohere-api-key"
   ```

## Usage

1. Run the application with:
   ```
   streamlit run app.py
   ```
2. Select your preferred LLM provider from the dropdown
3. Enter your message in the text input field
4. Click the "Send" button to submit your message
5. View the AI's response in the chat history
6. Continue the conversation by sending more messages

## Testing

### Unit Tests
Run unit tests with mocked LLM APIs:
```
pytest tests/test_app.py -v
```

### Integration Tests
Run integration tests (requires a running Streamlit app):
```
pytest tests/test_integration.py -v
```

### Manual Testing
For manual testing:
1. Set your API keys: 
   ```
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export COHERE_API_KEY="your-cohere-api-key"
   ```
2. Run the app: `streamlit run app.py`
3. Select a provider from the dropdown
4. Interact with the chat interface in your browser
5. Verify that messages are sent and responses are displayed correctly
6. Try different providers to ensure they all work
