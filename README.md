# Streamlit Chatbot with OpenAI

A simple chat UI built with Streamlit and the OpenAI API. This application allows users to input text, send it to the OpenAI ChatCompletion API, and display the API's response in a chat format.

## Features

- Text input form for users to enter messages
- Integration with OpenAI ChatCompletion API
- Chat-like display of conversation history
- Session persistence for message history
- Unit tests with mocked API calls
- Integration tests for UI interaction

## Requirements

- Python 3.11 or higher
- Streamlit
- OpenAI Python SDK
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

1. Run the application with:
   ```
   streamlit run app.py
   ```
2. Enter your message in the text input field
3. Click the "Send" button to submit your message
4. View the AI's response in the chat history
5. Continue the conversation by sending more messages

## Testing

### Unit Tests
Run unit tests with mocked OpenAI API:
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
1. Set your OpenAI API key: `export OPENAI_API_KEY="your-api-key"`
2. Run the app: `streamlit run app.py`
3. Interact with the chat interface in your browser
4. Verify that messages are sent and responses are displayed correctly
