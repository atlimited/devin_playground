# Streamlit Chatbot with OpenAI

A simple chat UI built with Streamlit and the OpenAI API. This application allows users to input text, send it to the OpenAI ChatCompletion API, and display the API's response in a chat format.

## Features

- Text input form for users to enter messages
- Integration with OpenAI ChatCompletion API
- Chat-like display of conversation history
- Session persistence for message history

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

Run the application with:
```
streamlit run app.py
```

## Testing

Run unit tests:
```
pytest tests/test_app.py
```

Run integration tests:
```
pytest tests/test_integration.py
```
