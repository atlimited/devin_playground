import pytest
import os
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set mock URL for the proxy server and API keys for testing"""
    os.environ["PROXY_URL"] = "http://localhost:8000"
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ["COHERE_API_KEY"] = "test-cohere-key"


@pytest.fixture(scope="session")
def mock_proxy_responses():
    """Mock the proxy server responses"""
    with patch("requests.get") as mock_get:
        with patch("requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "object": "list",
                "data": [
                    {"id": "gpt-4o-mini", "object": "model"},
                    {"id": "claude-3.5-sonnet", "object": "model"},
                    {"id": "command-nightly", "object": "model"}
                ]
            }
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "I'm a response from the LLM proxy!"
                        }
                    }
                ]
            }

            yield mock_get, mock_post


@pytest.mark.skip(reason="Integration tests require manual testing")
def test_chat_interface():
    """
    Manual testing instructions:

    1. Start the proxy server:
       ./run_proxy.sh

    2. In another terminal, run the app:
       streamlit run app.py

    3. Test the application with the following steps:
       a. Verify the model dropdown shows available models
       b. Select a model from the dropdown
       c. Enter a message in the text input field
       d. Click the "Send" button
       e. Verify that the message appears in the chat history
       f. Verify that a response from the selected model appears in the chat history
       g. Try different models and verify correct responses
       h. Refresh the page and verify that the chat history is preserved
       i. Stop the proxy server and verify the app switches to test mode with a warning
       j. Set the temperature slider to different values and verify it affects responses
       k. Check the "Show full JSON response" checkbox to view the raw API response


    4. API Key Testing:
       a. Set your provider API keys in the .env file:
          OPENAI_API_KEY="your-openai-key"
          ANTHROPIC_API_KEY="your-anthropic-key"
          COHERE_API_KEY="your-cohere-key"
       b. Restart the proxy server and app
       c. Verify that real API calls work correctly
    """
    pass
