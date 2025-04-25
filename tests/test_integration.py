import pytest
import os
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def set_api_key():
    """Set a mock API key for testing"""
    os.environ["OPENAI_API_KEY"] = "test-api-key"


@pytest.fixture(scope="session")
def mock_openai_response():
    """Mock the OpenAI API response"""
    with patch("openai.ChatCompletion.create") as mock_create:
        class MockResponse:
            def __init__(self):
                self.choices = [
                    type("Choice", (), {"message": type("Message", (), {"content": "I'm a mock AI response!"})})
                ]
        
        mock_create.return_value = MockResponse()
        yield mock_create


@pytest.mark.skip(reason="Integration tests require manual testing")
def test_chat_interface():
    """
    Manual testing instructions:
    
    1. Set your OpenAI API key: export OPENAI_API_KEY="your-api-key"
    2. Run the app: streamlit run app.py
    3. Enter a message in the text input field
    4. Click the "Send" button
    5. Verify that the message appears in the chat history
    6. Verify that a response from the AI appears in the chat history
    7. Enter another message and repeat steps 4-6
    8. Refresh the page and verify that the chat history is preserved
    """
    pass
