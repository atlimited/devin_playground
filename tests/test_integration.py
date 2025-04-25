import pytest
import os
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def set_api_keys():
    """Set mock API keys for testing"""
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ["COHERE_API_KEY"] = "test-cohere-key"


@pytest.fixture(scope="session")
def mock_litellm_responses():
    """Mock the LiteLLM client responses"""
    with patch("litellm.OpenAIClient.chat") as mock_openai:
        with patch("litellm.AnthropicClient.chat") as mock_anthropic:
            with patch("litellm.CohereClient.chat") as mock_cohere:
                class MockResponse:
                    def __init__(self, provider):
                        self.choices = [
                            type("Choice", (), {
                                "message": type("Message", (), {
                                    "content": f"I'm a mock {provider} response!"
                                })
                            })
                        ]

                mock_openai.return_value = MockResponse("OpenAI")
                mock_anthropic.return_value = MockResponse("Anthropic")
                mock_cohere.return_value = MockResponse("Cohere")
                yield mock_openai, mock_anthropic, mock_cohere


@pytest.mark.skip(reason="Integration tests require manual testing")
def test_chat_interface():
    """
    Manual testing instructions:

    1. Set your provider API keys:
       export OPENAI_API_KEY="your-openai-key"
       export ANTHROPIC_API_KEY="your-anthropic-key"
       export COHERE_API_KEY="your-cohere-key"
    2. Run the app: streamlit run app.py
    3. Select a provider from the dropdown
    4. Enter a message in the text input field
    5. Click the "Send" button
    6. Verify that the message appears in the chat history
    7. Verify that a response from the selected provider appears in the chat history
    8. Try different providers and verify correct responses
    9. Refresh the page and verify that the chat history is preserved
    """
    pass
