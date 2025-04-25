import os
import sys
import pytest
import requests
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app  # noqa: E402


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing"""

    class MockSessionState(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history = []
            self.last_response = None
            self.show_json = False

    return MockSessionState()


@pytest.fixture
def mock_proxy_models_response():
    """Mock the proxy server models response"""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "claude-3.5-sonnet", "object": "model"},
            {"id": "command-nightly", "object": "model"}
        ]
    }


@pytest.fixture
def mock_proxy_completion_response():
    """Mock the proxy server completion response for OpenAI"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Hello from OpenAI!"
                }
            }
        ]
    }


@pytest.fixture
def mock_anthropic_completion_response():
    """Mock the proxy server completion response for Anthropic"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Hello from Anthropic!"
                }
            }
        ]
    }


@pytest.fixture
def mock_cohere_completion_response():
    """Mock the proxy server completion response for Cohere"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Hello from Cohere!"
                }
            }
        ]
    }


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("requests.post")
@patch("requests.get")
def test_openai_flow(
    mock_get, mock_post, mock_text_input, mock_submit_button, mock_form,
    mock_session_state, mock_proxy_models_response, mock_proxy_completion_response
):
    """Test the OpenAI model flow using the proxy.

    Tests user input -> API call -> response history addition flow.
    """
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_proxy_models_response
    )

    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_proxy_completion_response
    )

    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="gpt-4o-mini"):
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.spinner"):
                if "history" not in app.st.session_state:
                    app.st.session_state.history = []
                if "last_response" not in app.st.session_state:
                    app.st.session_state.last_response = None

                app.st.session_state.history.append({
                    "role": "user",
                    "content": "Hi"
                })

                messages = [{"role": "user", "content": "Hi"}]
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.7
                }

                response = app.call_llm_proxy(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7
                )

                app.st.session_state.last_response = response
                assistant_msg = response["choices"][0]["message"]["content"]

                app.st.session_state.history.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "model": "gpt-4o-mini"
                })

                assert len(app.st.session_state.history) == 2
                assert app.st.session_state.history[0]["role"] == "user"
                assert app.st.session_state.history[0]["content"] == "Hi"
                assert app.st.session_state.history[1]["role"] == "assistant"
                assert (
                    app.st.session_state.history[1]["content"] == "Hello from OpenAI!"
                )
                assert app.st.session_state.history[1]["model"] == "gpt-4o-mini"

                mock_post.assert_called_once_with(
                    f"{app.PROXY_URL}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("requests.post")
@patch("requests.get")
def test_anthropic_flow(
    mock_get, mock_post, mock_text_input, mock_submit_button, mock_form,
    mock_session_state, mock_proxy_models_response, mock_anthropic_completion_response
):
    """Test the Anthropic model flow using the proxy.

    Tests user input -> API call -> response history addition flow.
    """
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_proxy_models_response
    )

    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_anthropic_completion_response
    )

    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="claude-3.5-sonnet"):
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.spinner"):
                if "history" not in app.st.session_state:
                    app.st.session_state.history = []
                if "last_response" not in app.st.session_state:
                    app.st.session_state.last_response = None

                app.st.session_state.history.append({
                    "role": "user",
                    "content": "Hi"
                })

                messages = [{"role": "user", "content": "Hi"}]
                payload = {
                    "model": "claude-3.5-sonnet",
                    "messages": messages,
                    "temperature": 0.7
                }

                response = app.call_llm_proxy(
                    model="claude-3.5-sonnet",
                    messages=messages,
                    temperature=0.7
                )

                app.st.session_state.last_response = response
                assistant_msg = response["choices"][0]["message"]["content"]

                app.st.session_state.history.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "model": "claude-3.5-sonnet"
                })

                assert (
                    app.st.session_state.history[1]["content"] ==
                    "Hello from Anthropic!"
                )
                assert app.st.session_state.history[1]["model"] == "claude-3.5-sonnet"

                mock_post.assert_called_once_with(
                    f"{app.PROXY_URL}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("requests.post")
@patch("requests.get")
def test_cohere_flow(
    mock_get, mock_post, mock_text_input, mock_submit_button, mock_form,
    mock_session_state, mock_proxy_models_response, mock_cohere_completion_response
):
    """Test the Cohere model flow using the proxy.

    Tests user input -> API call -> response history addition flow.
    """
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_proxy_models_response
    )

    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: mock_cohere_completion_response
    )

    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="command-nightly"):
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.spinner"):
                if "history" not in app.st.session_state:
                    app.st.session_state.history = []
                if "last_response" not in app.st.session_state:
                    app.st.session_state.last_response = None

                app.st.session_state.history.append({
                    "role": "user",
                    "content": "Hi"
                })

                messages = [{"role": "user", "content": "Hi"}]
                payload = {
                    "model": "command-nightly",
                    "messages": messages,
                    "temperature": 0.7
                }

                response = app.call_llm_proxy(
                    model="command-nightly",
                    messages=messages,
                    temperature=0.7
                )

                app.st.session_state.last_response = response
                assistant_msg = response["choices"][0]["message"]["content"]

                app.st.session_state.history.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "model": "command-nightly"
                })

                assert (
                    app.st.session_state.history[1]["content"] == "Hello from Cohere!"
                )
                assert app.st.session_state.history[1]["model"] == "command-nightly"

                mock_post.assert_called_once_with(
                    f"{app.PROXY_URL}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("requests.get")
def test_connection_error(
    mock_get, mock_text_input, mock_submit_button, mock_form,
    mock_session_state
):
    """Test handling of connection errors to the proxy server"""
    mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

    with patch("streamlit.warning") as mock_warning:
        with patch("streamlit.session_state", mock_session_state):
            with patch("streamlit.spinner"):
                models = app.get_available_models()

                assert models == {"data": []}

                # The warning is shown in the app.py, not in the function
                mock_warning.assert_not_called()
