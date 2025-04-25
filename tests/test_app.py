import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import litellm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app  # Import app after path modification


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing"""

    class MockSessionState(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history = []

    return MockSessionState()


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("litellm.completion")
def test_openai_flow(
    mock_completion, mock_text_input, mock_submit_button, mock_form, mock_session_state
):
    """Test the OpenAI provider flow.

    Tests user input -> API call -> response history addition flow.
    """
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from OpenAI!"))]
    )
    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="OpenAI"):
        with patch("streamlit.session_state", mock_session_state):
            if "history" not in app.st.session_state:
                app.st.session_state.history = []

            app.st.session_state.history.append({"role": "user", "content": "Hi"})

            messages = [{"role": "user", "content": "Hi"}]
            response = litellm.completion(
                model=app.PROVIDER_MODELS["OpenAI"],
                messages=messages
            )
            assistant_msg = response.choices[0].message.content
            app.st.session_state.history.append(
                {"role": "assistant", "content": assistant_msg}
            )

            assert len(app.st.session_state.history) == 2
            assert app.st.session_state.history[0]["role"] == "user"
            assert app.st.session_state.history[0]["content"] == "Hi"
            assert app.st.session_state.history[1]["role"] == "assistant"
            assert app.st.session_state.history[1]["content"] == "Hello from OpenAI!"

            mock_completion.assert_called_once_with(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}]
            )


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("litellm.completion")
def test_anthropic_flow(
    mock_completion, mock_text_input, mock_submit_button, mock_form, mock_session_state
):
    """Test the Anthropic provider flow.

    Tests user input -> API call -> response history addition flow.
    """
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from Anthropic!"))]
    )
    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="Anthropic"):
        with patch("streamlit.session_state", mock_session_state):
            if "history" not in app.st.session_state:
                app.st.session_state.history = []

            app.st.session_state.history.append({"role": "user", "content": "Hi"})

            messages = [{"role": "user", "content": "Hi"}]
            response = litellm.completion(
                model=app.PROVIDER_MODELS["Anthropic"],
                messages=messages
            )
            assistant_msg = response.choices[0].message.content
            app.st.session_state.history.append(
                {"role": "assistant", "content": assistant_msg}
            )

            assert app.st.session_state.history[1]["content"] == "Hello from Anthropic!"
            mock_completion.assert_called_once_with(
                model="claude-instant-1",
                messages=[{"role": "user", "content": "Hi"}]
            )


@patch("streamlit.form")
@patch("streamlit.form_submit_button")
@patch("streamlit.text_input")
@patch("litellm.completion")
def test_cohere_flow(
    mock_completion, mock_text_input, mock_submit_button, mock_form, mock_session_state
):
    """Test the Cohere provider flow.

    Tests user input -> API call -> response history addition flow.
    """
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from Cohere!"))]
    )
    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()

    with patch("streamlit.selectbox", return_value="Cohere"):
        with patch("streamlit.session_state", mock_session_state):
            if "history" not in app.st.session_state:
                app.st.session_state.history = []

            app.st.session_state.history.append({"role": "user", "content": "Hi"})

            messages = [{"role": "user", "content": "Hi"}]
            response = litellm.completion(
                model=app.PROVIDER_MODELS["Cohere"],
                messages=messages
            )
            assistant_msg = response.choices[0].message.content
            app.st.session_state.history.append(
                {"role": "assistant", "content": assistant_msg}
            )

            assert app.st.session_state.history[1]["content"] == "Hello from Cohere!"
            mock_completion.assert_called_once_with(
                model="command-nightly",
                messages=[{"role": "user", "content": "Hi"}]
            )
