import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import app  # Import app at the top level

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
@patch("openai.ChatCompletion.create")
def test_chat_flow(mock_create, mock_text_input, mock_submit_button, mock_form, mock_session_state):
    """Test the chat flow: user input -> API call -> response history addition"""
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello!"))]
    )
    mock_text_input.return_value = "Hi"
    mock_submit_button.return_value = True
    mock_form.return_value.__enter__.return_value = MagicMock()
    
    with patch("streamlit.session_state", mock_session_state):
        if "history" not in app.st.session_state:
            app.st.session_state.history = []
        
        app.st.session_state.history.append({"role": "user", "content": "Hi"})
        
        response = app.openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=app.st.session_state.history,
            temperature=0.7,
        )
        assistant_msg = response.choices[0].message.content
        app.st.session_state.history.append(
            {"role": "assistant", "content": assistant_msg}
        )
        
        assert len(app.st.session_state.history) == 2
        assert app.st.session_state.history[0]["role"] == "user"
        assert app.st.session_state.history[0]["content"] == "Hi"
        assert app.st.session_state.history[1]["role"] == "assistant"
        assert app.st.session_state.history[1]["content"] == "Hello!"
        
        mock_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=app.st.session_state.history,
            temperature=0.7,
        )
