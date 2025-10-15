"""Integration tests for the `QABot` against sample documentation."""

from pathlib import Path

from src.qa_bot import QABot


def test_bot_answers_from_requests_doc():
    """Check that the bot references the installation command from docs."""
    project_root = Path(__file__).resolve().parents[1]
    bot = QABot(documents_path=project_root / "data" / "documents" / "sample_docs")
    answer = bot.answer("How do I install the Python requests library?")
    assert "pip install requests" in answer.response.lower()
    assert answer.context
