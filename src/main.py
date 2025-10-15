from __future__ import annotations

import argparse
from pathlib import Path

from .qa_bot import QABot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple documentation QA bot")
    parser.add_argument("question", help="Question to ask the bot")
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        help="Override the path to the Markdown documentation directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to retrieve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = QABot(documents_path=args.documents, top_k=args.top_k)
    answer = bot.answer(args.question)
    print(answer.response)
    if answer.context:
        print("\nMost relevant documents:")
        for item in answer.context:
            print(f"- {item.document.title} (score={item.score:.3f})")


if __name__ == "__main__":
    main()
