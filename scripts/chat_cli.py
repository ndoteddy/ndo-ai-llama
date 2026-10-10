"""Interactive REPL for chatting with the RAG engine from the terminal.

Run with: python -m scripts.chat_cli
"""
import logging
import sys

from app.config import load_config
from app.rag import RagEngine

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    config = load_config()
    engine = RagEngine(
        docs_dir=config.docs_dir,
        index_persist_directory=config.index_persist_directory,
        init_index=config.init_index,
        ollama_model=config.ollama_model,
        embed_model_name=config.embed_model_name,
    )
    engine.initialize()

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        print(engine.chat(question, user_id="cli"))


if __name__ == "__main__":
    main()
