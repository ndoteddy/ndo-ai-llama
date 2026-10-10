"""Flask application factory."""
import logging
import os
import ssl
import sys
import typing

from flask import Flask
from flask_cors import CORS

from app.config import AppConfig, load_config

if typing.TYPE_CHECKING:
    from app.rag import RagEngine


def _configure_logging() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_app(rag_engine: "RagEngine | None" = None, config: AppConfig | None = None) -> Flask:
    """Build the Flask app.

    `rag_engine` can be injected (e.g. a fake with a `.chat()` method) so
    route tests never need to touch llama_index/chromadb/Ollama at all. The
    real `RagEngine` (and its heavy llama_index/chromadb/torch imports) is
    only imported when no fake is supplied.
    """
    _configure_logging()
    config = config or load_config()

    if config.disable_tls_verification:
        os.environ["CURL_CA_BUNDLE"] = ""
        ssl._create_default_https_context = ssl._create_unverified_context

    app = Flask(__name__)
    CORS(app)

    app.config["APP_CONFIG"] = config

    if rag_engine is None:
        from app.rag import RagEngine

        rag_engine = RagEngine(
            docs_dir=config.docs_dir,
            index_persist_directory=config.index_persist_directory,
            init_index=config.init_index,
            ollama_model=config.ollama_model,
            embed_model_name=config.embed_model_name,
        )
    app.config["RAG_ENGINE"] = rag_engine

    from app.routes import bp as api_bp

    app.register_blueprint(api_bp)

    return app
