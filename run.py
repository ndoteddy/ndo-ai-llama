"""Entrypoint: builds the RAG index once, then serves the Flask API."""
from app import create_app
from app.config import load_config

config = load_config()
app = create_app(config=config)

if __name__ == "__main__":
    app.config["RAG_ENGINE"].initialize()
    app.run(host="0.0.0.0", port=config.http_port, debug=config.debug)
