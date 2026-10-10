"""Application configuration, loaded from environment variables (see .env.example)."""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.lower() == "true" if value is not None else default


@dataclass(frozen=True)
class AppConfig:
    # RAG engine
    docs_dir: str
    index_persist_directory: str
    init_index: bool
    ollama_model: str
    embed_model_name: str

    # HTTP server
    http_port: int
    debug: bool

    # Reserved for future use (not yet wired into the app)
    target_url: str
    mongo_host: str
    mongo_port: int
    mongo_user: str
    mongo_pass: str

    # TLS verification is disabled by default upstream for local dev against
    # self-signed certs (e.g. a local Ollama/registry proxy). Keep it opt-in.
    disable_tls_verification: bool


def load_config() -> AppConfig:
    return AppConfig(
        docs_dir=os.getenv("DOCS_DIR", "./docs"),
        index_persist_directory=os.getenv("INDEX_PERSIST_DIRECTORY", "./data/chromadb"),
        init_index=_env_bool("INIT_INDEX", False),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        embed_model_name=os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        http_port=_env_int("HTTP_PORT", 7654),
        debug=_env_bool("FLASK_DEBUG", False),
        target_url=os.getenv("TARGET_URL", "https://open5gs.org/open5gs/docs/"),
        mongo_host=os.getenv("MONGO_HOST", "localhost"),
        mongo_port=_env_int("MONGO_PORT", 27017),
        mongo_user=os.getenv("MONGO_USER", "testuser"),
        mongo_pass=os.getenv("MONGO_PASS", "testpass"),
        disable_tls_verification=_env_bool("DISABLE_TLS_VERIFICATION", False),
    )
