"""Retrieval-augmented generation engine: Ollama LLM + HuggingFace embeddings + Chroma index."""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

QA_TEMPLATE = PromptTemplate(
    "Imagine you are an advanced AI in customer service "
    "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
    "Here is some context related to the query:\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, please respond to the following inquiry with detailed references "
    "precedents, or principles where appropriate:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly, starting with the phrase 'As per my understanding."
)

COLLECTION_NAME = "iollamas"


class RagEngine:
    """Encapsulates the LLM, embedding model, and vector index as instance state.

    Deliberately avoids llama_index's global `Settings` singleton so each
    engine instance is independent and the LLM/embedding model can be
    injected for testing without touching global state.
    """

    def __init__(
        self,
        docs_dir: str | Path = "./docs",
        index_persist_directory: str | Path = "./data/chromadb",
        init_index: bool = False,
        llm: LLM | None = None,
        embed_model: BaseEmbedding | None = None,
        ollama_model: str = "llama2",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_persist_directory = Path(index_persist_directory)
        self.init_index = init_index
        self.llm = llm if llm is not None else Ollama(model=ollama_model, request_timeout=300.0)
        self.embed_model = embed_model if embed_model is not None else HuggingFaceEmbedding(model_name=embed_model_name)
        self._query_engine = None

    def initialize(self) -> None:
        """Build (or load) the index and query engine once. Safe to call multiple times."""
        if self._query_engine is not None:
            return

        index = self._build_index()
        self._query_engine = index.as_query_engine(
            llm=self.llm, text_qa_template=QA_TEMPLATE, similarity_top_k=3
        )

    def _build_index(self) -> VectorStoreIndex:
        """Load the persisted index if one exists, otherwise build it from `docs_dir`.

        Split out as its own method so tests can monkeypatch it without
        touching chromadb/llama_index internals.
        """
        chroma_client = chromadb.PersistentClient(path=str(self.index_persist_directory))
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        if self.init_index or collection.count() == 0:
            documents = SimpleDirectoryReader(input_dir=str(self.docs_dir), recursive=True).load_data()
            logger.info("building index from %d documents in `%s`", len(documents), self.docs_dir)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embed_model
            )

        logger.info("loading existing index from `%s`", self.index_persist_directory)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embed_model)

    def chat(self, question: str, user_id: str) -> str:
        if self._query_engine is None:
            raise RuntimeError("RagEngine.chat() called before initialize()")

        logger.info("question `%s` from user `%s`", question, user_id)
        response = self._query_engine.query(question)
        logger.info("response: %s", response)
        return response.response


__all__ = ["RagEngine"]
