"""HTTP routes for the RAG chat API."""
import logging

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/question", methods=["POST"])
def post_question():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    question = body.get("question")
    user_id = body.get("user_id")
    if not question or not user_id:
        return jsonify({"error": "`question` and `user_id` are required"}), 400

    rag_engine = current_app.config["RAG_ENGINE"]
    answer = rag_engine.chat(question, user_id)

    return jsonify({"answer": answer}), 200
