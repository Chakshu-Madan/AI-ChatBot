import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from rag_engine import initialize_chatbot, ask_question, invoke_with_timeout
import uuid

app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://style-hub-ebon.vercel.app"
]

CORS(
    app,
    resources={r"/chat": {"origins": ALLOWED_ORIGINS}},
    methods=["POST"],
    allow_headers=["Content-Type"]
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],  # no global limit; we'll set it per-route
    storage_uri="memory://"
)

print("🚀 Initializing chatbot...")
qa_chain = initialize_chatbot()
print("✅ Chatbot ready!")

with open("index.html") as f:
    HTML = f.read()

@app.route("/")
def index():
    return render_template_string(HTML)

MAX_MESSAGE_LENGTH = 800

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")
def chat():
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Invalid request body"}), 400

    user_message = data.get("message", "")
    if not isinstance(user_message, str):
        return jsonify({"error": "Message must be a string"}), 400

    user_message = user_message.strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long (max {MAX_MESSAGE_LENGTH} characters)"}), 400

    session_id = data.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return jsonify({"error": "Invalid session_id"}), 400
    session_id = session_id or str(uuid.uuid4())

    try:
        result = ask_question(qa_chain, session_id, user_message)
        sources = list(set([os.path.basename(d.metadata.get("source", ""))
                            for d in result.get("source_documents", [])]))
        return jsonify({"answer": result["answer"], "sources": sources, "session_id": session_id})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Service temporarily unavailable. Please try again."}), 500

@app.route("/health")
def health():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
