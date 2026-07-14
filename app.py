import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
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

print("🚀 Initializing chatbot...")
qa_chain = initialize_chatbot()
print("✅ Chatbot ready!")

with open("index.html") as f:
    HTML = f.read()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
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
