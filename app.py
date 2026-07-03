import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from rag_engine import initialize_chatbot, ask_question

app = Flask(__name__)
CORS(app)

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
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    try:
        result = ask_question(qa_chain, user_message)
        sources = list(set([os.path.basename(d.metadata.get("source", ""))
                            for d in result.get("source_documents", [])]))
        return jsonify({"answer": result["answer"], "sources": sources})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Service temporarily unavailable. Please try again."}), 500

@app.route("/health")
def health():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

