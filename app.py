import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from rag_engine import initialize_chatbot, ask_question, invoke_with_timeout

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

# Debug routes
import time

@app.route("/debug-voyage")
def debug_voyage():
    import time
    try:
        from rag_engine import get_embeddings
        start = time.time()
        emb = get_embeddings()
        vector = emb.embed_query("test")
        return jsonify({
            "status": "ok",
            "took": f"{time.time()-start:.2f}s",
            "vector_len": len(vector)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug-groq")
def debug_groq():
    from langchain_groq import ChatGroq
    try:
        start = time.time()
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
        resp = llm.invoke("Say hello in 3 words")
        elapsed = time.time() - start
        return jsonify({"status": "ok", "elapsed_seconds": elapsed, "response": resp.content})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug-retrieval")
def debug_retrieval():
    from rag_engine import get_embeddings, CHROMA_PATH
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings
    import time
    try:
        print("[DEBUG] 1a: creating embeddings client", flush=True)
        start = time.time()
        embeddings = get_embeddings()
        print(f"[DEBUG] 1a done in {time.time()-start:.2f}s", flush=True)

        print("[DEBUG] 1b: opening Chroma", flush=True)
        start = time.time()
        vs = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False)
        )
        print(f"[DEBUG] 1b done in {time.time()-start:.2f}s", flush=True)

        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[DEBUG] ERROR: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug-full-chain")
def debug_full_chain():
    from rag_engine import load_vectorstore, build_qa_chain, invoke_with_timeout
    import time
    try:
        print("[DEBUG] A: loading vectorstore", flush=True)
        start = time.time()
        vs = load_vectorstore()
        print(f"[DEBUG] A done in {time.time()-start:.2f}s", flush=True)

        print("[DEBUG] B: retriever query", flush=True)
        start = time.time()
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke("What is your refund policy?")        
        print(f"[DEBUG] B done in {time.time()-start:.2f}s, num_docs={len(docs)}", flush=True)

        print("[DEBUG] C: building chain", flush=True)
        start = time.time()
        chain = build_qa_chain(vs)
        print(f"[DEBUG] C done in {time.time()-start:.2f}s", flush=True)

        print("[DEBUG] D: invoking full chain", flush=True)
        start = time.time()
        result = chain({"question": "What is your refund policy?"})
        print(f"[DEBUG] D done in {time.time()-start:.2f}s", flush=True)

        return jsonify({"status": "ok", "answer": result.get("answer")})
    except Exception as e:
        print(f"[DEBUG] ERROR: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug-embed-only")
def debug_embed_only():
    import time
    try:
        from rag_engine import get_embeddings
        start = time.time()
        emb = get_embeddings()
        print(f"[DEBUG] Model loaded in {time.time()-start:.2f}s", flush=True)

        start = time.time()
        vector = emb.embed_query("What is your refund policy?")
        print(f"[DEBUG] Embedded in {time.time()-start:.2f}s", flush=True)

        return jsonify({"status": "ok", "vector_len": len(vector)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/debug-chroma-only")
def debug_chroma_only():
    import time
    try:
        from rag_engine import load_vectorstore
        start = time.time()
        vs = load_vectorstore()
        print(f"[DEBUG] vectorstore loaded in {time.time()-start:.2f}s", flush=True)

        start = time.time()
        collection = vs._collection
        count = collection.count()
        print(f"[DEBUG] collection.count() done in {time.time()-start:.2f}s, count={count}", flush=True)

        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
