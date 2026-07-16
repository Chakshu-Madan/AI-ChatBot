import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

DOCS_PATH = "documents"

# --- Embeddings (unchanged, already proven fast and reliable) ---
class LocalEmbeddings:
    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="./fastembed_cache",
            local_files_only=True,
            threads=1,
        )

    def embed_documents(self, texts):
        return [emb.tolist() for emb in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0].tolist()

def get_embeddings():
    return LocalEmbeddings()
    
# --- In-memory index, built once at startup ---
_index_cache = None  # (texts, vectors, docs, embeddings)

def build_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    docs = split_documents(load_documents())
    texts = [d.page_content for d in docs]
    embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)
    _index_cache = (texts, vectors, docs, embeddings)
    return _index_cache

def retrieve_docs(query, k=3):
    texts, vectors, docs, embeddings = build_index()
    query_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(query_vec) + 1e-10)
    sims = (vectors @ query_vec) / (norms + 1e-10)
    top_k_idx = np.argsort(sims)[::-1][:k]
    return [docs[i] for i in top_k_idx]

# --- LLM + manual chat flow (replaces ConversationalRetrievalChain) ---
CUSTOM_PROMPT_TEMPLATE = """You are a friendly, warm customer support assistant for TechCorp.
Talk like a real helpful person, not a robot - use contractions (we're, don't, that's),
keep it conversational and brief.

Rules:
- Answer using ONLY the info below. Never make things up.
- If the answer isn't in the info, just say so naturally - something like
  "Hmm, I don't have that on hand, but feel free to email support@techcorp.com and they can help!"
  Don't explain what the document does or doesn't mention - just say you don't know and offer a next step.
- Keep answers short and to the point, like a real chat message - 1 to 3 sentences usually.
- Never sound like you're reading from a manual.
- NEVER say phrases like "according to the information", "based on what's provided", "the document states",
  "the information says", or anything that reveals you're reading from a knowledge base.

Info you have access to:
{context}

Previous conversation:
{chat_history}

Customer: {question}
You:"""

def build_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_retries=1,
    )

def invoke_with_timeout(func, *args, timeout=15, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except FutureTimeoutError:
        executor.shutdown(wait=False)
        raise TimeoutError(f"Call timed out after {timeout}s")
    
def load_documents():
    docs = []
    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)
        if file.endswith(".txt"):
            docs.extend(TextLoader(path).load())
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    print(f"Loaded {len(docs)} pages")
    return docs

def split_documents(docs):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

_session_memories = {}
MAX_SESSIONS = 500  # simple cap so this dict doesn't grow forever on a long-running server

def get_memory(session_id):
    if session_id not in _session_memories:
        if len(_session_memories) >= MAX_SESSIONS:
            oldest = next(iter(_session_memories))  # evict oldest session (dicts preserve insertion order)
            del _session_memories[oldest]
        _session_memories[session_id] = []
    return _session_memories[session_id]

def initialize_chatbot():
    build_index()  # warm the index once at startup
    llm = build_llm()
    return {"llm": llm}

def ask_question(chatbot_state, session_id, question):
    llm = chatbot_state["llm"]
    memory = get_memory(session_id)

    # Build a richer query for retrieval that includes recent context
    if memory:
        last_q, last_a = memory[-1]
        retrieval_query = f"{last_q} {last_a} {question}"
    else:
        retrieval_query = question

    print(f"[DEBUG] Retrieving docs for: {question}", flush=True)
    docs = retrieve_docs(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    history_text = "\n".join(f"Customer: {q}\nYou: {a}" for q, a in memory[-3:])

    prompt = CUSTOM_PROMPT_TEMPLATE.format(
        context=context, chat_history=history_text, question=question
    )

    print(f"[DEBUG] Calling LLM", flush=True)
    try:
        response = invoke_with_timeout(llm.invoke, prompt, timeout=30)
        answer = response.content
    except TimeoutError:
        print(f"[DEBUG] LLM call timed out", flush=True)
        return {
            "answer": "I'm getting a lot of questions right now — give me a few seconds and try again!",
            "source_documents": [],
        }

    memory.append((question, answer))
    print(f"[DEBUG] Done", flush=True)
    return {"answer": answer, "source_documents": docs}