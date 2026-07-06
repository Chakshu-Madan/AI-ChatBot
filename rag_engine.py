import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()

DOCS_PATH = "documents"
CHROMA_PATH = "chroma_db"

def get_embeddings():
    return VoyageAIEmbeddings(
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        model="voyage-3-lite",  # smallest, fastest, lowest memory
    )

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

# def split_documents(docs):
#     chunks = RecursiveCharacterTextSplitter(
#         chunk_size=150,
#         chunk_overlap=20
#     ).split_documents(docs)
#     print(f"Split into {len(chunks)} chunks")
#     return chunks

def split_documents(docs):
    full_text = "\n".join([d.page_content for d in docs])
    blocks = [b.strip() for b in full_text.split("\n\n") if b.strip()]

    from langchain_core.documents import Document
    chunks = [Document(page_content=block) for block in blocks]

    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client
    )
    print("Vector DB created")
    return vs

def load_vectorstore():
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return Chroma(
        client=client,
        embedding_function=embeddings
    )

# ============================================
# ORIGINAL VERSION (memory + ConversationalRetrievalChain)
# Commented out temporarily for demo reliability - restore later

# def build_qa_chain(vs):
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         temperature=0.4,
#         max_retries=1
#     )
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer"
#     )

#     custom_prompt = PromptTemplate(
#         input_variables=["context", "question", "chat_history"],
#         template="""You are a friendly, warm customer support assistant for TechCorp.
# Talk like a real helpful person, not a robot - use contractions (we're, don't, that's),
# keep it conversational and brief.

# Rules:
# - Answer using ONLY the info below. Never make things up.
# - If the answer isn't in the info, just say so naturally - something like
#   "Hmm, I don't have that on hand, but feel free to email support@techcorp.com and they can help!"
#   Don't explain what the document does or doesn't mention - just say you don't know and offer a next step.
# - Keep answers short and to the point, like a real chat message - 1 to 3 sentences usually.
# - Never sound like you're reading from a manual.
# - NEVER say phrases like "according to the information", "based on what's provided", "the document states",
#   "the information says", or anything that reveals you're reading from a knowledge base.
#   Just state the answer directly and naturally, like you already knew it.
# - Bad: "According to the information, support is available Monday to Friday."
# - Good: "Support's available Monday to Friday, 9 AM to 6 PM - we're closed on weekends though!"

# Info you have access to:
# {context}

# Previous conversation:
# {chat_history}

# Customer: {question}
# You:"""
#     )

#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vs.as_retriever(search_kwargs={"k": 3}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": custom_prompt},
#         return_source_documents=True,
#         verbose=True
#     )

# def initialize_chatbot():
#     if os.path.exists(CHROMA_PATH):
#         print("Loading existing DB...")
#         vs = load_vectorstore()
#     else:
#         print("Building DB from documents...")
#         vs = create_vectorstore(split_documents(load_documents()))
#     chain = build_qa_chain(vs)
#     print("Chatbot ready!")
#     return chain

# def ask_question(chain, question):
#     print(f"[DEBUG] Starting ask_question for: {question}", flush=True)
#     result = chain({"question": question})
#     print(f"[DEBUG] Finished ask_question", flush=True)
#     return result
    
# ============================================

# ============================================
# TEMPORARY VERSION (no memory) - for demo
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4
    )

CUSTOM_PROMPT = """You are a customer support assistant for TechCorp. Answer the customer's question using ONLY the information below.

Information:
{context}

Instructions:
- Answer the specific question asked, using the relevant information above.
- Keep it brief and conversational - 1 to 2 sentences.
- Do NOT give a generic greeting or ask "how can I help you" - just answer directly.
- If the information above doesn't contain the answer, say: "I don't have that on hand, but feel free to email support@techcorp.com!"

Customer question: {question}

Answer:"""

import numpy as np

_docs_cache = None
_vectors_cache = None

def _load_all_chunks_and_vectors(vs, embeddings):
    global _docs_cache, _vectors_cache
    if _docs_cache is not None:
        return _docs_cache, _vectors_cache

    print("[DEBUG] Loading all chunks into memory (one-time)", flush=True)
    raw = vs.get(include=["documents", "embeddings", "metadatas"])
    _docs_cache = raw["documents"]
    _vectors_cache = np.array(raw["embeddings"])
    print(f"[DEBUG] Loaded {len(_docs_cache)} chunks into memory", flush=True)
    return _docs_cache, _vectors_cache

def initialize_chatbot():
    if os.path.exists(CHROMA_PATH):
        print("Loading existing DB...")
        vs = load_vectorstore()
    else:
        print("Building DB from documents...")
        vs = create_vectorstore(split_documents(load_documents()))
    llm = get_llm()
    embeddings = get_embeddings()
    docs, vectors = _load_all_chunks_and_vectors(vs, embeddings)
    print("Chatbot ready!")
    return {"vs": vs, "llm": llm, "embeddings": embeddings, "docs": docs, "vectors": vectors}

def ask_question(chatbot, question):
    llm = chatbot["llm"]
    embeddings = chatbot["embeddings"]
    docs = chatbot["docs"]
    vectors = chatbot["vectors"]

    print(f"[DEBUG] Embedding question: {question}", flush=True)
    q_vec = np.array(embeddings.embed_query(question))
    print("[DEBUG] Question embedded", flush=True)

    sims = vectors @ q_vec / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
    top_idx = np.argsort(sims)[::-1][:3]
    top_chunks = [docs[i] for i in top_idx]
    print(f"[DEBUG] Retrieved top {len(top_chunks)} chunks manually", flush=True)
    print(f"[DEBUG] Chunk contents: {top_chunks}", flush=True)

    context = "\n\n".join(top_chunks)
    prompt = CUSTOM_PROMPT.format(context=context, question=question)

    print("[DEBUG] Calling LLM", flush=True)
    response = llm.invoke(prompt)
    print("[DEBUG] LLM responded", flush=True)

    class FakeDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    return {
        "answer": response.content,
        "source_documents": [FakeDoc(c) for c in top_chunks]
    }
# ============================================