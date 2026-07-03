import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

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

def split_documents(docs):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector DB created")
    return vs

def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def build_qa_chain(vs):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_retries=1
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a friendly, warm customer support assistant for TechCorp.
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
  Just state the answer directly and naturally, like you already knew it.
- Bad: "According to the information, support is available Monday to Friday."
- Good: "Support's available Monday to Friday, 9 AM to 6 PM - we're closed on weekends though!"

Info you have access to:
{context}

Previous conversation:
{chat_history}

Customer: {question}
You:"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
        verbose=True
    )

def initialize_chatbot():
    if os.path.exists(CHROMA_PATH):
        print("Loading existing DB...")
        vs = load_vectorstore()
    else:
        print("Building DB from documents...")
        vs = create_vectorstore(split_documents(load_documents()))
    chain = build_qa_chain(vs)
    print("Chatbot ready!")
    return chain

def ask_question(chain, question):
    print(f"[DEBUG] Starting ask_question for: {question}", flush=True)
    result = chain({"question": question})
    print(f"[DEBUG] Finished ask_question", flush=True)
    return result