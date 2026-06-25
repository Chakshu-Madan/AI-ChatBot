
import os, time
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

DOCS_PATH = "documents"
CHROMA_PATH = "chroma_db"

# Local embedding model - downloads once, then runs with no API calls
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    print("Vector DB created (local embeddings)")
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
        temperature=0.2,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
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
    return chain({"question": question})
