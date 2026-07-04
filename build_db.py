from rag_engine import load_documents, split_documents, create_vectorstore

create_vectorstore(split_documents(load_documents()))