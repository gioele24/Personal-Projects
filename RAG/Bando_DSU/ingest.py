import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "db"

docs = list()
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
        pdf_docs = loader.load()
        docs.extend(pdf_docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200) 
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_DIR
)

