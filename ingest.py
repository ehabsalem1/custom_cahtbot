# ingest.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader("data/knowledge", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=api_key)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")
