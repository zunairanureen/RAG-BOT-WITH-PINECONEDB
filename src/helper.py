from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain_pinecone import PineconeVectorStore as lang_pinecone
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Pinecone
import getpass
import os
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document




def load_pdf_files(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents



def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
    text_chunks=text_splitter.split_documents(extracted_data)
    
    return text_chunks

def convert_chunk_to_doc(chunks):
    documents = [Document(page_content=chunk.page_content) for chunk in chunks]
    return documents






def load_huggingface_model():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
