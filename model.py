from langchain_huggingface import HuggingFaceEmbeddings
import transformers
import torch
from sentence_transformers import SentenceTransformer



def load_huggingface_model():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
