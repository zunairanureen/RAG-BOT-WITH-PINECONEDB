from helper import load_pdf_files, text_split, load_huggingface_model, convert_chunk_to_doc
from langchain_pinecone import PineconeVectorStore as lang_pinecone
from pinecone import ServerlessSpec
#from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from pinecone import Pinecone
import os

PINECONE_API_KEY="pcsk_6G1qcw_3QJqAPqQKs2SjgVJpmrxwBX3WLgsKomxmFLTj6sgm2L21YTcAHDQhGTNcYZxtoo"
index_name="ai-rag"

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY



extracted_data=load_pdf_files(file_path="C:\\Users\\Ihtsham Mehmood\\Documents\\Downloads\\rag\\MEDICAL-CHATBOT\\data\\Steps_toward_Artificial_Intelligence.pdf")
chunks = text_split(extracted_data=extracted_data)
documents=convert_chunk_to_doc(chunks=chunks)
embeddings_model=load_huggingface_model()

pc=Pinecone(
    api_key=PINECONE_API_KEY,
    
)
pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

vector = lang_pinecone.from_documents(
    documents=documents,
    embedding=embeddings_model, 
    index_name=index_name
)


