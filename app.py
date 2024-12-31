from flask import Flask, render_template, request, jsonify
from src.helper import load_pdf_files, text_split, load_huggingface_model, convert_chunk_to_doc
from langchain_pinecone import PineconeVectorStore as lang_pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Error handling for missing API keys
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("PINECONE_API_KEY or GROQ_API_KEY is not set in the environment.")

# Set API keys in the environment
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Load the embeddings model
embeddings_model = load_huggingface_model()

# Set up Pinecone vector store
index_name = "ai-rag"
vector_store = lang_pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings_model
)

# Set up retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=512,  # Define a sensible limit
    timeout=None,
    max_retries=2,
)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>

    Question: {input}
    """
)

# Set up chains
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        msg = request.form.get('msg', '')
        if not msg:
            return jsonify({"response": "Please enter a message."})
        try:
            response = retrieval_chain.invoke({"input": msg})
            return jsonify({"response": response['answer']})
        except Exception as e:
            return jsonify({"response": "An error occurred while processing your message."})
        
if __name__ == "__main__":
    app.run(debug=True)
