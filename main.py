import uvicorn
import httpx
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- Langchain & RAG Imports (Updated) ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- NEW: Import PyPDFLoader ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings # Fix deprecation

# --- Load Environment Variables ---
# This loads the .env file (GEMINI_API_KEY)
load_dotenv()

# --- Global Variables ---
db = None  # This will hold our in-memory vector database
embeddings = None # This will hold our embedding model
gemini_api_key = os.getenv("GEMINI_API_KEY") # Load key from .env

# This is our "constrained prompt" - it's key to improving accuracy.
SYSTEM_PROMPT = """
You are an AI assistant providing information on Malaysian law based *only* on the provided text snippets.
Your task is to answer the user's question clearly and concisely using *only* the information from the "Legal Context" below.

- Do not use any of your general knowledge.
- If the answer cannot be found in the "Legal Context," you MUST state: "I'm sorry, I do not have information on that specific topic in my current knowledge base."
- Quote the relevant section if it helps clarify the answer.

Legal Context:
---
{context}
---

User Question:
{question}
"""

# --- RAG Setup (on Server Startup) ---
# This 'lifespan' function replaces the deprecated '@app.on_event("startup")'
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts.
    It now scans a 'pdf_knowledge_base' directory, loads all PDFs,
    chunks them, and builds the in-memory FAISS database.
    """
    global db, embeddings, gemini_api_key

    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        print("Please get a key from Google AI Studio and add it to your .env file.")
    else:
        print("GEMINI_API_KEY loaded successfully.")

    print("Loading embedding model...")
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding model loaded.")

    try:
        # --- NEW: PDF Loading Logic ---
        pdf_directory = "pdf_knowledge_base"
        all_documents = []
        
        if not os.path.exists(pdf_directory):
            print(f"Warning: Directory '{pdf_directory}' not found. Creating it.")
            os.makedirs(pdf_directory)
            print("Please add your PDF files to this directory and restart the server.")
        
        print(f"Scanning for PDF files in '{pdf_directory}'...")
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_directory, filename)
                print(f"Loading document: {filename}")
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)
        
        if not all_documents:
            print("Warning: No PDF documents were loaded. The knowledge base is empty.")
            # We can let the server run, but RAG will find no context.
        else:
            print(f"Loaded {len(all_documents)} pages from PDF files.")

            # --- This is the "smarter chunking" ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(all_documents)
            
            print(f"Split documents into {len(texts)} text chunks.")

            # Create the FAISS vector store
            db = FAISS.from_documents(texts, embeddings)
            print("In-memory FAISS vector database created successfully.")

    except Exception as e:
        print(f"Error initializing vector database: {e}")
    
    yield
    # Code below yield runs on shutdown (if any)
    print("Shutting down...")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan) # Use the new lifespan manager

origins = ["*"] # For PoC
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str


# --- Gemini API Call ---
async def get_gemini_response(context: str, question: str) -> str:
    """
    Calls the Gemini API with the constrained RAG prompt.
    """
    
    if not gemini_api_key:
        return "Server-side error: The GEMINI_API_KEY is missing. Please check the server configuration."

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={gemini_api_key}"

    # Construct the final prompt
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
            
            response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)

            result = response.json()
            
            if (result.get('candidates') and 
                result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts')):
                
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                # Handle cases where the response structure is unexpected
                print(f"Unexpected API response structure: {result}")
                return "Sorry, I received an unusual response from the AI. Please try again."

    except httpx.HTTPStatusError as e:
        # This will now clearly show the 403 error if the key is wrong
        print(f"HTTP error calling Gemini: {e.response.text}")
        return "Sorry, there was an error communicating with the AI service. Check the server logs."
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return "Sorry, an internal error occurred. Please try again later."


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Legal InfoBot Backend is running!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    The main RAG chat endpoint.
    """
    user_message = request.message
    
    if not db:
        raise HTTPException(status_code=503, detail="Vector database is not ready. Check server logs.")
    
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Server is missing API key configuration.")

    try:
        # 1. Retrieve: Find relevant docs from FAISS
        docs = db.similarity_search(user_message, k=3)
        
        # 2. Augment: Combine the content of the chunks
        context_chunks = "\n---\n".join([doc.page_content for doc in docs])
        
        if not context_chunks.strip():
             # This handles if the DB is empty and returns no docs
            context_chunks = "No relevant information found."

        # 3. Generate: Call Gemini with the context and question
        bot_response = await get_gemini_response(context_chunks, user_message)
        
        return {"response": bot_response}

    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat message.")


# --- Run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

