import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- RAG Specific Imports ---
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# -----------------------------

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

# --- Global variables for RAG ---
# In-memory database
db = None
# Embedding model
embeddings = None
# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
# -------------------------------

# --- Configuration ---
PDF_DIRECTORY = "pdf_knowledge_base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# This is the confidence threshold. 0 is perfect match, 1 is bad match.
# We will use this to decide WHEN to use the general web search.
RAG_RELEVANCE_THRESHOLD = 0.8
# ---------------------

def setup_rag_pipeline():
    """
    Loads PDFs, splits them into chunks, creates embeddings,
    and stores them in an in-memory FAISS vector database.
    """
    global db, embeddings
    
    print("Loading embedding model...")
    # Use CPU for embeddings, as RPi doesn't have a CUDA GPU
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model loaded.")

    # Load all PDF files from the directory
    pdf_files = glob.glob(f"{PDF_DIRECTORY}/*.pdf")
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}. The RAG context will be empty.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Loading documents...")
    
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {pdf_path}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    if not all_docs:
        print("No documents were successfully loaded. Aborting RAG setup.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Total document splits created: {len(splits)}")

    # Create the FAISS vector database from the chunks
    print("Creating in-memory FAISS vector database... (This may take a while)")
    db = FAISS.from_documents(splits, embeddings)
    print("In-memory FAISS vector database created successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Application startup...")
    setup_rag_pipeline()
    print("Application startup complete.")
    yield
    # This code runs on shutdown (won't be used in this simple app)
    print("Application shutdown...")


app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
# This is crucial for allowing your Vercel frontend to talk to your RPi backend
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "null", # Allows local file:// access for testing
    "https://law-ai-chatbot-git-main-howies-projects-cbd15e17.vercel.app" # Your Vercel app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Allow POST and OPTIONS (for preflight)
    allow_headers=["*"],
)
# ----------------------


class ChatRequest(BaseModel):
    message: str # Changed from 'prompt' to 'message' to match new HTML


# --- Tenacity Retry ---
# This will retry 4 times, waiting 2, 4, 8 seconds
# Only retries on 503 (Service Unavailable) errors
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPStatusError)
)
async def call_gemini_api(payload: dict):
    """
    Helper function to call the Gemini API with retry logic.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
    
    # Use httpx for async requests
    async with httpx.AsyncClient() as client:
        # Increase the timeout to 120 seconds for RPi
        response = await client.post(url, json=payload, timeout=120.0)
        
        # Raise an error if the status is 503, which triggers the retry
        if response.status_code == 503:
            print("Gemini API returned 503, retrying...")
            response.raise_for_status()
            
        # Also raise for other HTTP errors (like 4xx, 5xx)
        response.raise_for_status()
        
        return response.json()

async def call_general_search_api(user_query: str):
    """
    Calls Gemini with Google Search enabled (grounded generation).
    """
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question based on the provided Google Search results. "
        "If the search results are not relevant, just say you couldn't find information on that topic."
    )
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    try:
        api_response = await call_gemini_api(payload)
        text = api_response["candidates"][0]["content"]["parts"][0]["text"]
        
        # Add the web search disclaimer
        disclaimer = (
            "\n\n**Disclaimer:** *This information was sourced from the general web, "
            "not the specialized Malaysian law knowledge base. It is for general "
            "information only and is not legal advice.*"
        )
        return text + disclaimer
        
    except httpx.HTTPStatusError as http_err:
        print(f"HTTP error calling Gemini (Search): {http_err}")
        print(f"Response body: {http_err.response.text}")
        return "Sorry, there was an error communicating with the AI service (Search). Check the server logs."
    except Exception as e:
        print(f"An unexpected error occurred (Search): {e}")
        return "Sorry, an unexpected error occurred. Check the server logs."


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global db, embeddings
    
    user_query = request.message

    if not db:
        print("Vector DB not initialized. Falling back to general search.")
        response_text = await call_general_search_api(user_query)
        return JSONResponse(content={"response": response_text})

    try:
        # 1. Search the vector database
        docs_and_scores = db.similarity_search_with_relevance_scores(
            user_query, k=3, fetch_k=20
        )
        
        # 2. Check relevance
        if not docs_and_scores or docs_and_scores[0][1] > RAG_RELEVANCE_THRESHOLD:
            # Score is too high (bad match) or no docs found, fall back to web search
            print(f"No relevant context found (best score: {docs_and_scores[0][1]}). Falling back to web search.")
            response_text = await call_general_search_api(user_query)
            return JSONResponse(content={"response": response_text})

        # 3. We found good context! Proceed with RAG.
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, score in docs_and_scores]
        )
        
        # --- THIS IS THE CRITICAL CHANGE ---
        # We now tell the AI to reply with "NO_CONTEXT" if it can't find the answer.
        system_prompt = (
            "You are an AI assistant for Malaysian transportation law. "
            "You must answer the user's question **based ONLY on the provided legal context below**. "
            "Do not use any outside knowledge. If the answer is not in the context, "
            "you MUST reply with only the exact word `NO_CONTEXT` and nothing else."
        )
        
        prompt = (
            f"**Legal Context:**\n{context_text}\n\n"
            f"**User's Question:**\n{user_query}\n\n"
            f"**Answer (Based ONLY on the context):**"
        )

        # 4. Call Gemini API (RAG)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.3, # Make it more factual
                "topP": 0.8,
                "topK": 40
            }
        }
        
        api_response = await call_gemini_api(payload)
        response_text = api_response["candidates"][0]["content"]["parts"][0]["text"]
        
        # --- THIS IS THE NEW LOGIC ---
        # Check if the AI returned our "secret code".
        if "NO_CONTEXT" in response_text:
            print("RAG context was insufficient. Falling back to web search.")
            # If so, run the general web search instead.
            response_text = await call_general_search_api(user_query)
        
        return JSONResponse(content={"response": response_text})

    except httpx.HTTPStatusError as http_err:
        # This handles errors from the Gemini call, including 503
        print(f"HTTP error calling Gemini (RAG): {http_err}")
        print(f"Response body: {http_err.response.text}")
        return JSONResponse(
            content={"response": "Sorry, there was an error communicating with the AI service. Check the server logs."},
            status_code=500
        )
    except Exception as e:
        print(f"An unexpected error occurred in /chat: {e}")
        return JSONResponse(
            content={"response": "Sorry, an unexpected error occurred. Check the server logs."},
            status_code=500
        )


@app.get("/")
def read_root():
    return {"message": "Malaysian Legal AI Bot backend is running."}


if __name__ == "__main__":
    import uvicorn
    # Make sure this port matches your ngrok service and systemd file
    uvicorn.run(app, host="0.0.0.0", port=9999)