# --- Imports ---
import os
import json
import time
import traceback
import shutil
import stat
import sys
import hashlib
from pathlib import Path
from flask import Flask, request, render_template, jsonify, Response, send_from_directory
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Iterator
import logging
from datetime import datetime
from pypdf import errors as pypdf_errors

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initial Setup ---
load_dotenv()

# --- API Key Rotation Manager ---
class ApiKeyManager:
    """Manages a pool of API keys to rotate on rate limit errors."""
    def __init__(self):
        self.keys = self._load_keys()
        if not self.keys:
            raise ValueError("No GOOGLE_API_KEY or GEMINI_API_KEY* variables found in the .env file.")
        self.current_key_index = 0
        logging.info(f"🔑 Loaded {len(self.keys)} API keys for rotation.")

    def _load_keys(self) -> List[str]:
        """Loads all GOOGLE_API_KEY and GEMINI_API_KEY* variables from environment."""
        api_keys = []
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            api_keys.append(google_api_key)
        
        for key, value in os.environ.items():
            if key.startswith("GEMINI_API_KEY") and value and value not in api_keys:
                api_keys.append(value)
        return api_keys

    def get_current_key(self) -> str:
        """Returns the currently active API key."""
        return self.keys[self.current_key_index]

    def switch_to_next_key(self) -> str:
        """Rotates to the next key in the list and returns it."""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        new_key = self.get_current_key()
        logging.warning(f"🔄 Switching to API key index {self.current_key_index}.")
        genai.configure(api_key=new_key) # Re-configure the SDK with the new key
        return new_key

# --- Configuration ---
try:
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "models/embedding-001"
    
    API_KEY_MANAGER = ApiKeyManager()
    genai.configure(api_key=API_KEY_MANAGER.get_current_key())
    logging.info(f"✅ Google Generative AI SDK configured successfully for model: {MODEL_NAME}")

except Exception as e:
    logging.critical(f"❌ Configuration Error: {e}")
    sys.exit("Critical configuration error. Exiting.")

# --- Prompt and Summary Loading Functions ---
def load_text_file(file_path: Path, file_description: str) -> str:
    """Loads a text file, handling potential errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logging.info(f"✅ {file_description} loaded successfully from {file_path.name}.")
        return content
    except FileNotFoundError:
        logging.warning(f"⚠️ WARNING: The file '{file_path.name}' was not found. Continuing without it.")
        return ""
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to read the file '{file_path.name}': {e}")
        return ""

# --- Custom Embedding Function for ChromaDB ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function using the Gemini API with API key rotation."""
    def __init__(self, api_key_manager: ApiKeyManager, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        self._api_key_manager = api_key_manager
        logging.info(f"✅ GeminiEmbeddingFunction initialized with model: {self._model_name}")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if not input: return []
        valid_input = [doc for doc in input if doc and doc.strip()]
        if not valid_input: return []
        
        max_retries = len(self._api_key_manager.keys)
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=self._model_name, content=valid_input, task_type=self._task_type)
                return result['embedding']
            except google_exceptions.ResourceExhausted as e:
                logging.warning(f"⚠️ Embedding rate limit hit: {e}. Rotating key...")
                self._api_key_manager.switch_to_next_key()
                if attempt < max_retries - 1:
                    logging.info("Retrying embedding with new key...")
                else:
                    logging.error("❌ All API keys are rate-limited for embedding. Aborting.")
                    raise e
            except Exception as e:
                logging.error(f"⚠️ An unexpected embedding error occurred: {e}. Rotating key as a precaution.")
                self._api_key_manager.switch_to_next_key()
                if attempt >= max_retries - 1:
                     logging.error("❌ Embedding failed after exhausting all API keys.")
                     raise e
        
        logging.error("⚠️ Embedding failed after all retries. Returning zero vectors.")
        # Dimension for embedding-001 is 768
        return [[0.0] * 768 for _ in valid_input]


# --- Core Application Setup ---
app = Flask(__name__)

# --- Directory and Database Initialization ---
def setup_database_and_directories():
    base_dir = Path(__file__).parent
    storage_path = base_dir / "arc_chroma_db_storage"
    pdf_dir = base_dir / "documents"
    history_dir = base_dir / "history"
    receipts_dir = base_dir / "processing_receipts"
    
    pdf_dir.mkdir(exist_ok=True); history_dir.mkdir(exist_ok=True); storage_path.mkdir(parents=True, exist_ok=True); receipts_dir.mkdir(exist_ok=True)
    
    logging.info(f"📁 PDF Directory: {pdf_dir.resolve()}")
    logging.info(f"📁 ChromaDB Storage: {storage_path.resolve()}")
    client = chromadb.PersistentClient(path=str(storage_path))
    collection_name = "arc_career_guidance_pure_rag_v1"
    
    run_embeddings_file = base_dir / "RUN-EMBEDDINGS"
    if run_embeddings_file.exists():
        logging.warning("🚨 'RUN-EMBEDDINGS' file found. Forcing a full re-embedding of all documents.")
        try:
            client.delete_collection(name=collection_name)
            logging.info(f"  ✅ Deleted ChromaDB collection: '{collection_name}'")
        except ValueError:
            logging.info(f"  -> Collection '{collection_name}' did not exist, creating fresh.")
        except Exception as e:
            logging.error(f"❌ Could not delete collection '{collection_name}': {e}")
        
        if receipts_dir.exists():
            logging.info("  -> Deleting old processing receipts...")
            for receipt_file in receipts_dir.glob("*.receipt"):
                try:
                    receipt_file.unlink()
                except Exception as e:
                    logging.error(f"  ❌ Could not delete receipt file {receipt_file.name}: {e}")
            logging.info("  ✅ Finished deleting receipts.")

        run_embeddings_file.unlink()
        logging.info("  ✅ Deleted 'RUN-EMBEDDINGS' trigger file.")

    embedding_function = GeminiEmbeddingFunction(api_key_manager=API_KEY_MANAGER)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    return collection, pdf_dir, history_dir, receipts_dir

COLLECTION, PDF_DIRECTORY, HISTORY_DIRECTORY, RECEIPTS_DIRECTORY = setup_database_and_directories()
LOG_FILE_PATH = Path(__file__).parent / "conversation_log.txt"

# --- Load Prompts and Summaries ---
BASE_SYSTEM_PROMPT = load_text_file(Path(__file__).parent / "prompt.txt", "System prompt")
REF_SUMMARIES = load_text_file(Path(__file__).parent / "REFSummaries.txt", "REF Summaries")
REF_JOBS = load_text_file(Path(__file__).parent / "REFJobs.txt", "REF Jobs")

# Combine all context into a single, comprehensive system prompt
FULL_SYSTEM_PROMPT = f"""
{BASE_SYSTEM_PROMPT}

--- HIGH-LEVEL CONTEXT: SUMMARIES ---
{REF_SUMMARIES}
---

--- HIGH-LEVEL CONTEXT: JOB INFORMATION ---
{REF_JOBS}
---
"""

# --- Document Processing ---
def process_all_pdfs():
    logging.info("\n🔍 Scanning for PDF documents...")
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        logging.warning("📂 No PDF files found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    for pdf_file in pdf_files:
        try:
            # Check if the file is too small to be a valid PDF (e.g., under 1 KB)
            if pdf_file.stat().st_size < 1024:
                logging.warning(
                    f"📄 Skipping {pdf_file.name} because it is smaller than 1KB. "
                    f"This might be a Git LFS pointer instead of the actual PDF file. "
                    f"Please run 'git lfs pull' to download the file."
                )
                continue # Move to the next file

            receipt_file = RECEIPTS_DIRECTORY / f"{pdf_file.name}.receipt"
            if receipt_file.exists():
                receipt_mtime = receipt_file.stat().st_mtime
                pdf_mtime = pdf_file.stat().st_mtime
                if pdf_mtime <= receipt_mtime:
                    logging.info(f"📄 Skipping unchanged document: {pdf_file.name}")
                    continue

            logging.info(f"\n📄 Processing {pdf_file.name}...")
            COLLECTION.delete(where={"$and": [{"source": pdf_file.name}, {"chunk_type": "document"}]})
            logging.info(f"  -> Cleared old document entries for {pdf_file.name}.")

            pages = PyPDFLoader(str(pdf_file)).load()
            full_text = "\n".join([page.page_content for page in pages])
            chunks = text_splitter.split_text(full_text)
            
            if chunks:
                COLLECTION.add(
                    documents=chunks,
                    metadatas=[{"source": pdf_file.name, "chunk_index": i, "chunk_type": "document"} for i in range(len(chunks))],
                    ids=[f"{pdf_file.stem}_doc_{i}" for i in range(len(chunks))]
                )
                logging.info(f"  ✅ Added {len(chunks)} text chunks to ChromaDB.")

            receipt_file.touch()
            logging.info(f"  ✅ Created processing receipt for {pdf_file.name}")
        
        except pypdf_errors.PdfStreamError as pse:
            logging.error(
                f"❌ Error processing {pdf_file.name}: {pse}. "
                "The file may be corrupted or a Git LFS pointer. Please check the file."
            )
        except Exception as e:
            logging.error(f"❌ An unexpected error occurred while processing {pdf_file.name}: {e}", exc_info=True)
    logging.info(f"\n🎉 PDF processing complete. Total items in collection: {COLLECTION.count()}")

# --- Chat History and Logging ---
def save_chat_history(session_id: str, history: List[Dict]):
    with open(HISTORY_DIRECTORY / f"{session_id}.json", 'w', encoding='utf-8') as f: json.dump(history, f, indent=2)

def load_chat_history(session_id: str) -> List[Dict]:
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return []
    return []

def log_conversation(session_id: str, question: str, answer: str):
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(f"--- Log Entry: {datetime.now():%Y-%m-%d %H:%M:%S} ---\nSession: {session_id}\nQ: {question}\nA: {answer}\n---\n\n")

# --- Pure RAG Generation with Two-Step Query ---
def generate_deeper_question(message: str, history: List[Dict]) -> str:
    history_str = "\n".join([f"{t['role']}: {t['parts'][0]}" for t in history[-4:]])
    prompt = f"""
    Based on the latest user question and the recent conversation history, formulate a single, comprehensive question to search a vector database.
    **Conversation History:**
    {history_str}
    **Latest User Question:**
    "{message}"
    **Generated Search Query:**
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        deeper_question = response.text.strip()
        logging.info(f"🧠 Generated Deeper Question: {deeper_question}")
        return deeper_question
    except Exception as e:
        logging.error(f"❌ Failed to generate deeper question: {e}. Falling back to original message.")
        return message

def pure_rag_generation(message: str, history: List[Dict]) -> Iterator[str]:
    deeper_question = generate_deeper_question(message, history)
    
    try:
        doc_results = COLLECTION.query(query_texts=[deeper_question], n_results=15, where={"chunk_type": "document"})
        retrieved_docs = list(zip(doc_results.get('documents', [[]])[0], doc_results.get('metadatas', [[]])[0]))
    except Exception as e:
        logging.error(f"❌ ChromaDB query failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'payload': 'Failed to retrieve information from the knowledge base.'})}\n\n"
        return

    if not retrieved_docs:
        logging.warning("⚠️ No relevant documents found for the query.")
        no_info_message = "I'm sorry, but I couldn't find any specific information related to your question in the available documents."
        yield f"data: {json.dumps({'type': 'text', 'payload': no_info_message})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'payload': {'full_text': no_info_message}})}\n\n"
        return

    context = "DOCUMENT EXCERPTS FOR CONTEXT:\n"
    citations = []
    for i, (doc, meta) in enumerate(retrieved_docs):
        context += f"[{i+1}] Source: {meta.get('source', 'N/A')}\nContent: {doc}\n\n"
        citations.append({"id": str(i+1), "source": meta.get('source'), "content": doc})
    
    yield f"data: {json.dumps({'type': 'citations', 'payload': citations})}\n\n"

    max_retries = len(API_KEY_MANAGER.keys)
    for attempt in range(max_retries):
        try:
            model_history = [{'role': t['role'], 'parts': t['parts']} for t in history]
            model = genai.GenerativeModel(MODEL_NAME)
            chat = model.start_chat(history=model_history)
            
            final_prompt = (
                f"{FULL_SYSTEM_PROMPT}\n\n"
                f"---BEGIN CONTEXTUAL INFORMATION---\n{context}\n---END CONTEXTUAL INFORMATION---\n\n"
                f"Please answer the following user question based on the rules and context provided.\n\n"
                f"Question: {message}"
            )
            
            response_stream = chat.send_message(final_prompt, stream=True)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'type': 'text', 'payload': chunk.text})}\n\n"
            
            yield f"data: {json.dumps({'type': 'end', 'payload': {'full_text': full_response}})}\n\n"
            return

        except google_exceptions.ResourceExhausted:
            logging.warning(f"⚠️ Chat response rate limit on key index {API_KEY_MANAGER.current_key_index}. Rotating key...")
            API_KEY_MANAGER.switch_to_next_key()
            if attempt < max_retries - 1:
                logging.info("Retrying chat generation with new key...")
            else:
                logging.error("❌ All API keys are rate-limited. Aborting.")
                yield f"data: {json.dumps({'type': 'error', 'payload': 'All available API keys are currently rate-limited.'})}\n\n"
                return
        except Exception as e:
            logging.error(f"❌ An unexpected error occurred during chat generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'payload': 'An unexpected error occurred.'})}\n\n"
            return

# --- Flask API Endpoints ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message, session_id = data.get('message', ''), data.get('session_id')
    if not message or not session_id: return jsonify({'error': 'Invalid request'}), 400
    
    history = load_chat_history(session_id)
    
    def stream_and_save():
        full_response = ""
        citations = []
        
        for event_str in pure_rag_generation(message, history):
            yield event_str
            if event_str.strip().startswith('data:'):
                try:
                    event_data = json.loads(event_str[len('data:'):].strip())
                    if event_data.get("type") == "citations":
                        citations = event_data.get("payload", [])
                    elif event_data.get("type") == "end":
                        full_response = event_data.get("payload", {}).get("full_text", "")
                except (json.JSONDecodeError, IndexError):
                    pass

        if full_response:
            history.append({"role": "user", "parts": [message]})
            history.append({"role": "model", "parts": [full_response], "citations": citations})
            save_chat_history(session_id, history)
            log_conversation(session_id, message, full_response)

    return Response(stream_and_save(), mimetype='text/event-stream')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = []
    for f in sorted(HISTORY_DIRECTORY.glob("*.json"), key=os.path.getmtime, reverse=True):
        history = load_chat_history(f.stem)
        preview = next((t['parts'][0] for t in history if t['role'] == 'user'), "New Chat")
        sessions.append({"id": f.stem, "preview": preview[:50]})
    return jsonify({"sessions": sessions})

@app.route('/api/history', methods=['GET'])
def get_history():
    session_id = request.args.get('session_id')
    history = load_chat_history(session_id)
    if history:
        return jsonify({"success": True, "history": history})
    else:
        return jsonify({"success": False, "history": [], "message": "History not found or is empty."})

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    (HISTORY_DIRECTORY / f"{session_id}.json").unlink(missing_ok=True)
    return jsonify({"success": True})

# --- Main Execution ---

# Call the PDF processing function when the application starts.
# This ensures it runs in both local development (python app.py)
# and production environments (like Gunicorn on Render).
process_all_pdfs()

# The following block only runs when the script is executed directly
# (i.e., `python app.py`), not when imported by a server like Gunicorn.
if __name__ == '__main__':
    # This route is useful for serving PDFs for citations during local testing.
    @app.route('/documents/<path:filename>')
    def serve_document(filename):
        return send_from_directory(PDF_DIRECTORY, filename, as_attachment=False)
    
    # This starts the Flask development server.
    app.run(host='0.0.0.0', port=5003, debug=True)