# --- Imports ---
import os
import json
import time
import traceback
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory # Added send_from_directory
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging # Added logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initial Setup ---
load_dotenv()

# --- Configuration ---
try:
    MODEL_NAME = "gemini-2.0-flash" # Using a powerful and recent model for all tasks
    EMBEDDING_MODEL_NAME = "models/embedding-001"
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    logging.info("✅ Google Generative AI SDK configured successfully.")
except Exception as e:
    logging.critical(f"❌ Configuration Error: {e}")
    # Exit if API key is not set, as the application cannot function without it
    import sys
    sys.exit("Critical configuration error. Exiting.")

# --- Custom Embedding Function for ChromaDB ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function using the Gemini API."""
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        logging.info(f"✅ GeminiEmbeddingFunction initialized with model: {self._model_name}")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if not input:
            return []
        try:
            # Filter out empty strings which can cause embedding errors
            valid_input = [doc for doc in input if doc.strip()]
            if not valid_input:
                return []
                
            result = genai.embed_content(
                model=self._model_name,
                content=valid_input,
                task_type=self._task_type
            )
            return result['embedding']
        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"⚠️ Embedding rate limit hit: {e}. Pausing for 5s.")
            time.sleep(5) # Small pause and retry
            try:
                result = genai.embed_content(
                    model=self._model_name,
                    content=valid_input,
                    task_type=self._task_type
                )
                return result['embedding']
            except Exception as retry_e:
                logging.error(f"⚠️ Embedding retry failed: {retry_e}. Returning zero vectors.")
                return [[0.0] * 768 for _ in input]
        except Exception as e:
            logging.error(f"⚠️ Embedding error: {e}. Returning zero vectors for input of size {len(input)}.")
            return [[0.0] * 768 for _ in input]

# --- Knowledge Graph Manager (Enhanced for Career Guidance) ---
class ChromaDBKnowledgeGraph:
    """Manages detailed entity and relationship extraction for career guidance."""
    def __init__(self, collection):
        self.collection = collection
        self.enabled = True # Can be toggled if KG is optional
        logging.info("✅ ChromaDB Knowledge Graph manager initialized.")

    def extract_entities_with_gemini(self, text: str) -> Dict[str, Any]:
        """Uses Gemini to extract detailed, career-focused entities and relationships."""
        # Only process significant text lengths
        if not text.strip() or len(text) < 500: # Increased minimum text length for KG extraction
            return {"entities": [], "relationships": []}
        
        # Crafting a precise prompt for JSON output
        prompt = f"""From the following text about career guidance and economic trends, extract key entities and their relationships based on the provided schema.

Recognized Entity Types:
- **Career_Pathway**: Specific job roles or career tracks (e.g., "Data Scientist", "AI Ethics Officer", "Green Energy Consultant").
- **Skill**: Specific abilities, competencies, or knowledge areas (e.g., "Python Programming", "Critical Thinking", "Project Management", "Digital Marketing").
- **Industry**: A broad sector of the economy (e.g., "Financial Technology", "Healthcare", "Renewable Energy", "E-commerce").
- **Economic_Trend**: High-level economic, technological, or societal shifts (e.g., "AI Automation", "Gig Economy", "Green Transition", "Remote Work Adoption").
- **Organization**: Companies, institutions, government bodies, or educational establishments mentioned.

Recognized Relationship Types (always from Source to Target):
- **REQUIRES_SKILL**: A Career_Pathway needs a specific Skill. (e.g., "Data Scientist" REQUIRES_SKILL "Python Programming").
- **PART_OF_INDUSTRY**: A Career_Pathway or Organization belongs to an Industry. (e.g., "AI Ethics Officer" PART_OF_INDUSTRY "Technology Governance").
- **LEADS_TO**: A Skill or Economic_Trend can lead to a Career_Pathway or Industry. (e.g., "AI Automation" LEADS_TO "Robotics Engineer").
- **INFLUENCES**: An Economic_Trend influences an Industry or Career_Pathway. (e.g., "AI Automation" INFLUENCES "Financial Technology").
- **OFFERS**: An Organization offers a Career_Pathway or Skill. (e.g., "Google" OFFERS "Software Engineer").

Return the output as a single, valid JSON object with two top-level keys: "entities" and "relationships".
Each entity must have "name" (string) and "type" (string from recognized types).
Each relationship must have "source" (string, name of source entity), "target" (string, name of target entity), and "type" (string from recognized types).
Ensure all extracted entity names match their exact appearance in the text.
Do not include any other text or markdown formatting outside the JSON.

Text for Analysis (first 8000 characters):
{text[:8000]}
"""
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            # Use a low temperature for factual extraction
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.0))
            
            response_text = response.text.strip()
            # Clean up potential markdown code blocks that Gemini might include
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[len("```json"): -len("```")].strip()
            elif response_text.startswith("```") and response_text.endswith("```"):
                 response_text = response_text[len("```"): -len("```")].strip()

            return json.loads(response_text)
        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"⚠️ Knowledge Graph extraction rate limit. Pausing for 60s.")
            time.sleep(60)
            return {"entities": [], "relationships": []} # Return empty on rate limit
        except (json.JSONDecodeError, Exception) as e:
            response_text_for_error = response.text if hasattr(response, 'text') else 'No response text available'
            logging.error(f"⚠️ Knowledge Graph extraction failed: {e}\nResponse Text (first 500 chars): {response_text_for_error[:500]}\nTraceback: {traceback.format_exc()}")
            return {"entities": [], "relationships": []}

    def store_knowledge_graph(self, extraction_data: Dict[str, Any], source_doc: str):
        """Stores extracted entities and relationships in ChromaDB."""
        if not extraction_data or not extraction_data.get("entities"):
            logging.info(f"  -> No knowledge graph data to store for {source_doc}.")
            return

        documents, metadatas, ids = [], [], []
        
        for entity in extraction_data.get("entities", []):
            # Ensure required keys exist and are not empty
            if not all(k in entity and entity[k] for k in ["name", "type"]): continue
            
            doc_text = f"Entity: {entity['name']} (Type: {entity['type']})"
            if entity.get('description'): # Add description if available
                doc_text += f" - Description: {entity['description']}"
            
            documents.append(doc_text)
            metadatas.append({
                "chunk_type": "entity", 
                "source": source_doc, 
                "entity_name": entity['name'],
                "entity_type": entity['type']
            })
            ids.append(f"entity_{source_doc}_{entity['name'].replace(' ', '_').replace('.', '')}_{entity['type']}") # Sanitize ID

        for rel in extraction_data.get("relationships", []):
            if not all(k in rel and rel[k] for k in ["source", "target", "type"]): continue
            
            doc_text = f"Relationship: '{rel['source']}' --({rel['type']})--> '{rel['target']}'"
            documents.append(doc_text)
            metadatas.append({
                "chunk_type": "relationship", 
                "source": source_doc, 
                "source_entity": rel['source'],
                "target_entity": rel['target'], 
                "relationship_type": rel['type']
            })
            ids.append(f"rel_{source_doc}_{rel['source'].replace(' ', '_')}_{rel['target'].replace(' ', '_')}_{rel['type']}".replace('.', '')) # Sanitize ID

        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logging.info(f"  ✅ Stored {len(extraction_data.get('entities',[]))} entities and {len(extraction_data.get('relationships',[]))} relationships for {source_doc}")
            except Exception as e:
                logging.error(f"❌ Error adding KG data to ChromaDB for {source_doc}: {e}")

# --- Core Application Setup ---
app = Flask(__name__)

# --- Directory and Database Initialization ---
def setup_database_and_directories():
    """Initializes ChromaDB and creates necessary directories."""
    base_dir = Path(__file__).parent
    storage_path = base_dir / "arc_chroma_db_storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    
    pdf_dir = base_dir / "documents"
    pdf_dir.mkdir(exist_ok=True)
    
    history_dir = base_dir / "history"
    history_dir.mkdir(exist_ok=True)
    
    logging.info(f"📁 PDF Directory: {pdf_dir.resolve()}")
    logging.info(f"📁 History Directory: {history_dir.resolve()}")
    logging.info(f"📁 ChromaDB Storage: {storage_path.resolve()}")

    client = chromadb.PersistentClient(path=str(storage_path))
    embedding_function = GeminiEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="arc_career_guidance", 
        embedding_function=embedding_function
    )
    
    return collection, pdf_dir, history_dir

# Initialize global variables at startup
COLLECTION, PDF_DIRECTORY, HISTORY_DIRECTORY = setup_database_and_directories()
KG_MANAGER = ChromaDBKnowledgeGraph(COLLECTION)

# --- Document Processing ---
def process_all_pdfs():
    """Scans for and processes all PDF files using a text splitter for larger chunks."""
    logging.info("\n🔍 Scanning for PDF documents...")
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning("📂 No PDF files found. Add PDFs to the 'documents' folder and restart.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len,
    )

    logging.info(f"📁 Found {len(pdf_files)} PDF file(s) to process.")
    for pdf_file in pdf_files:
        try:
            # Check if document already processed by looking for chunks from this source
            existing_chunks = COLLECTION.get(where={"source": pdf_file.name}, limit=1)['ids']
            if existing_chunks:
                logging.info(f"📄 Skipping already processed document: {pdf_file.name}")
                continue

            logging.info(f"\n📄 Processing {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            full_document_text = "\n".join([page.page_content for page in pages])
            chunks = text_splitter.split_text(full_document_text)

            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk)
                    metadatas.append({"source": pdf_file.name, "chunk_index": i + 1, "chunk_type": "document"})
                    ids.append(f"{pdf_file.stem}_chunk_{i}")
            
            if documents:
                COLLECTION.add(documents=documents, metadatas=metadatas, ids=ids)
                logging.info(f"  ✅ Added {len(documents)} text chunks to ChromaDB.")
                
                # Extract and store knowledge graph data
                kg_data = KG_MANAGER.extract_entities_with_gemini(full_document_text)
                KG_MANAGER.store_knowledge_graph(kg_data, pdf_file.name)

        except Exception as e:
            logging.error(f"❌ Error processing {pdf_file.name}: {e}", exc_info=True)
    
    logging.info(f"\n🎉 PDF processing complete. Total items in collection: {COLLECTION.count()}")

# --- Chat History Functions ---
def save_chat_history(session_id: str, history: List[Dict]):
    """Saves chat history for a given session ID."""
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2) # Use indent for readability
    except IOError as e:
        logging.error(f"Error saving chat history for session {session_id}: {e}")

def load_chat_history(session_id: str) -> List[Dict]:
    """Loads chat history for a given session ID."""
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON for session {session_id}: {e}. Returning empty history.")
            return [] # Return empty history if file is corrupted
        except Exception as e:
            logging.error(f"Error loading chat history for session {session_id}: {e}. Returning empty history.")
            return []
    return []

# --- Retrieval and Generation (Now with History and Enhanced Citation Formatting!) ---
def enhanced_retrieval_and_generation(message: str, history: List[Dict]):
    """
    Performs RAG using conversation history for context-aware retrieval and generation.
    Formats citations to match frontend expectations.
    """
    # Create a contextual query from history for better retrieval
    # Take last 6 turns (including current user message) for context
    history_for_query_context = [
        f"{turn['role']}: {turn['parts'][0]}" 
        for turn in history[-6:] if turn.get('parts') and turn['parts']
    ]
    contextual_query = "\n".join(history_for_query_context) + f"\nuser: {message}"

    logging.info(f"🔎 Contextual Query for Retrieval:\n---\n{contextual_query}\n---")

    documents_results = []
    # Try fetching documents and entities/relationships
    try:
        document_query_results = COLLECTION.query(
            query_texts=[contextual_query],
            n_results=6, # Get up to 6 document chunks
            where={"chunk_type": "document"}
        )
        documents_results.extend(zip(
            document_query_results.get('documents', [[]])[0],
            document_query_results.get('metadatas', [[]])[0]
        ))
    except Exception as e:
        logging.warning(f"Failed to retrieve document chunks: {e}")

    try:
        # Also query for relevant KG entities/relationships
        kg_query_results = COLLECTION.query(
            query_texts=[contextual_query],
            n_results=2, # Get a couple of KG entries
            where={"chunk_type": {"$in": ["entity", "relationship"]}}
        )
        documents_results.extend(zip(
            kg_query_results.get('documents', [[]])[0],
            kg_query_results.get('metadatas', [[]])[0]
        ))
    except Exception as e:
        logging.warning(f"Failed to retrieve knowledge graph entries: {e}")

    # Remove duplicates if any (e.g., same chunk returned by different queries or types)
    unique_documents = {}
    for doc_text, meta_data in documents_results:
        # Use a combination of source, chunk_index, and chunk_type as a unique key
        key = (meta_data.get('source'), meta_data.get('chunk_index'), meta_data.get('chunk_type'))
        unique_documents[key] = (doc_text, meta_data)
    
    sorted_unique_documents = list(unique_documents.values()) # Convert back to list

    retrieved_context_str = "DOCUMENT EXCERPTS:\n"
    citations_for_frontend = []
    if not sorted_unique_documents:
        retrieved_context_str += "No relevant documents or knowledge graph entries were found for this query.\n"
    else:
        for i, (doc, meta) in enumerate(sorted_unique_documents):
            source = meta.get('source', 'Unknown Document')
            chunk_type = meta.get('chunk_type', 'chunk')
            
            # For page number, try to extract from chunk_index or add specific metadata
            page_info = f"Chunk {meta.get('chunk_index', 'N/A')}"
            
            retrieved_context_str += f"Source [{i+1}]: {source} ({chunk_type}, {page_info})\n{doc}\n\n"
            
            # Format citations specifically for the frontend
            citations_for_frontend.append({
                "id": i + 1, # Frontend expects a sequential ID for linking
                "source": source,
                "page_number": page_info, # Use the chunk info as 'page_number'
                "content": doc # Send the full chunk content
            })
    
    # Format chat history for the generation prompt
    formatted_history = ""
    # Only include 'user' and 'model' roles, and ensure 'parts' is present and not empty
    for turn in history:
        if isinstance(turn.get('parts'), list) and turn['parts']:
            role = turn['role'].capitalize()
            # If citations are present, they are already part of the model's text in the history
            formatted_history += f"**{role}**: {turn['parts'][0]}\n\n"

    system_prompt = (
        "You are the ARC Principal Career Strategist, an expert on the Refracted Economies Framework. "
        "Your role is to provide detailed, comprehensive, and strategic career guidance based on the user's questions, "
        "the conversation history, and the provided document excerpts. "
        "Synthesize all information to answer the user's latest question. Explain the 'why' and 'how'. "
        "Structure your response with clear headings, bullet points, and bolded keywords. "
        "Crucially, cite your sources precisely using numerical superscripts like [^1], [^2], etc., "
        "corresponding to the 'Source [N]' in the 'DOCUMENT EXCERPTS' section. "
        "For example, if you use information from 'Source [1]: my_document.pdf', cite it as [^1]."
        "Prioritize information from the provided document excerpts when relevant. If the answer is not in the documents, state that you don't know but offer general guidance."
    )
    
    final_prompt = (
        f"{system_prompt}\n\n"
        f"CONVERSATION HISTORY:\n---\n{formatted_history.strip()}\n---\n\n"
        f"{retrieved_context_str}\n\n"
        f"Based on the conversation history and the provided context, answer the following question:\n"
        f"**User**: {message}\n\n"
        f"**Principal Strategist's Response**:"
    )
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Pass the formatted prompt to the model directly
        response = model.generate_content(final_prompt, generation_config=genai.GenerationConfig(temperature=0.2))
        
        # Access response text safely
        model_response_content = ""
        if response.candidates and response.candidates[0].content:
            model_response_content = response.candidates[0].content.parts[0].text
        else:
            logging.warning("Model response content was empty.")
            model_response_content = "Sorry, I couldn't generate a detailed response."

        return model_response_content, citations_for_frontend
    
    except google_exceptions.ResourceExhausted as e:
        logging.error(f"❌ LLM generation error due to rate limit: {e}")
        return "The service is currently busy due to high demand. Please wait a minute and try your question again.", []
    except Exception as e:
        logging.error(f"❌ LLM generation error: {e}", exc_info=True)
        return "Sorry, I encountered an error generating a response.", []

# --- Flask API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles chat requests, interacts with the Generative Model, and manages history."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')

        if not user_message or not session_id:
            return jsonify({'error': True, 'response': 'Message and session_id are required'}), 400

        logging.info(f"Received message for session {session_id}: {user_message}")

        # Load existing chat history
        chat_history = load_chat_history(session_id)

        # Get response and citations from the RAG pipeline
        response_text, citations_data = enhanced_retrieval_and_generation(user_message, chat_history)
        
        # Append user's message to history *before* the model's response for correct turn order
        chat_history.append({"role": "user", "parts": [user_message]})
        
        # Append model's response and structured citations to history for saving
        chat_history.append({
            "role": "model",
            "parts": [response_text], # Store the full text
            "citations": citations_data # Store structured citations
        })
        save_chat_history(session_id, chat_history)
        
        logging.info(f"Model responded for session {session_id}. Citations sent: {len(citations_data)}")
        return jsonify({
            'response': response_text,
            'citations': citations_data, # Ensure citations are part of the API response
            'success': True
        })

    except Exception as e:
        logging.error(f"General error in /api/chat: {e}", exc_info=True)
        return jsonify({'error': True, 'response': f'An unexpected error occurred: {e}'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Returns a list of all available chat sessions with previews."""
    sessions = []
    try:
        # Sort files by modification time, newest first
        files = sorted(HISTORY_DIRECTORY.glob("*.json"), key=os.path.getmtime, reverse=True)
    except Exception as e:
        logging.error(f"Error listing history files: {e}")
        return jsonify({'success': False, 'sessions': [], 'message': 'Could not read history directory'})

    for f in files:
        session_id = f.stem
        try:
            with open(f, 'r') as hist_file:
                history_data = json.load(hist_file)

            # Find the first user message for preview
            first_user_message = "New Chat Session" # Default preview if no user messages
            for turn in history_data:
                # Ensure 'parts' exists and has content before accessing
                if turn.get('role') == 'user' and isinstance(turn.get('parts'), list) and turn['parts']:
                    first_user_message = turn['parts'][0]
                    break # Found the first user message, exit loop

            preview = first_user_message[:50] + ('...' if len(first_user_message) > 50 else '')
            sessions.append({'id': session_id, 'preview': preview})
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logging.warning(f"Could not process session file {f.name}: {e}. Skipping this file.")
            continue # Skip corrupted or malformed files
    
    logging.info(f"Found {len(sessions)} chat sessions.")
    return jsonify({'success': True, 'sessions': sessions})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the chat history for a specific session."""
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': True, 'message': 'Session ID required'}), 400
    
    # Load history including the 'citations' field
    history = load_chat_history(session_id)
    
    # The history loaded already contains 'parts' and 'citations' as needed by the frontend
    return jsonify({'success': True, 'history': history})

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Deletes a specific chat session's history file."""
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    if history_file.exists():
        try:
            os.remove(history_file)
            logging.info(f"🗑️ Deleted session file: {session_id}.json")
            return jsonify({'success': True, 'message': f'Session {session_id} deleted successfully.'})
        except Exception as e:
            logging.error(f"❌ Error deleting session file {session_id}.json: {e}", exc_info=True)
            return jsonify({'success': False, 'message': f'Failed to delete session file: {str(e)}'}), 500
    logging.warning(f"Attempted to delete non-existent session: {session_id}.json")
    return jsonify({'success': False, 'message': 'Session not found.'}), 404

@app.route('/api/pdf/<path:filename>') # Use path: to allow slashes in filename if needed
def serve_pdf(filename):
    """Serves PDF files from the documents directory."""
    logging.info(f"Serving PDF: {filename}")
    try:
        # Prevent directory traversal with send_from_directory security
        return send_from_directory(PDF_DIRECTORY, filename, as_attachment=False)
    except FileNotFoundError:
        logging.warning(f"PDF file not found: {filename}")
        return "File not found.", 404
    except Exception as e:
        logging.error(f"Error serving PDF file {filename}: {e}", exc_info=True)
        return f"Error serving file: {e}", 500

# --- Main Execution ---
if __name__ == '__main__':
    # Process all PDFs on startup to populate the ChromaDB
    process_all_pdfs()
    
    # Get port from environment variable or default to 5004
    port = int(os.environ.get('PORT', 5004))
    logging.info(f"\n🚀 Starting Flask server on http://127.0.0.1:{port}")
    
    # Set debug=False for production to avoid running initial setup twice and for security.
    # Use host='0.0.0.0' to make the server accessible externally (e.g., in a Docker container).
    app.run(host='0.0.0.0', port=port, debug=False)