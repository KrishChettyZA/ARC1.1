# --- Imports ---
import os
import json
import time
import traceback
import shutil
import stat
import csv
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
import re
from datetime import datetime

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
        logging.warning(f"🔄 Switching to API key index {self.current_key_index}.")
        return self.get_current_key()

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

# --- Prompt Loading Function ---
def load_system_prompt(file_path: Path) -> str:
    """Loads the system prompt from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        logging.info(f"✅ System prompt loaded successfully from {file_path.name}.")
        return prompt_text
    except FileNotFoundError:
        logging.critical(f"❌ CRITICAL ERROR: The prompt file '{file_path.name}' was not found.")
        sys.exit(f"Exiting: Missing required file '{file_path.name}'.")
    except Exception as e:
        logging.critical(f"❌ CRITICAL ERROR: Failed to read the prompt file: {e}")
        sys.exit(f"Exiting: Error reading prompt file.")

# --- Custom Embedding Function for ChromaDB ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function using the Gemini API."""
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        logging.info(f"✅ GeminiEmbeddingFunction initialized with model: {self._model_name}")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if not input: return []
        valid_input = [doc for doc in input if doc and doc.strip()]
        if not valid_input: return []
        try:
            result = genai.embed_content(model=self._model_name, content=valid_input, task_type=self._task_type)
            return result['embedding']
        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"⚠️ Embedding rate limit hit: {e}. Pausing for 5s.")
            time.sleep(5)
            result = genai.embed_content(model=self._model_name, content=valid_input, task_type=self._task_type)
            return result['embedding']
        except Exception as e:
            logging.error(f"⚠️ Embedding error: {e}. Returning zero vectors.")
            return [[0.0] * 768 for _ in valid_input]

# --- Knowledge Graph Manager with API Key Rotation ---
class ChromaDBKnowledgeGraph:
    """Manages entity extraction with API key rotation for rate limits."""
    def __init__(self, collection):
        self.collection = collection
        self.enabled = True
        logging.info("✅ ChromaDB Knowledge Graph manager initialized.")

    def extract_entities_with_gemini(self, text: str) -> Dict[str, Any]:
        """Uses Gemini to extract entities, rotating API keys on rate limit errors."""
        if not text.strip():
            return {"entities": [], "relationships": []}

        prompt = f"""You are an expert in economic and career pathway analysis. Your task is to extract a knowledge graph from the text below using the Refracted Economies Framework (REF).

**Step 1: Define the Framework.**
First, you must understand the core concepts of the REF provided here. These definitions are your ground truth.

* **REF Color-Coded Economies (as 'Economic_Domain' entities):**
    * `Orange Economy`: Creative, cultural, and leisure activities (arts, media, sports, tourism).
    * `Green Economy`: Environmental sustainability (renewable energy, conservation).
    * `Blue Economy`: Water-resource-based activities (fishing, maritime transport).
    * `Lavender Economy`: Care and helping professions (healthcare, social work).
    * `Yellow Economy`: Public and social sector (government, education, NGOs).
    * `Bronze Economy`: Extraction and cultivation (mining, agriculture).
    * `Iron Economy`: Manufacturing, distribution, infrastructure (construction, logistics).
    * `Gold Economy`: Financial services (banking, fintech).
    * `Platinum Economy`: Technology and innovation (IT, AI, software).

* **REF Characteristics (as 'Characteristic' entities):**
    * `Skilled`/`Unskilled`: Requires advanced qualifications vs. minimal training.
    * `Knowledge`/`Physical`: Information-driven vs. manual labor.
    * `Elastic`/`Inelastic`: Resilient to technological disruption vs. routine and automatable.
    * `Entrepreneurial`/`Imitative`: Involving novel ventures vs. routine replication.
    * `Formal`/`Informal`: Regulated employment vs. unpaid or under-the-table work.
    * `Permanent`/`Gig`: Long-term contracts vs. short-term, project-based work.

**Step 2: Define the Schema for Extraction.**
Use the following entity and relationship types for your output.

* **Entity Types:** `Career_Pathway`, `Skill`, `Industry`, `Economic_Trend`, `Organization`, `Policy_Or_Strategy`, `Government_Body`, `Educational_Qualification`, `Location`, `Characteristic` (from REF list), `Economic_Domain` (from REF list).
* **Relationship Types:** `REQUIRES_SKILL`, `PART_OF_INDUSTRY`, `LEADS_TO`, `INFLUENCED_BY`, `HAS_PREREQUISITE`, `GOVERNED_BY`, `IMPLEMENTS`, `HAS_CHARACTERISTIC`, `BELONGS_TO_DOMAIN`, `PROGRESSES_TO`, `TEACHES_SKILL`, `LOCATED_IN`.

**Step 3: Perform the Extraction.**
Now, analyze the text below. Follow these rules precisely:
1.  **Identify Core Entities:** First, identify all entities like `Career_Pathway`, `Organization`, `Industry`, and `Location`.
2.  **Apply REF Lens:** For every `Career_Pathway`, determine its `Economic_Domain` and `Characteristic` entities based *only* on the definitions provided.
3.  **CRITICAL: Map Locations:** Whenever a career, organization, or industry is mentioned in the context of a specific place (city, province, country), you MUST create a `LOCATED_IN` relationship. For example, if the text says "Marine biology jobs are common in the Western Cape", you must create the relationship: `("Marine Biologist", "LOCATED_IN", "Western Cape")`.
4.  **Link Everything:** Connect all other entities using the defined relationship types.
5.  **Output Format:** Return a single, valid JSON object with "entities" and "relationships" keys. Do not add any commentary.

**Text for Analysis:**
{text}
"""
        max_retries = len(API_KEY_MANAGER.keys)
        for attempt in range(max_retries):
            response = None
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(prompt, request_options={"timeout": 180})
                
                cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(cleaned_text)
                
                if "entities" not in parsed_json or "relationships" not in parsed_json:
                    raise ValueError("Parsed JSON is missing required keys.")

                logging.info(f"  -> Extracted {len(parsed_json.get('entities',[]))} entities and {len(parsed_json.get('relationships',[]))} relationships.")
                return parsed_json

            except google_exceptions.ResourceExhausted:
                logging.warning(f"⚠️ Knowledge Graph rate limit on key index {API_KEY_MANAGER.current_key_index}. Rotating key...")
                new_key = API_KEY_MANAGER.switch_to_next_key()
                genai.configure(api_key=new_key)
                
                if attempt < max_retries - 1:
                    logging.info("Retrying with new key...")
                else:
                    logging.error("❌ All API keys are rate-limited. Pausing for 60s before restarting cycle.")
                    time.sleep(60)
            except (json.JSONDecodeError, ValueError, Exception) as e:
                response_text = response.text if response else "No response"
                logging.error(f"⚠️ Knowledge Graph extraction failed: {e}\nResponse: {response_text[:200]}")
                return {"entities": [], "relationships": []}

        return {"entities": [], "relationships": []}

    def store_knowledge_graph(self, extraction_data: Dict[str, Any], source_doc: str):
        if not extraction_data or (not extraction_data.get("entities") and not extraction_data.get("relationships")):
            return

        documents, metadatas, ids = [], [], []
        
        for entity in extraction_data.get("entities", []):
            if isinstance(entity, dict) and all(k in entity for k in ["name", "type"]):
                doc_text = f"Entity: {entity['name']} ({entity['type']})"
                unique_id_str = f"{source_doc}_{entity['name']}_{entity['type']}"
                hashed_id = hashlib.sha256(unique_id_str.encode()).hexdigest()
                documents.append(doc_text)
                metadatas.append({"chunk_type": "entity", "source": source_doc, "entity_name": entity['name'], "entity_type": entity['type']})
                ids.append(f"entity_{hashed_id}")

        for rel in extraction_data.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["source", "target", "type"]):
                doc_text = f"Relationship: {rel['source']} -> {rel['type']} -> {rel['target']}"
                unique_id_str = f"{source_doc}_{rel['source']}_{rel['target']}_{rel['type']}"
                hashed_id = hashlib.sha256(unique_id_str.encode()).hexdigest()
                documents.append(doc_text)
                metadatas.append({"chunk_type": "relationship", "source": source_doc, "source_entity": rel['source'], "target_entity": rel['target'], "relationship_type": rel['type']})
                ids.append(f"rel_{hashed_id}")

        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logging.info(f"  ✅ Stored consolidated graph for {source_doc}")
            except Exception as e:
                logging.error(f"❌ Error adding consolidated KG to ChromaDB for {source_doc}: {e}")

# --- Post-Extraction Consolidation ---
def consolidate_knowledge_graph_for_document(all_chunk_extractions: List[Dict]) -> Dict[str, Any]:
    """Consolidates and deduplicates entities and relationships from multiple chunks."""
    consolidated_entities, all_relationships = {}, []
    for data in all_chunk_extractions:
        for entity in data.get("entities", []):
            if isinstance(entity, dict) and "name" in entity and "type" in entity:
                consolidated_entities[entity["name"].lower()] = {"name": entity["name"], "type": entity["type"]}
        for rel in data.get("relationships", []):
            if isinstance(rel, dict) and all(k in rel for k in ["source", "target", "type"]):
                all_relationships.append(rel)

    unique_relationships_str = {json.dumps(d, sort_keys=True) for d in all_relationships}
    final_relationships = [json.loads(s) for s in unique_relationships_str]
    
    return {"entities": list(consolidated_entities.values()), "relationships": final_relationships}

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
    collection = client.get_or_create_collection(name="arc_career_guidance_v6", embedding_function=GeminiEmbeddingFunction())
    return collection, pdf_dir, history_dir, receipts_dir

COLLECTION, PDF_DIRECTORY, HISTORY_DIRECTORY, RECEIPTS_DIRECTORY = setup_database_and_directories()
KG_MANAGER = ChromaDBKnowledgeGraph(COLLECTION)
LOG_FILE_PATH = Path(__file__).parent / "conversation_log.txt"
SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "prompt.txt")

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
            receipt_file = RECEIPTS_DIRECTORY / f"{pdf_file.name}.receipt"
            if receipt_file.exists():
                receipt_mtime = receipt_file.stat().st_mtime
                pdf_mtime = pdf_file.stat().st_mtime
                if pdf_mtime <= receipt_mtime:
                    logging.info(f"📄 Skipping unchanged document: {pdf_file.name}")
                    continue

            logging.info(f"\n📄 Processing {pdf_file.name}...")
            COLLECTION.delete(where={"source": pdf_file.name})
            logging.info(f"  -> Cleared old entries for {pdf_file.name}.")

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

            if KG_MANAGER.enabled:
                all_extractions = [KG_MANAGER.extract_entities_with_gemini(chunk) for chunk in chunks]
                if any(ext.get("entities") for ext in all_extractions):
                    logging.info(f"  🤝 Consolidating knowledge graph for {pdf_file.name}...")
                    consolidated_graph = consolidate_knowledge_graph_for_document(all_extractions)
                    KG_MANAGER.store_knowledge_graph(consolidated_graph, pdf_file.name)

            receipt_file.touch()
            logging.info(f"  ✅ Created processing receipt for {pdf_file.name}")

        except Exception as e:
            logging.error(f"❌ Error processing {pdf_file.name}: {e}", exc_info=True)
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

# --- KG EXPORT FUNCTIONALITY ---
def export_knowledge_graph_to_csv():
    """Exports all entities and relationships from ChromaDB to CSV files."""
    try:
        logging.info("📈 Starting Knowledge Graph export to CSV...")
        batch_size = 1000 # Increased batch size for efficiency
        offset = 0
        all_items = []
        while True:
            batch = COLLECTION.get(offset=offset, limit=batch_size, include=["metadatas"])
            if not batch or not batch['ids']:
                break
            all_items.extend(batch['metadatas'])
            offset += len(batch['ids'])

        if not all_items:
            logging.warning("⚠️ No items found in the database to export.")
            return "No data to export."

        entities_path = Path(__file__).parent / "knowledge_graph_entities.csv"
        relationships_path = Path(__file__).parent / "knowledge_graph_relationships.csv"

        entities = []
        relationships = []

        for item in all_items:
            if item and item.get("chunk_type") == "entity":
                entities.append({
                    "entity_name": item.get("entity_name"),
                    "entity_type": item.get("entity_type"),
                    "source_document": item.get("source")
                })
            elif item and item.get("chunk_type") == "relationship":
                relationships.append({
                    "source_entity": item.get("source_entity"),
                    "relationship_type": item.get("relationship_type"),
                    "target_entity": item.get("target_entity"),
                    "source_document": item.get("source")
                })
        
        with open(entities_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["entity_name", "entity_type", "source_document"])
            writer.writeheader()
            writer.writerows(entities)
        
        with open(relationships_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["source_entity", "relationship_type", "target_entity", "source_document"])
            writer.writeheader()
            writer.writerows(relationships)

        logging.info(f"✅ Successfully exported {len(entities)} entities and {len(relationships)} relationships.")
        return f"Successfully exported {len(entities)} entities and {len(relationships)} relationships to CSV."
    except Exception as e:
        logging.error(f"❌ Failed to export knowledge graph: {e}", exc_info=True)
        return "Failed to export knowledge graph."

# --- RAG with Integrated Knowledge Graph ---
def enhanced_retrieval_and_generation(message: str, history: List[Dict]) -> Iterator[str]:
    if message.strip().lower() == "kg output":
        result = export_knowledge_graph_to_csv()
        yield f"data: {json.dumps({'type': 'text', 'payload': result})}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'payload': {'full_text': result}})}\n\n"
        return

    contextual_query = "\n".join([f"{t['role']}: {t['parts'][0]}" for t in history[-4:]]) + f"\nuser: {message}"
    
    doc_results = COLLECTION.query(query_texts=[contextual_query], n_results=7, where={"chunk_type": "document"})
    kg_results = COLLECTION.query(query_texts=[contextual_query], n_results=5, where={"$or": [{"chunk_type": "entity"}, {"chunk_type": "relationship"}]})
    
    retrieved_kg_facts = kg_results.get('documents', [[]])[0]
    retrieved_docs = list(zip(doc_results.get('documents', [[]])[0], doc_results.get('metadatas', [[]])[0]))

    context = "RELEVANT KNOWLEDGE GRAPH FACTS:\n" + "\n".join(f"- {fact}" for fact in retrieved_kg_facts) + "\n\nDOCUMENT EXCERPTS FOR CONTEXT:\n"
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
            prompt = f"{SYSTEM_PROMPT}\n\n---BEGIN CONTEXTUAL INFORMATION---\n{context}\n---END CONTEXTUAL INFORMATION---\n\nQuestion: {message}"
            
            response_stream = chat.send_message(prompt, stream=True)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'type': 'text', 'payload': chunk.text})}\n\n"
            
            yield f"data: {json.dumps({'type': 'end', 'payload': {'full_text': full_response}})}\n\n"
            return

        except google_exceptions.ResourceExhausted:
            logging.warning(f"⚠️ Chat response rate limit on key index {API_KEY_MANAGER.current_key_index}. Rotating key...")
            new_key = API_KEY_MANAGER.switch_to_next_key()
            genai.configure(api_key=new_key)
            
            if attempt < max_retries - 1:
                logging.info("Retrying chat generation with new key...")
            else:
                logging.error("❌ All API keys are rate-limited for chat. Aborting.")
                yield f"data: {json.dumps({'type': 'error', 'payload': 'All available API keys are currently rate-limited. Please try again in a few minutes.'})}\n\n"
        except Exception as e:
            logging.error(f"❌ An unexpected error occurred during chat generation: {e}")
            yield f"data: {json.dumps({'type': 'error', 'payload': 'An unexpected error occurred while generating the response.'})}\n\n"
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
        citations = [] # Store citations to save them in history
        
        for event_str in enhanced_retrieval_and_generation(message, history):
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

        # --- FIX for History Saving ---
        # This logic now runs for all messages that get a full response.
        if full_response:
            # Don't save the KG export command itself to the history to avoid clutter
            if message.strip().lower() != "kg output":
                history.append({"role": "user", "parts": [message]})
                # Include citations in the saved model response
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
    # --- FIX for Loading History ---
    # Return success: false if history is not found or empty
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
if __name__ == '__main__':
    # Add a static route for serving documents
    @app.route('/documents/<path:filename>')
    def serve_document(filename):
        return send_from_directory(PDF_DIRECTORY, filename, as_attachment=False)

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        process_all_pdfs()
    app.run(host='0.0.0.0', port=5003, debug=True)
