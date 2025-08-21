# --- Imports ---
import os
import json
import time
import traceback
import shutil
import sqlite3
import stat
import csv
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

# --- Configuration ---
try:
    # Using a valid and recent model name as per instructions
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME = "models/embedding-001"

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    logging.info("✅ Google Generative AI SDK configured successfully.")
except Exception as e:
    logging.critical(f"❌ Configuration Error: {e}")
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
            valid_input = [doc for doc in input if doc and doc.strip()]
            if not valid_input:
                logging.warning("⚠️ Embedding input was empty after filtering. Returning empty list.")
                return []

            result = genai.embed_content(
                model=self._model_name,
                content=valid_input,
                task_type=self._task_type
            )
            return result['embedding']
        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"⚠️ Embedding rate limit hit: {e}. Pausing for 5s.")
            time.sleep(5)
            try:
                result = genai.embed_content(
                    model=self._model_name,
                    content=valid_input,
                    task_type=self._task_type
                )
                return result['embedding']
            except Exception as retry_e:
                logging.error(f"⚠️ Embedding retry failed: {retry_e}. Returning zero vectors.")
                return [[0.0] * 768 for _ in valid_input]
        except Exception as e:
            logging.error(f"⚠️ Embedding error: {e}. Returning zero vectors for input of size {len(input)}.")
            return [[0.0] * 768 for _ in input]


# --- Knowledge Graph Manager (Enhanced for Career Guidance) ---
class ChromaDBKnowledgeGraph:
    """Manages detailed entity and relationship extraction for career guidance."""
    def __init__(self, collection):
        self.collection = collection
        self.enabled = True
        logging.info("✅ ChromaDB Knowledge Graph manager initialized.")

    def extract_entities_with_gemini(self, text: str) -> Dict[str, Any]:
        """Uses Gemini to extract detailed, career-focused entities and relationships."""
        if not text.strip():
            return {"entities": [], "relationships": []}

        # Step 1 & 2: Updated Prompt with Characteristic Entity, Relationship, and new instructions
        prompt = f"""From the following text, extract key entities and their relationships. You must analyze the text against the 14 characteristics of the Refracted Economies Framework, which include spectrums like Skilled-Unskilled, Formal-Informal, and Permanent-Gig.

**Schema:**
* **Entity Types:**
    * `Career_Pathway`: Specific job roles or career tracks (e.g., "Data Scientist", "AI Ethics Officer").
    * `Skill`: Specific abilities or knowledge areas (e.g., "Python Programming", "Critical Thinking").
    * `Industry`: A broad sector of the economy (e.g., "Financial Technology", "Aquaculture").
    * `Economic_Trend`: High-level economic or technological shifts (e.g., "AI Automation", "Gig Economy").
    * `Organization`: Companies, institutions, or bodies mentioned.
    * `Policy_Or_Strategy`: A named government or organizational initiative (e.g., "Operation Phakisa").
    * `Government_Body`: A specific government department or agency (e.g., "Ministry of Education").
    * `Educational_Qualification`: A degree, certificate, or exam (e.g., "Bachelor of Science", "Ethiopian School Leaving Examination").
    * `Location`: A country, city, or region mentioned (e.g., "Ghana", "Nairobi").
    * `Characteristic`: A specific quality of a job from the REF (e.g., "Skilled", "Formal", "Gig", "Knowledge-based", "Elastic").
* **Relationship Types:**
    * `REQUIRES_SKILL` (Career_Pathway -> Skill)
    * `PART_OF_INDUSTRY` (Career_Pathway -> Industry)
    * `LEADS_TO` (Skill -> Career_Pathway)
    * `INFLUENCED_BY` (Industry -> Economic_Trend)
    * `IS_SYNONYM_FOR` (Organization -> Organization): Links an acronym to a full name.
    * `HAS_PREREQUISITE` (Career_Pathway -> Educational_Qualification)
    * `GOVERNED_BY` (Educational_Qualification -> Government_Body)
    * `IMPLEMENTS` (Government_Body -> Policy_Or_Strategy)
    * `LOCATED_IN` (Organization -> Location)
    * `HAS_CHARACTERISTIC` (Career_Pathway -> Characteristic)

**Extraction Rules:**
1.  **Synonyms:** If an entity is mentioned with both a full name and an acronym (e.g., Ministry of Education and MoE), you MUST create one canonical entity with the full name and use the `IS_SYNONYM_FOR` relationship to link the acronym to it.
2.  **Careers vs. Industries:** An `Industry` is a broad sector (e.g., 'Aquaculture'). A `Career_Pathway` is a specific job within it (e.g., 'Hatchery Manager'). Always create a `PART_OF_INDUSTRY` relationship from the career to the industry.
3.  **Canonical Names:** Use the most complete and formal name for an entity as its primary name.

**Logical Sequence & Main Task:**
1.  First, identify all potential entities in the text. For every `Career_Pathway` you identify, you must also identify its associated characteristics from the text.
2.  Second, normalize them to their canonical names. Create a distinct `Characteristic` node for each trait found.
3.  Third, identify the relationships between these normalized entities, linking careers to their traits using the `HAS_CHARACTERISTIC` relationship.

Return the output as a single, valid JSON object. Do not include any other text or markdown formatting.

**Text for Analysis:**
{text}
"""
        response = None
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192
            )
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            
            response = model.generate_content(
                prompt, 
                generation_config=generation_config,
                request_options={"timeout": 180},
                safety_settings=safety_settings
            )
            
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            parsed_json = json.loads(cleaned_text)
            entity_count = len(parsed_json.get("entities", []))
            rel_count = len(parsed_json.get("relationships", []))
            logging.info(f"  -> Extracted {entity_count} entities and {rel_count} relationships from chunk.")
            return parsed_json

        except google_exceptions.ResourceExhausted as e:
            logging.warning(f"⚠️ Knowledge Graph extraction rate limit. Pausing for 60s.")
            time.sleep(60)
            return {"entities": [], "relationships": []}
        except (json.JSONDecodeError, Exception) as e:
            response_text_for_error = response.text if response and hasattr(response, 'text') else 'No response text available'
            logging.error(f"⚠️ Knowledge Graph extraction failed: {e}\nResponse Text: {response_text_for_error[:200]}\nTraceback: {traceback.format_exc()}")
            return {"entities": [], "relationships": []}

    def store_knowledge_graph(self, extraction_data: Dict[str, Any], source_doc: str):
        """Stores consolidated entities and relationships in ChromaDB."""
        if not extraction_data or (not extraction_data.get("entities") and not extraction_data.get("relationships")):
            logging.info(f"  -> No consolidated knowledge graph data to store for {source_doc}.")
            return

        documents, metadatas, ids = [], [], []

        for entity in extraction_data.get("entities", []):
            if not all(k in entity for k in ["name", "type"]): continue
            doc_text = f"Entity: {entity['name']} ({entity['type']})"
            sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', entity['name'].replace(' ', '_'))
            documents.append(doc_text)
            metadatas.append({
                "chunk_type": "entity", "source": source_doc, "entity_name": entity['name'],
                "entity_type": entity['type']
            })
            ids.append(f"entity_{source_doc}_{sanitized_name}")

        for rel in extraction_data.get("relationships", []):
            if not all(k in rel for k in ["source", "target", "type"]): continue
            doc_text = f"Relationship: {rel['source']} -> {rel['type']} -> {rel['target']}"
            sanitized_rel = re.sub(r'[^a-zA-Z0-9_-]', '', f"{rel['source']}_{rel['target']}_{rel['type']}")
            documents.append(doc_text)
            metadatas.append({
                "chunk_type": "relationship", "source": source_doc, "source_entity": rel['source'],
                "target_entity": rel['target'], "relationship_type": rel['type']
            })
            ids.append(f"rel_{source_doc}_{sanitized_rel}")

        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logging.info(f"  ✅ Stored consolidated graph with {len(extraction_data.get('entities',[]))} entities and {len(extraction_data.get('relationships',[]))} relationships for {source_doc}")
            except Exception as e:
                logging.error(f"❌ Error adding consolidated KG data to ChromaDB for {source_doc}: {e}")

# --- Phase 3: Post-Extraction Consolidation ---
def consolidate_knowledge_graph_for_document(all_chunk_extractions: List[Dict]) -> Dict[str, Any]:
    """Merges entities and relationships from all chunks of a single document."""
    consolidated_entities = {}
    all_relationships = []
    
    # Pre-defined aliases for major entities
    known_aliases = {
        'GTEC': 'Ghana Tertiary Education Commission',
        'ESSLE': 'Ethiopian School Leaving Examination',
        'MoE': 'Ministry of Education'
    }

    # First pass: collect all entities and relationships
    for data in all_chunk_extractions:
        for entity in data.get("entities", []):
            # Use lowercased name as a key to handle case variations
            key = entity["name"].lower()
            if key not in consolidated_entities:
                consolidated_entities[key] = {"name": entity["name"], "type": entity["type"]}
        all_relationships.extend(data.get("relationships", []))
        
    # Build a map of all aliases to their canonical names
    alias_to_canonical = {alias.lower(): canonical for alias, canonical in known_aliases.items()}
    for rel in all_relationships:
        if rel.get("type") == "IS_SYNONYM_FOR":
            source_lower = rel["source"].lower()
            target_lower = rel["target"].lower()
            if source_lower in consolidated_entities and target_lower in consolidated_entities:
                # Assume target is the canonical name
                alias_to_canonical[source_lower] = consolidated_entities[target_lower]["name"]

    # Second pass: resolve aliases in relationships
    resolved_relationships = []
    for rel in all_relationships:
        source_lower = rel["source"].lower()
        target_lower = rel["target"].lower()
        
        canonical_source = alias_to_canonical.get(source_lower, rel["source"])
        canonical_target = alias_to_canonical.get(target_lower, rel["target"])
        
        resolved_relationships.append({
            "source": canonical_source,
            "target": canonical_target,
            "type": rel["type"]
        })

    # Final lists, removing duplicates
    final_entities_dict = {v["name"].lower(): v for k, v in consolidated_entities.items()}
    final_entities = list(final_entities_dict.values())
    
    final_relationships_set = set()
    for rel in resolved_relationships:
        final_relationships_set.add(
            (rel["source"], rel["type"], rel["target"])
        )
    final_relationships = [{"source": s, "type": t, "target": r} for s, t, r in final_relationships_set]
    
    return {"entities": final_entities, "relationships": final_relationships}


# --- Core Application Setup ---
app = Flask(__name__)

# --- Error handler for shutil.rmtree on Windows ---
def remove_readonly(func, path, excinfo):
    excvalue = excinfo[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == 13:
        logging.warning(f"Attempting to clear read-only flag on: {path}")
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

# --- Directory and Database Initialization ---
def setup_database_and_directories():
    base_dir = Path(__file__).parent
    storage_path = base_dir / "arc_chroma_db_storage"
    pdf_dir = base_dir / "documents"
    history_dir = base_dir / "history"

    pdf_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)
    
    logging.info(f"📁 PDF Directory: {pdf_dir.resolve()}")
    logging.info(f"📁 History Directory: {history_dir.resolve()}")
    logging.info(f"📁 ChromaDB Storage: {storage_path.resolve()}")

    storage_path.mkdir(parents=True, exist_ok=True) 
    
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
LOG_FILE_PATH = Path(__file__).parent / "conversation_log.txt"

# --- Document Processing ---
def process_all_pdfs():
    logging.info("\n🔍 Scanning for PDF documents...")
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning("📂 No PDF files found. Add PDFs to the 'documents' folder and restart.")
        return

    # Phase 1: Robust Document Chunking for Extraction
    kg_text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
    doc_text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)


    logging.info(f"📁 Found {len(pdf_files)} PDF file(s) to process.")
    for pdf_file in pdf_files:
        try:
            where_filter_doc = {"$and": [{"source": pdf_file.name}, {"chunk_type": "document"}]}
            where_filter_entity = {"$and": [{"source": pdf_file.name}, {"chunk_type": "entity"}]}

            doc_chunks_exist = COLLECTION.get(where=where_filter_doc, limit=1)['ids']
            entity_chunks_exist = COLLECTION.get(where=where_filter_entity, limit=1)['ids']

            if doc_chunks_exist and entity_chunks_exist:
                logging.info(f"📄 Skipping fully processed document: {pdf_file.name}")
                continue
            
            if doc_chunks_exist or entity_chunks_exist:
                logging.warning(f"⚠️ Incomplete processing detected for {pdf_file.name}. Deleting and reprocessing.")
                ids_to_delete = COLLECTION.get(where={"source": pdf_file.name})['ids']
                if ids_to_delete:
                    COLLECTION.delete(ids=ids_to_delete)
                    logging.info(f"  🗑️ Deleted {len(ids_to_delete)} incomplete entries for {pdf_file.name}.")

            logging.info(f"\n📄 Processing {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            full_document_text = "\n".join([page.page_content for page in pages])
            
            # Add document chunks
            doc_chunks = doc_text_splitter.split_text(full_document_text)
            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(doc_chunks):
                if chunk and chunk.strip():
                    documents.append(chunk)
                    metadatas.append({"source": pdf_file.name, "chunk_index": i + 1, "chunk_type": "document"})
                    ids.append(f"{pdf_file.stem}_chunk_{i}")
            if documents:
                COLLECTION.add(documents=documents, metadatas=metadatas, ids=ids)
                logging.info(f"  ✅ Added {len(documents)} text chunks to ChromaDB.")
            
            # Extract and consolidate KG from chunks
            if KG_MANAGER.enabled:
                kg_chunks = kg_text_splitter.split_text(full_document_text)
                all_extractions = []
                logging.info(f"  🔬 Extracting knowledge graph from {len(kg_chunks)} chunks...")
                for chunk in kg_chunks:
                    kg_data = KG_MANAGER.extract_entities_with_gemini(chunk)
                    if kg_data.get("entities"):
                        all_extractions.append(kg_data)
                
                if all_extractions:
                    logging.info(f"  🤝 Consolidating knowledge graph for {pdf_file.name}...")
                    consolidated_graph = consolidate_knowledge_graph_for_document(all_extractions)
                    KG_MANAGER.store_knowledge_graph(consolidated_graph, pdf_file.name)
                else:
                    logging.info(f"  -> No entities found to build knowledge graph for {pdf_file.name}.")


        except Exception as e:
            logging.error(f"❌ Error processing {pdf_file.name}: {e}", exc_info=True)
    
    logging.info(f"\n🎉 PDF processing complete. Total items in collection: {COLLECTION.count()}")

# --- Chat History and Logging Functions ---
def save_chat_history(session_id: str, history: List[Dict]):
    if not re.match(r'^session_[a-zA-Z0-9_]+$', session_id):
        logging.error(f"Invalid session_id format for saving: {session_id}")
        return
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        logging.error(f"Error saving chat history for session {session_id}: {e}")

def load_chat_history(session_id: str) -> List[Dict]:
    if not re.match(r'^session_[a-zA-Z0-9_]+$', session_id):
        logging.error(f"Invalid session_id format for loading: {session_id}")
        return []
    history_file = HISTORY_DIRECTORY / f"{session_id}.json"
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading/decoding history for session {session_id}: {e}. Returning empty list.")
            return []
    return []

def log_conversation(session_id: str, question: str, answer: str):
    """Appends a question and its full answer to the conversation log file."""
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- Log Entry: {timestamp} ---\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("---------------------------------\n\n")
    except IOError as e:
        logging.error(f"❌ Could not write to log file: {e}")

# --- Knowledge Graph Export Function ---
def export_knowledge_graph_to_csv():
    """Exports the entire knowledge graph from ChromaDB to CSV files."""
    try:
        # 1. Export Entities
        entities_data = COLLECTION.get(where={"chunk_type": "entity"}, include=["metadatas"])
        entities_metadatas = entities_data.get('metadatas', [])
        
        entities_filepath = Path(__file__).parent / "knowledge_graph_entities.csv"
        with open(entities_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["entity_name", "entity_type", "source_document"])
            if entities_metadatas:
                for meta in entities_metadatas:
                    writer.writerow([
                        meta.get("entity_name", ""),
                        meta.get("entity_type", ""),
                        meta.get("source", "")
                    ])
        
        # 2. Export Relationships
        relationships_data = COLLECTION.get(where={"chunk_type": "relationship"}, include=["metadatas"])
        relationships_metadatas = relationships_data.get('metadatas', [])

        relationships_filepath = Path(__file__).parent / "knowledge_graph_relationships.csv"
        with open(relationships_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["source_entity", "relationship_type", "target_entity", "source_document"])
            if relationships_metadatas:
                for meta in relationships_metadatas:
                    writer.writerow([
                        meta.get("source_entity", ""),
                        meta.get("relationship_type", ""),
                        meta.get("target_entity", ""),
                        meta.get("source", "")
                    ])
        
        num_entities = len(entities_metadatas)
        num_rels = len(relationships_metadatas)
        logging.info(f"✅ Successfully exported {num_entities} entities and {num_rels} relationships to CSV.")
        return f"Successfully exported {num_entities} entities to `knowledge_graph_entities.csv` and {num_rels} relationships to `knowledge_graph_relationships.csv`."

    except Exception as e:
        logging.error(f"❌ Failed to export knowledge graph to CSV: {e}", exc_info=True)
        return "Sorry, I encountered an error while exporting the knowledge graph."

# --- Phase 4: Analytical Functions ---
def compare_admissions(country1: str, country2: str) -> str:
    """Compares admission systems of two countries using the knowledge graph."""
    try:
        response = f"**Admissions Comparison: {country1.title()} vs {country2.title()}**\n\n"
        for country in [country1, country2]:
            response += f"--- **{country.title()}** ---\n"
            # Find governing bodies in the country
            gov_bodies = COLLECTION.get(where={"$and": [{"entity_type": "Government_Body"}, {"source": {"$like": f"%{country}%"}}]}, include=["metadatas"])
            
            if not gov_bodies['ids']:
                response += "No governing body information found.\n"
                continue

            gov_body_names = [meta['entity_name'] for meta in gov_bodies['metadatas']]
            
            # Find qualifications governed by these bodies
            quals = COLLECTION.get(where={"$and": [{"relationship_type": "GOVERNED_BY"}, {"target_entity": {"$in": gov_body_names}}]}, include=["metadatas"])
            
            if not quals['ids']:
                response += f"Governing Body: {', '.join(gov_body_names)}\n"
                response += "No specific admission qualifications found linked to this body.\n\n"
            else:
                qual_names = [meta['source_entity'] for meta in quals['metadatas']]
                response += f"Governing Body: {', '.join(gov_body_names)}\n"
                response += f"Key Qualifications: {', '.join(qual_names)}\n\n"
        return response
    except Exception as e:
        logging.error(f"Error in compare_admissions: {e}")
        return "Could not perform comparison due to an internal error."

def trace_policy_to_career(policy_name: str) -> str:
    """Traces a policy to its related industries and careers."""
    try:
        response = f"**Impact Trace for Policy: {policy_name}**\n\n"
        # Find industries implemented by the policy
        rels = COLLECTION.get(where={"$and": [{"relationship_type": "IMPLEMENTS"}, {"target_entity": {"$like": f"%{policy_name}%"}}]}, include=["metadatas"])

        if not rels['ids']:
            return f"No information found for policy: {policy_name}"

        gov_bodies = [meta['source_entity'] for meta in rels['metadatas']]
        response += f"Implemented by: {', '.join(gov_bodies)}\n"

        # This is a simplified trace; a real one would need multi-hop graph traversal
        # For now, we find careers in the same source documents as the policy
        policy_docs = [meta['source'] for meta in rels['metadatas']]
        careers = COLLECTION.get(where={"$and": [{"entity_type": "Career_Pathway"}, {"source": {"$in": policy_docs}}]}, include=["metadatas"])
        
        if careers['ids']:
            career_names = list(set([meta['entity_name'] for meta in careers['metadatas']]))
            response += f"Associated Career Pathways: {', '.join(career_names)}\n"
        else:
            response += "No directly associated career pathways found in the same documents.\n"
            
        return response
    except Exception as e:
        logging.error(f"Error in trace_policy_to_career: {e}")
        return "Could not perform trace due to an internal error."

def map_industry(industry_name: str, location: str) -> str:
    """Maps out the ecosystem of an industry in a specific location."""
    try:
        response = f"**Ecosystem Map for {industry_name} in {location.title()}**\n\n"
        # Find careers in the industry and location
        careers = COLLECTION.get(where={"$and": [
            {"entity_type": "Career_Pathway"},
            {"source": {"$like": f"%{location}%"}}
        ]}, include=["metadatas"])

        if not careers['ids']:
            return f"No information found for {industry_name} in {location}."

        # This is a simplified map; a real one would traverse the graph
        # We will filter careers that have a PART_OF_INDUSTRY relationship
        industry_rels = COLLECTION.get(where={"$and": [
            {"relationship_type": "PART_OF_INDUSTRY"},
            {"target_entity": {"$like": f"%{industry_name}%"}}
        ]}, include=["metadatas"])

        if not industry_rels['ids']:
             return f"No careers explicitly linked to the '{industry_name}' industry found."

        career_names = list(set([meta['source_entity'] for meta in industry_rels['metadatas']]))
        response += f"**Career Pathways:**\n- " + "\n- ".join(career_names) + "\n\n"

        # Find skills for these careers
        skills_rels = COLLECTION.get(where={"$and": [
            {"relationship_type": "REQUIRES_SKILL"},
            {"source_entity": {"$in": career_names}}
        ]}, include=["metadatas"])
        
        if skills_rels['ids']:
            skill_names = list(set([meta['target_entity'] for meta in skills_rels['metadatas']]))
            response += f"**Associated Skills:**\n- " + "\n- ".join(skill_names) + "\n"

        return response
    except Exception as e:
        logging.error(f"Error in map_industry: {e}")
        return "Could not perform mapping due to an internal error."

# --- Retrieval and Generation ---
def enhanced_retrieval_and_generation(message: str, history: List[Dict]) -> Iterator[str]:
    history_for_query_context = [
        f"{turn.get('role', 'unknown')}: {turn.get('parts', [''])[0]}" 
        for turn in history[-4:] if turn.get('parts')
    ]
    contextual_query = "\n".join(history_for_query_context) + f"\nuser: {message}"

    logging.info(f"🔎 Contextual Query for Retrieval:\n---\n{contextual_query}\n---")

    retrieved_docs = []
    try:
        query_results = COLLECTION.query(
            query_texts=[contextual_query],
            n_results=10,
            where={"chunk_type": "document"}
        )
        documents = query_results.get('documents', [[]])[0]
        metadatas = query_results.get('metadatas', [[]])[0]
        retrieved_docs = list(zip(documents, metadatas))
    except Exception as e:
        logging.warning(f"⚠️ Failed to retrieve from ChromaDB: {e}")

    retrieved_context = "DOCUMENT EXCERPTS FOR CONTEXT:\n"
    citations = []
    if not retrieved_docs:
        retrieved_context += "No relevant documents were found for this query.\n"
    else:
        for i, (doc, meta) in enumerate(retrieved_docs):
            cleaned_doc = ' '.join(doc.split())
            
            source = meta.get('source', 'Unknown')
            page_info = f"Chunk {meta.get('chunk_index', 'N/A')}" if meta.get('chunk_type') == 'document' else meta.get('entity_type', 'Knowledge Graph')
            citation_index = i + 1
            retrieved_context += f"[{citation_index}] Source: {source}, Detail: {page_info}\nContent: {cleaned_doc}\n\n"
            citations.append({
                "id": str(citation_index), 
                "source": source, 
                "page_number": page_info,
                "content": cleaned_doc,
            })
    
    citations_event = {"type": "citations", "payload": citations}
    yield f"data: {json.dumps(citations_event)}\n\n"

    system_prompt = (
        "You are an AI Powered Career Guidance Tool, an expert AI specializing in the Refracted Economies Framework (REF). "
        "Your purpose is to help users, particularly young Africans, understand and navigate the complex career landscape. "
        "Your primary goal is to provide clear, relevant, and appropriately detailed answers based on the user's question type.\n\n"

        "Your analysis and  responses MUST be grounded in the REF's principles. Your purpose is to help users, particularly young Africans, "
        "understand and navigate the complex career landscape using this specific framework. "
        "DO NOT provide generic career advice. Every answer must be interpreted and explained through the REF lens.\n\n"
        
        "**Core Principles of Your Analysis:**\n"
        "1.  Understand the colour-coded economies of the Refracted Economies Framework. The economies are:\n"
        "    - **Orange:** Creative, cultural, leisure (arts, media, sports, fashion).\n"
        "    - **Green:** Environmental sustainability (renewables, conservation).\n"
        "    - **Blue:** Water-based resources (fishing, maritime transport).\n"
        "    - **Lavender:** Care and helping professions (healthcare, social work).\n"
        "    - **Yellow:** Public and social sector (government, education, NGOs).\n"
        "    - **Bronze:** Extraction and cultivation (mining, agriculture).\n"
        "    - **Iron:** Manufacturing, distribution, infrastructure (construction, logistics).\n"
        "    - **Gold:** Financial services (banking, fintech).\n"
        "    - **Platinum:** Technology and innovation (IT, AI, software dev).\n"
        "2.  Understand that all forms of work have characteristics. The characteristics exist on a spectrum noting that the broad dimensions are:\n"
        "    - **Skill/Knowledge:** Skilled-Unskilled, Knowledge-Physical.\n"
        "    - **Adaptability/Innovation:** Elastic-Inelastic (resilience to AI), Entrepreneurial-Imitative, New-Traditional.\n"
        "    - **Organizational/Social:** Formal-Informal, Private-Public, Individual-Collective, Permanent-Gig.\n"
        "    - **Sustainability/Values:** Sustainable-Finite, Creative, Compliant, Decent and Dignified, Meaningful.\n"
        "3.  **Recognize Intersections:** Acknowledge that a single career can span multiple economies.\n"
       
        "**IMPORTANT: First, analyze the user's question to determine its type, then respond according to the specific instructions for that type.**\n\n"        
        "---"
        "### Type 1: Career Opportunity Questions\n"
        "These are practical questions about jobs, skills, or educational paths (e.g., 'What jobs are in the blue economy?', 'What subjects do I need for engineering?', 'How can I get into agriculture?').\n\n"
        "**Instructions for Career Questions:**\n"
        "1.  **Be Direct and Concise:** Give a short, punchy, and to-the-point answer. Get straight to the practical advice. Avoid long introductions.\n"
        "2.  **Mention REF Economy Briefly:** Briefly mention the primary REF color-coded economy/economies relevant to the career (e.g., 'Engineering is mainly in the Iron and Platinum Economies.'). Do NOT list or describe all nine economies.\n"
        "3.  **Limit Characteristics:** Do NOT explain the 14 characteristics matrix. Only mention a key characteristic if it's essential to answer the question directly (e.g., 'It's a highly **Skilled** role, so you'll need a degree.').\n"
        "4.  **Cite Your Sources:** Base your answer on the provided document excerpts and cite them with numbered references like [1], [2], etc.\n"
        "5.  **Use Simple Formatting:** Use lists and bold text for clarity, but keep the overall response brief.\n\n"
        "5.  **Prerequiste Queries:** Examine the information about the undergraduate admission prerequisites before presenting an answer.\n\n"
        
        "---"

        "### Type 2: Academic or Framework Questions\n"
        "These are questions about the REF itself (e.g., 'What is the Refracted Economies Framework?', 'Explain the characteristics matrix.', 'What is the philosophy behind the REF?').\n\n"
        "**Instructions for Academic Questions:**\n"
        "1.  **Be Comprehensive:** Provide a detailed and thorough explanation. This is where you should elaborate on the framework's structure and philosophy.\n"
        "2.  **Explain the Concepts:** Fully explain the color-coded economies and the 14 characteristics as needed to answer the question. This is the time for detailed descriptions.\n"
        "3.  **Recognize Intersections:** Explain how different economies can intersect.\n"
        "4.  **Dignify All Work:** Reinforce the REF's core philosophy of valuing all contributions to society.\n"
        "5.  **Cite Your Sources:** Base your analysis on the provided document excerpts and cite them correctly.\n"
        "6.  **Use Clear Formatting:** Use Markdown (headings, lists, bold text) to structure your detailed response.\n\n"

        "---"

        "### Type 3: Off-Topic Questions\n"
        "These are questions unrelated to career guidance, economics, skills, or the REF, or the prior context of previous questions and answers.\n\n"
        "**Instructions for Off-Topic Questions:**\n"
        "1.  **Redirect Politely:** Do not answer the question. Instead, gently guide the user back to your primary purpose.\n"
        "2.  **Use a Standard Response:** Respond with something similar to: 'My purpose is to be an AI career guidance tool using the Refracted Economies Framework. I can help you with questions about careers, industries, and the skills needed for them. How can I assist you with your career journey?'\n\n"

        "---"
        
        "**Final instruction for all responses:** Always use the provided document excerpts to ground your answers in facts. When you use information from an excerpt, you MUST cite it. Do not refer to the Question Types"
    )
    
    model_history = []
    for turn in history:
        role = 'user' if turn.get('role') == 'user' else 'model'
        parts = [str(p) for p in turn.get('parts', []) if p]
        if parts:
            model_history.append({'role': role, 'parts': parts})

    prompt_for_model = (
        f"{system_prompt}\n\n"
        f"---BEGIN DOCUMENT EXCERPTS---\n{retrieved_context}\n---END DOCUMENT EXCERPTS---\n\n"
        "Based on the conversation history and the document excerpts above, answer the user's latest question. "
        f"**User's Question**: {message}"
    )
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        chat_session = model.start_chat(history=model_history)
        response_stream = chat_session.send_message(prompt_for_model, stream=True)
        
        full_response_text = ""
        for chunk in response_stream:
            if chunk.text:
                full_response_text += chunk.text
                text_event = {"type": "text", "payload": chunk.text}
                yield f"data: {json.dumps(text_event)}\n\n"
        
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
        final_data = {"type": "final_content", "text": full_response_text, "citations": citations}
        yield f"data: {json.dumps(final_data)}\n\n"

    except Exception as e:
        logging.error(f"❌ LLM generation error: {e}", exc_info=True)
        error_event = {"type": "error", "payload": "Sorry, I encountered an error generating a response."}
        yield f"data: {json.dumps(error_event)}\n\n"


# --- Flask API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    session_id = data.get('session_id')
    
    if not message or not session_id:
        return jsonify({'success': False, 'response': 'Invalid request.'}), 400

    msg_lower = message.lower()

    # --- Analytical Query Router ---
    def stream_analytical_query(response_message):
        event_data = {"type": "text", "payload": response_message}
        yield f"data: {json.dumps(event_data)}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        history = load_chat_history(session_id)
        history.append({"role": "user", "parts": [message]})
        history.append({"role": "model", "parts": [response_message], "citations": []})
        save_chat_history(session_id, history)
        log_conversation(session_id, message, response_message)

    if msg_lower == "kg output":
        return Response(stream_analytical_query(export_knowledge_graph_to_csv()), mimetype='text/event-stream')
    
    if msg_lower.startswith("compare admissions:"):
        try:
            countries = msg_lower.replace("compare admissions:", "").strip().split("vs")
            country1 = countries[0].strip()
            country2 = countries[1].strip()
            return Response(stream_analytical_query(compare_admissions(country1, country2)), mimetype='text/event-stream')
        except IndexError:
            return Response(stream_analytical_query("Please format your request as 'compare admissions: [country1] vs [country2]'."), mimetype='text/event-stream')

    if msg_lower.startswith("trace policy:"):
        policy_name = msg_lower.replace("trace policy:", "").strip()
        return Response(stream_analytical_query(trace_policy_to_career(policy_name)), mimetype='text/event-stream')

    if msg_lower.startswith("map industry:"):
        try:
            parts = msg_lower.replace("map industry:", "").strip().split("in")
            industry = parts[0].strip()
            location = parts[1].strip()
            return Response(stream_analytical_query(map_industry(industry, location)), mimetype='text/event-stream')
        except IndexError:
            return Response(stream_analytical_query("Please format your request as 'map industry: [industry] in [location]'."), mimetype='text/event-stream')


    history = load_chat_history(session_id)
    
    def stream_and_save():
        full_response_text = ""
        citations_data = []
        
        response_generator = enhanced_retrieval_and_generation(message, history)
        
        for event_str in response_generator:
            yield event_str
            
            if event_str.strip().startswith('data:'):
                try:
                    data_str = event_str[len('data:'):].strip()
                    event_data = json.loads(data_str)
                    if event_data.get("type") == "final_content":
                        full_response_text = event_data["text"]
                        citations_data = event_data["citations"]
                except json.JSONDecodeError:
                    pass
        
        if full_response_text:
            history.append({"role": "user", "parts": [message]})
            history.append({"role": "model", "parts": [full_response_text], "citations": citations_data})
            save_chat_history(session_id, history)
            log_conversation(session_id, message, full_response_text)
            logging.info(f"✅ History and conversation log saved for session {session_id}")

    return Response(stream_and_save(), mimetype='text/event-stream')


@app.route('/documents/<path:filename>')
def serve_document(filename):
    return send_from_directory(PDF_DIRECTORY, filename, as_attachment=False)

# --- Session Management Endpoints ---
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = []
    for session_file in sorted(HISTORY_DIRECTORY.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            session_id = session_file.stem
            history = load_chat_history(session_id)
            preview = "New Chat Session"
            if history:
                if isinstance(history[-1], dict) and history[-1].get('session_name'):
                    preview = history[-1]['session_name']
                else:
                    for turn in history:
                        if turn.get('role') == 'user' and turn.get('parts'):
                            preview = turn['parts'][0]
                            break
            sessions.append({
                "id": session_id,
                "preview": preview[:50] + '...' if len(preview) > 50 else preview
            })
        except Exception as e:
            logging.error(f"Failed to process session file {session_file.name}: {e}")
    return jsonify({"success": True, "sessions": sessions})

@app.route('/api/history', methods=['GET'])
def get_history():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"success": False, "message": "Session ID is required."}), 400
    history = load_chat_history(session_id)
    return jsonify({"success": True, "history": history})

@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if not re.match(r'^session_[a-zA-Z0-9_]+$', session_id):
        return jsonify({"success": False, "message": "Invalid session ID."}), 400
    session_file = HISTORY_DIRECTORY / f"{session_id}.json"
    if session_file.exists():
        try:
            session_file.unlink()
            return jsonify({"success": True, "message": "Session deleted."})
        except OSError as e:
            logging.error(f"Error deleting file {session_file}: {e}")
            return jsonify({"success": False, "message": "Error deleting session."}), 500
    return jsonify({"success": False, "message": "Session not found."}), 404

@app.route('/api/rename_session', methods=['POST'])
def rename_session():
    data = request.json
    session_id = data.get('session_id')
    new_name = data.get('name', '').strip()

    if not all([session_id, new_name]):
        return jsonify({"success": False, "message": "Session ID and new name are required."}), 400

    history = load_chat_history(session_id)
    if not history:
        return jsonify({"success": False, "message": "Session not found."}), 404

    if history and isinstance(history[-1], dict) and history[-1].get('session_name'):
         history[-1]['session_name'] = new_name
    else:
        history.append({'session_name': new_name})
    save_chat_history(session_id, history)
    return jsonify({"success": True, "message": "Session renamed successfully."})


# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
            process_all_pdfs()
    app.run(host='0.0.0.0', port=5003, debug=True)