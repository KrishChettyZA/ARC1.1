# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader  # For loading PDFs
import os
from pathlib import Path
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
import traceback # Import traceback for better error logging

load_dotenv()

# --- Configuration ---
# Configure Google Generative AI SDK with API key validation
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable")
    genai.configure(api_key=api_key)
    print("Google Generative AI SDK configured successfully.")
except Exception as e:
    print(f"Critical error during initialization: {e}")
    # Re-raise the exception to stop the app if API key is missing
    # This prevents the app from starting if it can't connect to the AI service
    traceback.print_exc()
    raise

# --- Flask App Setup ---
app = Flask(__name__)
# Use a stronger, randomly generated secret key in production
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- Vector Database Setup ---
# Using in-memory client for simplicity in this example.
# For persistence, use PersistentClient:
# client = chromadb.PersistentClient(path="./chroma_storage")
client = chromadb.Client()

# Ensure the documents directory exists
pdf_directory = Path(__file__).parent / "documents"
pdf_directory.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
print(f"ABSOLUTE PATH FOR DOCUMENTS: {pdf_directory.resolve()}")

# --- Custom Embedding Function ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name="models/embedding-001"):
        self._model_name = model_name
        print(f"GeminiEmbeddingFunction initialized with model: {self._model_name}")
        try:
            # Test embedding a simple string to catch potential issues early
            # Use a dummy text input
            test_input = ["This is a test sentence."]
            genai.embed_content(model=self._model_name, content=test_input, task_type="retrieval_document")
            print(f"Embedding model '{self._model_name}' tested successfully.")
        except Exception as e:
            print(f"Warning: Could not test embedding model '{self._model_name}': {e}")
            print("Please ensure the model name is correct and available for your API key.")
            # You might want to handle this more strictly in production

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """
        Generates embeddings for a list of documents (strings).
        Args:
            input: A list of strings (documents).
        Returns:
            A list of embeddings (list of lists of floats).
        """
        embeddings = []
        # Gemini's embed_content can take a list of strings for batching
        # but the Chromadb EmbeddingFunction.__call__ is often invoked with a single string or small batch.
        # We'll pass the entire input list to genai.embed_content
        # Note: Ensure input is not empty before calling genai.embed_content
        if not input:
            return []

        try:
            # Pass the list of texts directly to embed_content
            result = genai.embed_content(
                model=self._model_name,
                content=input, # Pass the list of documents here
                task_type="retrieval_document" # Or "retrieval_query" depending on context
            )
            # embed_content returns a dict with 'embedding' key, which is a list of embeddings
            embeddings = result['embedding']

        except Exception as e:
            print(f"Error generating embeddings for batch (first 100 chars): '{input[0][:100]}...': {e}")
            traceback.print_exc()
            # Use the expected dimension size (768 for embedding-001) for fallback
            fallback_vector_size = 768 # embedding-001 dimension
            # Add more checks here if you use different embedding models
            print(f"Using fallback vectors of size {fallback_vector_size} for the batch.")
            # Return a list of zero vectors matching the size of the input batch
            embeddings = [[0.0] * fallback_vector_size for _ in input]

        # Ensure the number of returned embeddings matches the number of input documents
        if len(embeddings) != len(input):
             print(f"Warning: Mismatch in embedding count. Expected {len(input)}, got {len(embeddings)}.")
             # Fallback to returning zero vectors if embedding failed for some reason
             fallback_vector_size = 768
             embeddings = [[0.0] * fallback_vector_size for _ in input]


        return embeddings

# --- Collection Management ---
collection_name = "my_pdf_collection"
# Initialize the embedding function
gemini_ef = GeminiEmbeddingFunction()

# Clean up the collection each time the app starts for this example
# In a real application, you would manage this persistence carefully
try:
    # Check if collection exists before trying to delete
    existing_collections = client.list_collections()
    if any(col.name == collection_name for col in existing_collections):
         client.delete_collection(name=collection_name)
         print(f"Collection '{collection_name}' deleted successfully.")
    else:
         print(f"Collection '{collection_name}' does not exist - will create new")

except Exception as e:
    print(f"Error during collection deletion attempt: {e}")
    # Continue, as the goal is just to ensure a fresh start

# Re-create the collection
try:
    collection = client.create_collection(
        name=collection_name,
        embedding_function=gemini_ef # Assign the custom embedding function
    )
    print(f"Collection '{collection_name}' created successfully.")
except Exception as e:
    print(f"Error creating collection '{collection_name}': {e}")
    traceback.print_exc()
    # If collection creation fails, the app cannot function correctly for chat
    collection = None # Set collection to None if creation failed


# --- PDF Processing ---
# Only process PDFs if collection was created successfully
if collection is not None:
    print(f"Loading PDFs from: {pdf_directory}")
    doc_id_counter = 0
    pdf_files_found = list(pdf_directory.glob("*.pdf"))

    if not pdf_files_found:
        print(f"⚠️ No PDF files found in {pdf_directory}")
    else:
        print(f"Found {len(pdf_files_found)} PDF file(s). Processing...")
        for pdf_file in pdf_files_found:
            print(f"Processing {pdf_file.name}...")
            try:
                loader = PyPDFLoader(str(pdf_file))
                # Using load_and_split() which handles splitting into Document objects
                pages = loader.load_and_split()

                if not pages:
                    print(f"⚠️ No content extracted from {pdf_file.name}")
                    continue

                # Batching for adding documents to ChromaDB
                # A reasonable batch size balances memory usage and add performance
                batch_size = 100
                for i in range(0, len(pages), batch_size):
                    batch = pages[i:i+batch_size]

                    # Prepare documents, metadatas, and ids for the batch
                    batch_documents = [item.page_content for item in batch]
                    batch_metadatas = []
                    batch_ids = []
                    for idx, item in enumerate(batch):
                         # Ensure metadata is a dict and page number is handled
                         metadata = item.metadata if isinstance(item.metadata, dict) else {}
                         # Langchain's PyPDFLoader adds 'page' key (0-indexed)
                         page_num = metadata.get('page')
                         if page_num is not None:
                             # Convert to 1-indexed page number for users
                             page_num += 1
                         else:
                             # Fallback if 'page' is not in metadata
                             page_num = f"unknown_{i+idx+1}" # Use batch index + 1 as fallback

                         batch_metadatas.append({
                            "page": page_num,
                            "source": pdf_file.name
                         })
                         # Ensure IDs are unique string identifiers
                         batch_ids.append(f"{pdf_file.stem}_{doc_id_counter + i + idx}") # Use overall counter for uniqueness

                    # Add the batch to the collection
                    # ChromaDB add method expects lists for documents, metadatas, and ids
                    if batch_documents: # Ensure batch is not empty
                        try:
                             collection.add(
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                             print(f"-> Added batch of {len(batch_documents)} from {pdf_file.name} (IDs {batch_ids[0]} to {batch_ids[-1]})")
                        except Exception as add_error:
                             print(f"❌ Error adding batch from {pdf_file.name}: {add_error}")
                             traceback.print_exc()


                doc_id_counter += len(pages) # Increment total counter after processing file
                print(f"✅ Finished processing and adding pages from {pdf_file.name}")

            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
                traceback.print_exc() # Print detailed traceback for processing errors

        print("Finished processing PDF files.")
        print(f"Total documents in collection: {collection.count()}")
else:
     print("Skipping PDF processing as collection creation failed.")


# --- UI Configuration ---
categories = {
    'all': 'All Topics',
    'education': 'Education Planning',
    'career': 'Career Development',
    'skills': 'Skills & Training',
    'finance': 'Financial Aid'
}

# Simple in-memory storage for chat sessions.
# Use Redis, a database, or server-side sessions for production persistence.
chat_sessions = {}

# --- Routes ---
@app.route('/')
def home():
    # Pass a flag indicating if the collection is ready (PDFs loaded)
    # This allows the frontend to enable the chat interface
    is_ready = collection is not None and collection.count() > 0
    return render_template('index.html', categories=categories, is_ready=is_ready)

@app.route('/api/suggestions')
def get_suggestions():
    category = request.args.get('category', 'all')
    suggestions = {
        'all': [
            "What are some good study techniques?",
            "How to choose the right career?",
            "Can you tell me about scholarship opportunities?"
        ],
        'education': [
            "How to improve my study habits?",
            "What degrees lead to high-paying jobs?",
            "Should I consider graduate school?"
        ],
        'career': [
            "How to write an effective resume?",
            "Tips for job interviews?",
            "How to negotiate a salary?"
        ],
        'skills': [
            "What programming languages should I learn?",
            "How to develop leadership skills?",
            "Online learning platforms recommendation?"
        ],
        'finance': [
            "How to apply for financial aid?",
            "FAFSA application tips?",
            "Scholarships for first-generation students?"
        ]
    }
    return jsonify({'suggestions': suggestions.get(category, suggestions['all'])})

@app.route('/api/chat', methods=['POST'])
def chat():
    # Check if the collection is ready before processing chats
    if collection is None or collection.count() == 0:
         print("Chat requested but database is not ready.")
         return jsonify({'error': True, 'response': 'Database is not ready. Please wait for documents to load or check server logs for errors during processing.'}), 503

    try:
        data = request.json
        if not data:
            return jsonify({'error': True, 'response': 'Empty request'}), 400

        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session')
        category = data.get('category', 'all')

        if not message:
            return jsonify({'error': True, 'response': 'Empty message'}), 400

        # Initialize session if not exists
        if session_id not in chat_sessions:
            # Ensure history format is correct for genai.start_chat
            # [{"role": "user" | "model", "parts": [content_objects]}]
            chat_sessions[session_id] = []

        # Build context from history
        # Limit history length to avoid exceeding context windows and token limits
        # Keep history in the correct format for genai.start_chat
        history = chat_sessions[session_id][-10:] # Limit history to last 10 turns (user+model)

        # --- RAG: Retrieve relevant documents based on user query ---
        retrieved_context = "No relevant documents found."
        try:
            # FIX: Manually embed the query text using the same embedding function
            # used for documents, then query ChromaDB using the embeddings.
            # This bypasses the apparent bug in ChromaDB's text query validation.
            query_embedding_result = gemini_ef([message]) # Pass message in a list to __call__
            query_embedding = query_embedding_result[0] # Get the first (and only) embedding

            # Now query the collection using the embedding
            results = collection.query(
                query_embeddings=[query_embedding], # Pass the embedding in a list
                n_results=5 # Get top 5 results for better context
            )

            # Extract text from results. ChromaDB returns results['documents'] as a list of lists
            # The outer list corresponds to the query (we have one query here).
            # The inner list contains the document contents for the top results.
            if results and results.get('documents') and results['documents'][0]:
                retrieved_context = "\n\n".join(results['documents'][0])
            else:
                 retrieved_context = "No relevant documents found."

        except Exception as e:
            print(f"Error during document retrieval: {e}")
            traceback.print_exc()
            # Continue with "No context found" message if retrieval fails
            retrieved_context = "Error retrieving documents."


        # --- Generate Response using the LLM ---
        try:
            # Construct the final prompt including context and query
            # system_instruction is preferable but not available in this SDK version with chat history
            # So we format the prompt manually
            final_prompt = f"""You are EduCareer Guide, focused on {category}. Provide helpful and informative answers based *strictly* on the provided RELEVANT INFORMATION. If the information does not contain the answer to the user's query, state clearly that you cannot answer based on the documents.

RELEVANT INFORMATION:
{retrieved_context}

User Query: {message}
Answer:
"""

            # Load Gemini model
            # Verify 'gemini-2.0-flash' is supported in your google-generativeai==0.4.0 version.
            # If you get model-related errors, try 'gemini-1.5-flash' or 'gemini-1.0-pro'.
            model = genai.GenerativeModel(model_name='gemini-2.0-flash')

            # Start chat with previous history
            # history must be in the correct format [{"role": ..., "parts": [...]}, ...]
            chat_session = model.start_chat(history=history)

            # Send the final prompt containing system instructions and context as the next user turn
            # The content needs to be in a list of parts
            response = chat_session.send_message([final_prompt])

            # Append user message and assistant response to session history
            # Ensure the appended messages are also in the correct format
            # [{"role": ..., "parts": [...]}, ...]
            chat_sessions[session_id].append({"role": "user", "parts": [message]})
            # Make sure the model response parts are also a list of strings
            chat_sessions[session_id].append({"role": "model", "parts": [response.text]})

            return jsonify({'response': response.text})

        except Exception as e:
            print(f"Model error during chat generation: {e}")
            traceback.print_exc() # Print detailed traceback for model errors
            # Provide a user-friendly error message
            return jsonify({'error': True, 'response': 'AI service is currently unavailable or encountered an error generating a response.'}), 503

    except Exception as e:
        print(f"Server error during chat processing: {e}")
        traceback.print_exc() # Print detailed traceback for general server errors
        return jsonify({'error': True, 'response': 'Internal server error processing your request.'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clears the chat history for a given session ID."""
    try:
        session_id = request.json.get('session_id')
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            print(f"Cleared chat session: {session_id}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing chat session: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # For production, use a production WSGI server like Waitress or Gunicorn:
    # waitress-serve --listen=*:5001 app:app
    # gunicorn --bind 0.0.0.0:5001 app:app
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Starting Flask development server on port {port}")
    print("⚠️ WARNING: Using Flask's built-in development server. Not recommended for production.")
    # debug=True should only be used during development
    app.run(host='0.0.0.0', port=port, debug=True)