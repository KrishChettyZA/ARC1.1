# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader # For loading PDFs
import os # For interacting with the operating system (environment variables, paths)
from pathlib import Path # For handling file paths
from flask import Flask, request, render_template, session, jsonify # Flask framework components
from dotenv import load_dotenv # For loading environment variables from a .env file
import chromadb # Vector database client
import markdown2 # For converting Markdown to HTML (optional, if needed for display)
import google.generativeai as genai # Google Generative AI SDK

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Configuration ---
# Configure the Google Generative AI SDK with the API key
# IMPORTANT: Set the GOOGLE_API_KEY environment variable in your Render service settings.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("Google Generative AI SDK configured successfully.")
except ValueError as e:
    print(f"Error configuring Google Generative AI SDK: {e}")
    # Handle the error appropriately - maybe exit or disable AI features
    # For now, we'll print the error and continue, but API calls will fail.
except Exception as e:
    print(f"An unexpected error occurred during genai configuration: {e}")


# --- Flask App Initialization ---
app = Flask(__name__)
# Set a secret key for Flask session management. Use environment variable or generate a random one.
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- ChromaDB Setup ---
# Initialize ChromaDB client (consider using a persistent client for production)
# client = chromadb.PersistentClient(path="/path/to/persist/data") # Example for persistence
client = chromadb.Client() # In-memory client (data lost on restart)

# Directory where PDF files are stored relative to this script
pdf_directory = Path(__file__).parent / "documents"
# After defining pdf_directory
print(f"ABSOLUTE PATH FOR DOCUMENTS: {pdf_directory.resolve()}")
# Ensure the documents directory exists
os.makedirs(pdf_directory, exist_ok=True)

# --- Custom Embedding Function for ChromaDB ---
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom embedding function using Google's Gemini API (models/embedding-001)
    that adheres to ChromaDB's expected interface (post v0.4.16).
    """
    def __init__(self, api_key: str | None = None, model_name: str = "models/embedding-001", task_type: str = "retrieval_document"):
        # If an API key is provided here, configure genai specifically for this instance
        if api_key:
            genai.configure(api_key=api_key)
        self._model_name = model_name
        self._task_type = task_type
        print(f"GeminiEmbeddingFunction initialized with model: {self._model_name}, task_type: {self._task_type}")

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """
        Generates embeddings for a list of texts.
        Args:
            input: A list of strings (documents) to generate embeddings for.
        Returns:
            A list of embeddings (lists of floats), one for each input text.
        """
        embeddings: chromadb.Embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model=self._model_name,
                    content=text,
                    task_type=self._task_type
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error generating embedding for text snippet starting with '{text[:50]}...': {e}")
                embeddings.append([0.0] * 768)  # Fallback zero vector
        return embeddings
    
# Instantiate the embedding function
# It will use the API key configured globally via genai.configure() earlier
gemini_ef = GeminiEmbeddingFunction()

# --- ChromaDB Collection Management ---
collection_name = "my_pdf_collection" # Use a descriptive name

# WARNING: This setup deletes and recreates the collection on every app start.
# This is suitable for development but NOT for production as all data is lost on restart.
# In production, you'd typically check if the collection exists and only create/populate if needed,
# or use a persistent ChromaDB client.

try:
    print(f"Attempting to delete existing collection '{collection_name}'...")
    client.delete_collection(name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except Exception as e:
    # This is expected if the collection doesn't exist on the first run
    print(f"Collection '{collection_name}' does not exist or could not be deleted (this might be normal on first run): {e}")

print(f"Creating collection '{collection_name}'...")
# Create the collection with the custom Gemini embedding function
collection = client.create_collection(name=collection_name, embedding_function=gemini_ef)
print(f"Collection '{collection_name}' created successfully.")


# --- PDF Loading and Indexing ---
# Load PDF documents and add them to the ChromaDB collection
print(f"Loading PDFs from: {pdf_directory}")
doc_id_counter = 0
pdf_files_found = list(pdf_directory.glob("*.pdf"))

if not pdf_files_found:
    print(f"Warning: No PDF files found in {pdf_directory}. The knowledge base will be empty.")
else:
    print(f"Found {len(pdf_files_found)} PDF file(s). Processing...")
    for pdf_file in pdf_files_found:
        print(f"Processing {pdf_file.name}...")
        try:
            # Load PDF using PyPDFLoader
            pdf_loader = PyPDFLoader(str(pdf_file))
            # Split the document into pages (or smaller chunks if needed)
            pages = pdf_loader.load_and_split() # Consider chunking strategies for large pages

            if not pages:
                print(f"Warning: No content extracted from {pdf_file.name}.")
                continue

            # Prepare data for batch insertion into ChromaDB
            documents_to_add = []
            metadatas_to_add = []
            ids_to_add = []

            for page in pages:
                # Extract text content (ensure it's a string)
                page_text = str(page.page_content)
                # Basic check for empty pages
                if not page_text.strip():
                    print(f"Skipping empty page in {pdf_file.name}")
                    continue

                current_id = f"{pdf_file.stem}_{doc_id_counter}" # Create a more robust ID
                documents_to_add.append(page_text)
                # Include page number from metadata if available, otherwise use counter
                page_num = page.metadata.get('page', doc_id_counter) # PyPDFLoader adds 'page' metadata
                metadatas_to_add.append({"page_number": page_num, "source": pdf_file.name})
                ids_to_add.append(current_id)
                doc_id_counter += 1

            # Add documents to the collection in batches (more efficient)
            if documents_to_add:
                collection.add(
                    documents=documents_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                print(f"Added {len(documents_to_add)} pages/chunks from {pdf_file.name} to the collection.")

        except FileNotFoundError:
             print(f"Error: PDF file not found at {pdf_file}. Skipping.")
        except Exception as e:
            # Catch other potential errors during PDF processing or ChromaDB adding
            print(f"Error processing {pdf_file.name}: {e}")
            # Consider logging the traceback for debugging: import traceback; traceback.print_exc()

print("Finished processing PDF files.")


# --- UI Data ---
# Categories for the UI dropdown/selection
categories = {
    'all': 'All Topics',
    'education': 'Education Planning',
    'career': 'Career Development',
    'skills': 'Skills & Training',
    'finance': 'Financial Aid'
}

# Example suggestions for the chat interface based on category
suggestions = {
    'all': [
        "What are some good study techniques?",
        "How to choose the right career?",
        "Can you tell me about scholarship opportunities?",
        "What skills are in demand right now?"
    ],
    'education': [
        "How to improve my study habits?",
        "What degrees lead to high-paying jobs?",
        "Should I consider graduate school?",
        "How to prepare for standardized tests?"
    ],
    'career': [
        "How to write an effective resume?",
        "Tips for job interviews?",
        "How to negotiate a salary?",
        "Career paths in technology?"
    ],
    'skills': [
        "What programming languages should I learn?",
        "How to develop leadership skills?",
        "Soft skills needed in today's workplace?",
        "Online learning platforms recommendation?"
    ],
    'finance': [
        "How to apply for financial aid?",
        "FAFSA application tips?",
        "Scholarships for first-generation students?",
        "Student loan repayment strategies?"
    ]
}

# --- In-Memory Chat Session Storage ---
# WARNING: Stores chat history in memory. History is lost if the app restarts.
# For production, consider using a database (like Redis, PostgreSQL) or a file-based session store.
chat_sessions = {}

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main chat page."""
    # Pass categories to the template for the UI
    return render_template('index.html', categories=categories)

@app.route('/api/suggestions')
def get_suggestions():
    """API endpoint to get suggestions based on the selected category."""
    category = request.args.get('category', 'all') # Default to 'all' if no category specified
    category_suggestions = suggestions.get(category, suggestions['all']) # Fallback to 'all' suggestions
    return jsonify({'suggestions': category_suggestions})

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint to handle incoming chat messages and generate responses."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': True, 'response': 'Invalid request: No JSON data received.'}), 400

        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session') # Use a default or generate unique IDs
        category = data.get('category', 'all')

        if not message:
            return jsonify({'error': True, 'response': 'Message cannot be empty.'}), 400

        # Initialize session history if it's a new session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Add user message to the session history
        chat_sessions[session_id].append({'role': 'user', 'parts': [message]}) # Using 'parts' as per genai format

        # --- RAG Implementation ---
        # 1. Query ChromaDB to find relevant documents based on the user message
        try:
            query_results = collection.query(
                query_texts=[message],
                n_results=3 # Retrieve top 3 relevant document chunks
                # You might add 'where' clauses here for filtering if needed
            )
            retrieved_docs = query_results.get('documents', [[]])[0] # Get the list of document texts
            context_from_docs = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
            print(f"Retrieved {len(retrieved_docs)} documents for context.")

        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            context_from_docs = "Error retrieving relevant documents." # Inform the LLM about the retrieval issue


        # --- Prepare Prompt for Generative Model ---
        # 2. Construct the prompt including history, context, and instructions

        # Define category-specific instructions
        category_focus = {
            'education': "focusing on education planning and academic advice",
            'career': "focusing on career development and job search guidance",
            'skills': "focusing on skills development and training resources",
            'finance': "focusing on financial aid and scholarship opportunities"
        }
        category_instruction = category_focus.get(category, "covering all topics") # Default instruction

        # Build the history part for the model (ensure correct format)
        # The history should alternate between 'user' and 'model' roles
        model_history = []
        for exchange in chat_sessions[session_id]:
             model_history.append(exchange)
             # We need the previous model response if it exists to complete the history
             # Note: The current user message is already added above.
             # If the last entry has a 'response' key, add it as the model's turn
             # This logic assumes 'response' is added *after* the model generates it.
             # Let's adjust how history is built for the `start_chat` method.

        # Prepare history EXCLUDING the latest user message (which will be the final prompt part)
        history_for_model = chat_sessions[session_id][:-1] # All except the last user message

        # System prompt / Initial instruction
        # Note: Gemini API prefers multi-turn history over a single large prompt string sometimes.
        # We will use the `start_chat` method for better conversation management.

        system_prompt = f"""You are EduCareer Guide, a helpful and friendly AI assistant {category_instruction}.
Use the provided RELEVANT INFORMATION to answer the user's query accurately.
If the information isn't sufficient, state that you couldn't find specific details in your knowledge base but provide general advice.
Keep your responses concise, informative, and encouraging. Format key points or lists clearly using Markdown.

RELEVANT INFORMATION:
---
{context_from_docs}
---
"""

        # --- Generate Response using Gemini ---
        # 3. Call the Generative Model
        try:
            # Use a valid Gemini model name (e.g., gemini-1.5-flash-latest)
            # model = genai.GenerativeModel('gemini-1.5-flash-latest')
            # For conversational context, it's better to use start_chat
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                # system_instruction=system_prompt # System instructions are better handled this way
            )

            # Start or continue the chat session with history
            chat_session = model.start_chat(history=history_for_model)

            # Send the latest user message combined with the system prompt and context
            # We inject the system prompt and context conceptually here for the model's current turn.
            # A cleaner way might be to adjust the system prompt passed to GenerativeModel if supported directly,
            # or structure the history carefully. Let's try adding context to the user message.

            prompt_with_context = f"{system_prompt}\n\nUser Query: {message}"

            # Send the message to the model via the chat session
            response = chat_session.send_message(prompt_with_context) # Send the user message

            bot_response_text = response.text

        except Exception as e:
            print(f"Error generating response from Gemini: {e}")
            # import traceback
            # traceback.print_exc() # Print detailed traceback for debugging
            return jsonify({'error': True, 'response': f"Sorry, I encountered an error while generating a response. Please try again. Error: {str(e)}"}), 500

        # --- Update Session History ---
        # Add the bot's response to the last exchange in the session history
        # Ensure the format matches what `start_chat` expects ('role': 'model', 'parts': [text])
        chat_sessions[session_id].append({'role': 'model', 'parts': [bot_response_text]})

        # Return the bot's response
        return jsonify({'response': bot_response_text})

    except Exception as e:
        # Catch-all for any other unexpected errors in the route
        print(f"Unexpected error in /api/chat: {e}")
        # import traceback
        # traceback.print_exc()
        return jsonify({'error': True, 'response': f"An internal server error occurred. Please try again later. Error: {str(e)}"}), 500


@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """API endpoint to clear the chat history for a given session."""
    data = request.json
    session_id = data.get('session_id', 'default_session')

    # Clear the chat history for this session if it exists
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
        print(f"Chat history cleared for session_id: {session_id}")
        return jsonify({'success': True, 'message': 'Chat history cleared.'})
    else:
        print(f"Attempted to clear non-existent session_id: {session_id}")
        return jsonify({'success': False, 'message': 'Session not found.'}), 404

# --- Main Execution ---
if __name__ == '__main__':
    # Get port from environment variable (Render sets this) or default to 5001 for local dev
    port = int(os.environ.get('PORT', 5001))
    # Run the Flask app
    # debug=True is useful for development but should be False in production
    # host='0.0.0.0' makes the server accessible externally (needed for Render)
    print(f"Starting Flask server on host 0.0.0.0, port {port}")
    app.run(debug=False, host='0.0.0.0', port=port) # Set debug=False for production
