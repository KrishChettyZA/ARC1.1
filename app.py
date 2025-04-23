from langchain_community.document_loaders import PyPDFLoader
import os
from pathlib import Path
from flask import Flask, request, render_template, session, jsonify
from dotenv import load_dotenv
import chromadb
import markdown2
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Use the environment variable for secret key if available, otherwise generate a random one
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))


# Initialize ChromaDB client
client = chromadb.Client()

# Directory where PDF files are stored
pdf_directory = Path(__file__).parent / "documents"

# Create a custom embedding function for Gemini that follows ChromaDB's interface
class GeminiEmbeddingFunction:
    def __init__(self):
        pass
        
    def __call__(self, input):
        """
        Generate embeddings for a list of texts using Gemini API.
        
        Args:
            input: A list of strings to generate embeddings for
            
        Returns:
            A list of embeddings, one for each text
        """
        embeddings = []
        for text in input:
            try:
                # Use Gemini's embedding capability
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding['embedding'])
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Provide a fallback empty embedding
                embeddings.append([0.0] * 768)  # Typical embedding size
        return embeddings

gemini_ef = GeminiEmbeddingFunction()

# Create and populate collection only once when the application starts
collection_name = "my_collection"
try:
    client.delete_collection(name=collection_name)
    print(f"Existing collection '{collection_name}' deleted.")
except Exception as e:
    print(f"Collection '{collection_name}' does not exist or could not be deleted: {e}")

# Create the collection with the embedding function
collection = client.create_collection(name=collection_name, embedding_function=gemini_ef)

# Load PDF documents and add them to the collection
doc_id = 0
for pdf_file in pdf_directory.glob("*.pdf"):
    try:
        pdf_loader = PyPDFLoader(str(pdf_file))
        pages = pdf_loader.load_and_split()

        for page in pages:
            text = str(page.page_content)
            collection.add(documents=[text], metadatas=[{"page_number": doc_id, "source": pdf_file.name}], ids=[str(doc_id)])
            doc_id += 1
        print(f"Added {len(pages)} pages from {pdf_file.name}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

# Categories for the UI
categories = {
    'all': 'All Topics',
    'education': 'Education Planning',
    'career': 'Career Development',
    'skills': 'Skills & Training',
    'finance': 'Financial Aid'
}

# Suggestions for different categories
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

# Store chat sessions
chat_sessions = {}

@app.route('/')
def home():
    return render_template('index.html', categories=categories)

@app.route('/api/suggestions')
def get_suggestions():
    category = request.args.get('category', 'all')
    category_suggestions = suggestions.get(category, suggestions['all'])
    return jsonify({'suggestions': category_suggestions})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default_session')
    category = data.get('category', 'all')
    
    # Initialize session if it doesn't exist
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message to session history
    chat_sessions[session_id].append({'role': 'user', 'content': message})
    
    try:
        # Query ChromaDB based on the user's question
        query_results = collection.query(query_texts=[message], n_results=5)
        documents = query_results.get('documents', [[]])[0]

        # Construct the conversation history for the prompt
        history_text = "\n".join(f"User: {exchange['content']}\nAssistant: {exchange.get('response', '')}" 
                              for exchange in chat_sessions[session_id] if 'content' in exchange)

        # Include the most relevant document text in the prompt
        context_from_docs = "\n\n".join(documents[:3]) if documents else ""
        
        # Adjust the prompt based on category
        category_focus = {
            'education': "with a focus on education planning and academic advice",
            'career': "with a focus on career development and job search guidance",
            'skills': "with a focus on skills development and training resources",
            'finance': "with a focus on financial aid and scholarship opportunities"
        }
        
        category_instruction = category_focus.get(category, "")
        
        # Construct the prompt - avoid using system role
        prompt_text = f"""You are an educational and career guidance AI assistant {category_instruction}. Provide helpful, accurate information and advice based on the user's needs.

Previous conversation:
{history_text}

User query: {message}

Relevant information:
{context_from_docs}

Respond in a helpful, encouraging manner. Focus on providing practical advice and actionable steps.
"""

        # Use Gemini to generate a response
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Fixed: Use simple content generation without system role
        response = model.generate_content(prompt_text)
        
        bot_response = response.text
        
        # Add bot response to session history
        chat_sessions[session_id][-1]['response'] = bot_response
        
        return jsonify({'response': bot_response})
    except Exception as e:
        print(f"Error getting answer: {e}")
        return jsonify({'error': True, 'response': f"I'm sorry, I encountered an error. Please try again. Error: {str(e)}"})

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id', 'default_session')
    
    # Clear the chat history for this session
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    
    return jsonify({'success': True})

if __name__ == '__main__':
    # Create documents directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
