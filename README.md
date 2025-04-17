# Educational and Career Guidance Chatbot with ChromaDB and RAG

## Overview
This application is an AI-powered educational and career guidance chatbot that uses Retrieval Augmented Generation (RAG) with ChromaDB and Google's Gemini API. The chatbot provides personalized guidance on education planning, career development, skills training, and financial aid by retrieving relevant information from a knowledge base of PDF documents.

## Features
- Interactive web-based chat interface
- Category-based guidance (Education, Career, Skills, Finance)
- Context-aware responses using RAG technology
- Document retrieval from PDF knowledge base
- Conversation history tracking
- Suggested questions for each category
- Responsive design for desktop and mobile devices

## Technical Architecture

### Components
1. **Flask Web Application**: Provides the backend server and API endpoints
2. **ChromaDB**: Vector database for storing and retrieving document embeddings
3. **Google Gemini API**: Generates embeddings and AI responses
4. **RAG Implementation**: Retrieves relevant context from documents to enhance AI responses
5. **Frontend**: HTML/CSS/JavaScript for the user interface

### Data Flow
1. User sends a question through the web interface
2. The question is embedded and used to query ChromaDB for relevant documents
3. Retrieved documents are combined with the conversation history
4. This context is sent to the Gemini API to generate a response
5. The response is returned to the user and stored in the conversation history

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Google API key for Gemini

### Dependencies
The application requires the following Python packages:
```
flask==2.3.3
python-dotenv==1.0.0
chromadb==0.4.18
langchain-community==0.0.13
google-generativeai==0.3.1
markdown2==2.4.10
langchain
```

### Setup Instructions
1. Clone the repository or download the source code
2. Navigate to the project directory
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_here
   ```
5. Ensure you have PDF documents in the `documents` directory
6. Run the application:
   ```
   python app.py
   ```
7. Access the application at `http://localhost:5001`

## Usage Guide

### Adding Knowledge Documents
Place PDF documents in the `documents` directory. The application will automatically load and index these documents when it starts.

### Interacting with the Chatbot
1. Select a category from the sidebar to focus on a specific area
2. Type your question in the input field and press Enter or click the send button
3. The chatbot will respond with relevant information based on the documents in the knowledge base
4. You can click on suggested questions to quickly ask common questions
5. Use the "Clear Chat" button to start a new conversation

### Categories
- **All Topics**: General guidance on all educational and career topics
- **Education Planning**: Academic advice, study techniques, degree planning
- **Career Development**: Resume writing, interviews, job search, career paths
- **Skills & Training**: In-demand skills, learning resources, certifications
- **Financial Aid**: Scholarships, student loans, FAFSA, financial literacy

## Project Structure
```
educareer_guide/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── create_sample_pdfs.py   # Script to create sample PDF documents
├── documents/              # Directory for PDF knowledge base
│   ├── study_techniques.pdf
│   ├── career_guide.pdf
│   ├── financial_aid.pdf
│   └── skills_development.pdf
├── static/                 # Static assets
│   ├── styles.css          # CSS styles
│   └── script.js           # JavaScript for frontend
└── templates/              # HTML templates
    └── index.html          # Main application template
```

## Customization

### Adding New Categories
To add new categories, modify the `categories` and `suggestions` dictionaries in `app.py`.

### Changing the UI
Modify the CSS in `static/styles.css` and HTML in `templates/index.html` to customize the appearance.

### Extending Functionality
- Add authentication for personalized experiences
- Implement document upload through the UI
- Add analytics to track common questions
- Integrate with external APIs for real-time data

## Troubleshooting
- If you encounter errors with ChromaDB, ensure you have the correct version installed
- For Gemini API errors, verify your API key is correct and has sufficient quota
- If PDFs aren't loading, check file permissions and format compatibility

## License
This project is licensed under the MIT License - see the LICENSE file for details.
