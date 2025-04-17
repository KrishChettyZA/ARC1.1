document.addEventListener('DOMContentLoaded', function() {
    const chatBody = document.getElementById('chat-body');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-message');
    const categoryButtons = document.querySelectorAll('.category-btn');
    const suggestionsContainer = document.getElementById('suggestions-container');
    const currentFocus = document.getElementById('current-focus');
    const clearChatButton = document.getElementById('clear-chat');
    
    // Generate a unique session ID
    const sessionId = 'session_' + Date.now();
    let currentCategory = 'all';
    
    // Function to show loading indicator
    function showLoading() {
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatBody.appendChild(loadingIndicator);
        chatBody.scrollTop = chatBody.scrollHeight;
        return loadingIndicator;
    }
    
    // Function to remove loading indicator
    function removeLoading(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }
    
    // Function to add a new message to the chat
    function addMessage(text, isUser = false) {
        const messageElement = document.createElement('div');
        messageElement.className = isUser ? 'user-message message' : 'bot-message message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = text;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = 'Just now';
        
        messageElement.appendChild(contentDiv);
        messageElement.appendChild(timeDiv);
        
        chatBody.appendChild(messageElement);
        chatBody.scrollTop = chatBody.scrollHeight;
        return messageElement;
    }
    
    // Function to show error message
    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        chatBody.appendChild(errorElement);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    // Function to get suggestions based on category
    async function loadSuggestions(category) {
        try {
            const response = await fetch(`/api/suggestions?category=${category}`);
            const data = await response.json();
            
            suggestionsContainer.innerHTML = '';
            
            data.suggestions.forEach(suggestion => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.textContent = suggestion;
                chip.addEventListener('click', function() {
                    userInput.value = this.textContent;
                    sendMessage();
                });
                suggestionsContainer.appendChild(chip);
            });
        } catch (error) {
            console.error('Error loading suggestions:', error);
        }
    }
    
    // Function to handle sending a message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show loading indicator
        const loadingIndicator = showLoading();
        
        try {
            // Call the API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId,
                    category: currentCategory
                }),
            });
            
            const data = await response.json();
            
            // Remove loading indicator
            removeLoading(loadingIndicator);
            
            if (data.error) {
                showError(data.response || 'An error occurred while processing your request.');
            } else {
                // Add bot response to chat
                addMessage(data.response);
            }
        } catch (error) {
            // Remove loading indicator
            removeLoading(loadingIndicator);
            
            // Show error message
            showError("Sorry, I couldn't process your request. Please try again later.");
            console.error('Error:', error);
        }
    }
    
    // Function to clear chat
    async function clearChat() {
        try {
            await fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId
                }),
            });
            
            // Clear chat UI except for initial greeting
            chatBody.innerHTML = '';
            
            // Add initial greeting
            addMessage('Hello! I\'m your EduCareer Guide assistant. How can I help you with your education or career journey today?');
            
            // Load suggestions
            loadSuggestions(currentCategory);
            
        } catch (error) {
            showError("Couldn't clear the conversation. Please try again.");
            console.error('Error clearing chat:', error);
        }
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    
    // Clear chat button
    clearChatButton.addEventListener('click', clearChat);
    
    // Category button event listeners
    categoryButtons.forEach(button => {
        button.addEventListener('click', function() {
            const newCategory = this.dataset.category;
            
            // Skip if already on this category
            if (newCategory === currentCategory) return;
            
            // Update current category
            currentCategory = newCategory;
            
            // Remove active class from all buttons
            categoryButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Update current focus text
            currentFocus.textContent = this.textContent;
            
            // Load new suggestions
            loadSuggestions(currentCategory);
            
            // Add system message about category change
            addMessage(`Now focusing on: ${this.textContent}. How can I help you with this area?`);
        });
    });
    
    // Resource links
    document.getElementById('studyGuide').addEventListener('click', function(e) {
        e.preventDefault();
        userInput.value = "Can you share some effective study techniques?";
        sendMessage();
    });
    
    document.getElementById('careerAssessment').addEventListener('click', function(e) {
        e.preventDefault();
        userInput.value = "What career assessment tools do you recommend?";
        sendMessage();
    });
    
    document.getElementById('scholarships').addEventListener('click', function(e) {
        e.preventDefault();
        userInput.value = "How can I find scholarships for my education?";
        sendMessage();
    });
    
    document.getElementById('resumeBuilder').addEventListener('click', function(e) {
        e.preventDefault();
        userInput.value = "What should I include in my resume?";
        sendMessage();
    });
    
    // Load initial suggestions
    loadSuggestions('all');
});
