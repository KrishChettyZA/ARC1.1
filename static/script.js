document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const sessionList = document.getElementById('sessionList');
    const referencesPanel = document.getElementById('referencesPanel');
    const referencesContent = document.getElementById('referencesContent');
    const citationCountSpan = document.getElementById('citationCount');

    let currentSessionId = null;
    let isLoading = false; // To prevent multiple sends
    // Removed currentCitationMap since we restart from 1 for each response

    // Function to generate a unique session ID
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Function to save current chat state to local storage (for robust session handling)
    function saveCurrentSessionId(sessionId) {
        localStorage.setItem('currentSessionId', sessionId);
    }

    function getSavedSessionId() {
        return localStorage.getItem('currentSessionId');
    }

    // Function to render a message in the chat
    function addMessage(sender, text, citations = [], shouldRenumberCitations = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', `${sender}-message`);

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble');

        let finalText = text;
        let finalCitations = citations;

        // Renumber citations if this is a new bot response (not loading from history)
        if (shouldRenumberCitations && sender === 'bot' && citations.length > 0) {
            const result = renumberCitationsForSession(citations, text);
            finalText = result.updatedMessageText;
            finalCitations = result.renumberedCitations;
        }

        // Basic Markdown parsing (for bold, italics, lists, headings, and citations)
        let formattedText = finalText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')             // Italics
            .replace(/^- (.*)/gm, '<li>$1</li>')              // Unordered list items
            .replace(/^(\d+)\. (.*)/gm, '<li>$1. $2</li>');   // Ordered list items
        
        // Wrap lists with <ul> or <ol>
        formattedText = formattedText.replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, (match) => {
            const lines = match.split('<li>').filter(line => line.trim() !== '');
            if (lines.every(line => /^\d+\./.test(line.trim()))) {
                return `<ol>${match}</ol>`;
            }
            return `<ul>${match}</ul>`;
        });

        // Basic heading parsing (e.g., ## Heading)
        formattedText = formattedText.replace(/^### (.*)/gm, '<h3>$1</h3>');
        formattedText = formattedText.replace(/^## (.*)/gm, '<h2>$1</h2>');
        formattedText = formattedText.replace(/^# (.*)/gm, '<h1>$1</h1>');

        // Enhanced citation processing with proper superscript formatting
        if (finalCitations && finalCitations.length > 0) {
            // Process both [^1] and [1] citation formats and convert to proper superscripts
            finalCitations.forEach(citation => {
                // Handle [^1] format (preferred backend format)
                const caretRegex = new RegExp(`\\[\\^${citation.id}\\]`, 'g');
                formattedText = formattedText.replace(caretRegex, 
                    `<sup class="citation-link" data-citation-id="${citation.id}">${citation.id}</sup>`);
                
                // Handle [1] format (alternative format)
                const bracketRegex = new RegExp(`\\[${citation.id}\\]`, 'g');
                formattedText = formattedText.replace(bracketRegex, 
                    `<sup class="citation-link" data-citation-id="${citation.id}">${citation.id}</sup>`);
            });
            
            // Also handle any remaining citation patterns that might not match exactly
            formattedText = formattedText.replace(/\[\^(\d+)\]/g, 
                (match, num) => {
                    // Check if this citation ID exists in our citations array
                    const citationExists = finalCitations.some(c => c.id == num);
                    if (citationExists) {
                        return `<sup class="citation-link" data-citation-id="${num}">${num}</sup>`;
                    }
                    return match; // Return unchanged if citation doesn't exist
                });
        }

        messageBubble.innerHTML = formattedText;
        messageElement.appendChild(messageBubble);
        
        // Store the final citations on the message element for later use
        messageElement.finalCitations = finalCitations;
        messageElement.finalText = finalText;
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to latest message
        
        return { finalCitations, finalText };
    }

    // Function to renumber citations for the current session - RESTART FROM 1 FOR EACH RESPONSE
    function renumberCitationsForSession(citations, messageText) {
        if (!citations || citations.length === 0) {
            return { renumberedCitations: [], updatedMessageText: messageText };
        }

        // Find which citations are actually referenced
        const referencedCitations = findReferencedCitations(messageText, citations);
        if (referencedCitations.length === 0) {
            return { renumberedCitations: [], updatedMessageText: messageText };
        }

        // Always start from 1 for each new response
        let newCitationNumber = 1;
        
        const renumberedCitations = [];
        let updatedMessageText = messageText;

        // Sort referenced citations by their original ID to maintain consistent order
        const sortedReferencedCitations = referencedCitations.sort((a, b) => parseInt(a.id) - parseInt(b.id));

        sortedReferencedCitations.forEach(citation => {
            const originalId = parseInt(citation.id);
            const newId = newCitationNumber;
            
            // Create new citation with renumbered ID
            const renumberedCitation = {
                ...citation,
                id: newId.toString()
            };
            renumberedCitations.push(renumberedCitation);

            // Update all occurrences in the message text
            const citationPatterns = [
                new RegExp(`\\[\\^${originalId}\\]`, 'g'),  // [^1] format
                new RegExp(`\\[${originalId}\\]`, 'g'),     // [1] format
                new RegExp(`\\^${originalId}(?!\\d)`, 'g')  // ^1 format (not followed by digit)
            ];

            citationPatterns.forEach(pattern => {
                updatedMessageText = updatedMessageText.replace(pattern, `[^${newId}]`);
            });

            newCitationNumber++;
        });

        return { renumberedCitations, updatedMessageText };
    }
    function findReferencedCitations(messageText, allCitations) {
        if (!messageText || !allCitations || allCitations.length === 0) {
            return [];
        }

        const referencedCitations = [];
        const citationIds = new Set();

        // Look for citation patterns in the text - more comprehensive search
        const citationPatterns = [
            /\[\^(\d+)\]/g,  // [^1] format
            /\[(\d+)\]/g,    // [1] format
            /\^(\d+)/g       // ^1 format (without brackets)
        ];

        citationPatterns.forEach(pattern => {
            // Reset regex lastIndex to ensure proper matching
            pattern.lastIndex = 0;
            let match;
            while ((match = pattern.exec(messageText)) !== null) {
                const citationId = parseInt(match[1]);
                if (!isNaN(citationId)) {
                    citationIds.add(citationId);
                }
            }
        });

        // Filter citations to only include those that are actually referenced
        allCitations.forEach(citation => {
            const citationIdNum = parseInt(citation.id);
            if (citationIds.has(citationIdNum)) {
                referencedCitations.push(citation);
            }
        });

        // Sort by citation ID to maintain order
        return referencedCitations.sort((a, b) => parseInt(a.id) - parseInt(b.id));
    }

    // Function to display citations in the references panel - ONLY cited ones
    function displayCitations(citations, messageText = '') {
        referencesContent.innerHTML = ''; // Clear previous citations

        // Only show citations that are actually referenced in the message
        const referencedCitations = messageText ? 
            findReferencedCitations(messageText, citations) : 
            [];

        citationCountSpan.textContent = `(${referencedCitations.length})`;

        if (referencedCitations.length === 0) {
            referencesContent.innerHTML = '<p class="no-references">Citations for responses will appear here.</p>';
            return;
        }

        referencedCitations.forEach(citation => {
            const citationItem = document.createElement('div');
            citationItem.classList.add('reference-item');
            citationItem.setAttribute('data-citation-id', citation.id);

            // Truncate content if too long for display
            const truncatedContent = citation.content.length > 300 
                ? citation.content.substring(0, 300) + '...' 
                : citation.content;

            citationItem.innerHTML = `
                <h3>
                    <span class="ref-id">${citation.id}</span> 
                    <span class="ref-source">${citation.source}</span>
                </h3>
                <p class="ref-page">${citation.page_number}</p>
                <div class="ref-content">${truncatedContent}</div>
                <a href="/api/pdf/${encodeURIComponent(citation.source)}" target="_blank" class="view-pdf-btn">
                    <i class="fas fa-file-pdf"></i> View PDF
                </a>
            `;
            referencesContent.appendChild(citationItem);
        });

        // Add click listeners to citation links in chat messages
        setTimeout(() => {
            document.querySelectorAll('.citation-link').forEach(link => {
                link.removeEventListener('click', handleCitationClick);
                link.addEventListener('click', handleCitationClick);
            });
        }, 100);
    }

    function handleCitationClick(event) {
        event.preventDefault();
        const citationId = event.target.dataset.citationId;
        const referenceItem = document.querySelector(`.references-panel .reference-item[data-citation-id="${citationId}"]`);
        if (referenceItem) {
            // Scroll to the reference item
            referencesContent.scrollTo({
                top: referenceItem.offsetTop - referencesContent.offsetTop,
                behavior: 'smooth'
            });
            // Briefly highlight the item
            referenceItem.style.transition = 'background-color 0.5s ease-in-out';
            referenceItem.style.backgroundColor = '#e0f7fa'; // Light blue highlight
            setTimeout(() => {
                referenceItem.style.backgroundColor = ''; // Revert to original
            }, 1500);
        }
    }

    // Function to show/hide loading indicator
    function showLoadingIndicator() {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('chat-message', 'bot-message', 'loading-indicator');
        loadingElement.innerHTML = `
            <div class="message-bubble">
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideLoadingIndicator() {
        const loadingElement = chatMessages.querySelector('.loading-indicator');
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    // Function to send message to backend
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '' || isLoading) {
            return;
        }

        isLoading = true;
        sendBtn.disabled = true;
        userInput.disabled = true;
        
        addMessage('user', message);
        userInput.value = ''; // Clear input field

        showLoadingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message, session_id: currentSessionId }),
            });

            const data = await response.json();

            hideLoadingIndicator();

            if (data.success) {
                const result = addMessage('bot', data.response, data.citations, true); // true = renumber citations
                displayCitations(result.finalCitations, result.finalText);
                // Update session preview in sidebar for the current session
                updateSessionPreview(currentSessionId, message);
            } else {
                addMessage('bot', data.response || 'An error occurred.');
                displayCitations([]); // Clear citations on error
            }
        } catch (error) {
            hideLoadingIndicator();
            console.error('Error:', error);
            addMessage('bot', 'Sorry, I am unable to connect to the server. Please try again later.');
            displayCitations([]); // Clear citations on network error
        } finally {
            isLoading = false;
            sendBtn.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent new line
            sendMessage();
        }
    });

    newChatBtn.addEventListener('click', startNewChat);

    // --- Session Management Functions ---

    function startNewChat() {
        currentSessionId = generateSessionId();
        saveCurrentSessionId(currentSessionId);
        
        chatMessages.innerHTML = `
            <div class="chat-message bot-message intro-message">
                <div class="message-bubble">
                    Hello! I am the **ARC Principal Career Strategist**, an expert on the Refracted Economies Framework. I'm here to provide detailed, comprehensive, and strategic career guidance. How can I assist you today?
                </div>
            </div>
        `;
        displayCitations([], ''); // Clear citations for a new chat
        userInput.value = '';
        listSessions(); // Refresh session list to show new chat
        updateActiveSessionUI();
        userInput.focus();
    }

    async function loadSession(sessionId) {
        if (isLoading) return; // Prevent switching while loading

        currentSessionId = sessionId;
        saveCurrentSessionId(currentSessionId);
        
        chatMessages.innerHTML = ''; // Clear current messages
        displayCitations([]); // Clear citations until loaded

        showLoadingIndicator(); // Show loading for history
        isLoading = true;
        sendBtn.disabled = true;
        userInput.disabled = true;

        try {
            const response = await fetch(`/api/history?session_id=${sessionId}`);
            const data = await response.json();

            hideLoadingIndicator();

            if (data.success && data.history) {
                let lastCitations = [];
                let lastResponseText = '';
                
                data.history.forEach(turn => {
                    // Ensure 'parts' and 'citations' exist for compatibility
                    const messageText = (turn.parts && turn.parts[0]) || '';
                    const messageCitations = turn.citations || [];
                    
                    if (turn.role === 'model' && messageCitations.length > 0) {
                        // Renumber citations when loading from history (restart from 1 for each response)
                        const result = addMessage(turn.role, messageText, messageCitations, true);
                        lastCitations = result.finalCitations;
                        lastResponseText = result.finalText;
                    } else {
                        addMessage(turn.role, messageText, messageCitations, false);
                        if (turn.role === 'model') {
                            lastCitations = messageCitations;
                            lastResponseText = messageText;
                        }
                    }
                });
                
                // Display only the cited citations from the last model message
                displayCitations(lastCitations, lastResponseText);
            } else {
                addMessage('bot', 'Failed to load chat history.');
            }
        } catch (error) {
            hideLoadingIndicator();
            console.error('Error loading history:', error);
            addMessage('bot', 'Failed to load chat history due to a network error.');
        } finally {
            isLoading = false;
            sendBtn.disabled = false;
            userInput.disabled = false;
            userInput.focus();
            updateActiveSessionUI();
        }
    }

    async function listSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();

            if (data.success) {
                sessionList.innerHTML = ''; // Clear existing list
                if (data.sessions.length === 0) {
                    sessionList.innerHTML = '<li class="no-sessions-msg">No past sessions. Start a new chat!</li>';
                    return;
                }
                data.sessions.forEach(session => {
                    const listItem = document.createElement('li');
                    listItem.classList.add('session-item');
                    listItem.dataset.sessionId = session.id;
                    
                    // Create session item with menu button (3 dots)
                    listItem.innerHTML = `
                        <span class="session-item-text">${session.preview}</span>
                        <div class="session-menu">
                            <button class="session-menu-btn" title="Session Options">
                                <i class="fas fa-ellipsis-v"></i>
                            </button>
                            <div class="session-menu-dropdown" style="display: none;">
                                <button class="menu-option rename-btn">
                                    <i class="fas fa-edit"></i> Rename
                                </button>
                                <button class="menu-option new-tab-btn">
                                    <i class="fas fa-external-link-alt"></i> New Tab
                                </button>
                                <button class="menu-option delete-btn">
                                    <i class="fas fa-trash-alt"></i> Delete
                                </button>
                            </div>
                        </div>
                    `;
                    
                    // Add event listener for session loading
                    listItem.querySelector('.session-item-text').addEventListener('click', () => {
                        loadSession(session.id);
                    });
                    
                    // Add event listener for menu button
                    const menuBtn = listItem.querySelector('.session-menu-btn');
                    const menuDropdown = listItem.querySelector('.session-menu-dropdown');
                    
                    menuBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        // Hide other open menus
                        document.querySelectorAll('.session-menu-dropdown').forEach(dropdown => {
                            if (dropdown !== menuDropdown) {
                                dropdown.style.display = 'none';
                            }
                        });
                        // Toggle current menu
                        menuDropdown.style.display = menuDropdown.style.display === 'none' ? 'block' : 'none';
                    });
                    
                    // Add event listeners for menu options
                    listItem.querySelector('.rename-btn').addEventListener('click', (e) => {
                        e.stopPropagation();
                        renameSession(session.id, listItem);
                        menuDropdown.style.display = 'none';
                    });
                    
                    listItem.querySelector('.new-tab-btn').addEventListener('click', (e) => {
                        e.stopPropagation();
                        openSessionInNewTab(session.id);
                        menuDropdown.style.display = 'none';
                    });
                    
                    listItem.querySelector('.delete-btn').addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteSession(session.id);
                        menuDropdown.style.display = 'none';
                    });
                    
                    sessionList.appendChild(listItem);
                });
                
                updateActiveSessionUI(); // Highlight active session after rendering
            } else {
                console.error('Failed to list sessions:', data.message);
            }
        } catch (error) {
            console.error('Error fetching sessions:', error);
        }
        
        // Close menu dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.session-menu')) {
                document.querySelectorAll('.session-menu-dropdown').forEach(dropdown => {
                    dropdown.style.display = 'none';
                });
            }
        });
    }

    async function deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this chat session?')) {
            return;
        }

        try {
            const response = await fetch(`/api/delete_session/${sessionId}`, {
                method: 'DELETE',
            });
            const data = await response.json();

            if (data.success) {
                console.log(data.message);
                listSessions(); // Refresh list
                if (currentSessionId === sessionId) {
                    // If deleted session was active, start a new chat
                    startNewChat();
                }
            } else {
                console.error('Failed to delete session:', data.message);
                alert('Failed to delete session: ' + (data.message || 'Unknown error.'));
            }
        } catch (error) {
            console.error('Error deleting session:', error);
            alert('Error deleting session due to network issue.');
        }
    }

    // Function to update the active session's UI in the sidebar
    function updateActiveSessionUI() {
        document.querySelectorAll('.session-item').forEach(item => {
            if (item.dataset.sessionId === currentSessionId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    function updateSessionPreview(sessionId, latestUserMessage) {
        const sessionItem = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
        if (sessionItem) {
            const previewText = sessionItem.querySelector('.session-item-text');
            if (previewText) {
                // Update preview if it's still 'New Chat Session' or similar generic text
                if (previewText.textContent === 'New Chat Session' || previewText.textContent.startsWith('Chat Session')) {
                    const newPreview = latestUserMessage.substring(0, 50) + (latestUserMessage.length > 50 ? '...' : '');
                    previewText.textContent = newPreview;
                }
            }
            // Move the current session to the top of the list for better UX
            sessionList.prepend(sessionItem);
        } else {
            // If the session item doesn't exist (e.g., first message in a new chat), refresh the list
            listSessions();
        }
    }

    // New session management functions
    function renameSession(sessionId, listItem) {
        const currentText = listItem.querySelector('.session-item-text').textContent;
        const newName = prompt('Enter new session name:', currentText);
        if (newName && newName.trim() !== '' && newName !== currentText) {
            listItem.querySelector('.session-item-text').textContent = newName.trim();
            // Note: In a full implementation, you'd also save this to the backend
            console.log(`Renamed session ${sessionId} to: ${newName}`);
        }
    }

    function openSessionInNewTab(sessionId) {
        const currentUrl = window.location.href;
        const newTabUrl = currentUrl + (currentUrl.includes('?') ? '&' : '?') + `session=${sessionId}`;
        window.open(newTabUrl, '_blank');
    }

    // --- Initial Load ---
    const initialSessionId = getSavedSessionId();
    if (initialSessionId) {
        loadSession(initialSessionId);
    } else {
        startNewChat(); // Start a new chat if no saved session
    }
    listSessions(); // Always list sessions on page load
});