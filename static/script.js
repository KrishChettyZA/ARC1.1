document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const sessionList = document.getElementById('sessionList');
    const referencesContent = document.getElementById('referencesContent');
    const citationCountSpan = document.getElementById('citationCount');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    const toggleIcon = toggleSidebarBtn.querySelector('i');
    const referencesPanel = document.getElementById('referencesPanel');

    // --- Mobile Specific Elements ---
    const mobileHistoryBtn = document.getElementById('mobileHistoryBtn');
    const mobileReferencesBtn = document.getElementById('mobileReferencesBtn');
    const overlayBackdrop = document.getElementById('overlayBackdrop');


    let currentSessionId = null;
    let isLoading = false;
    const messageCitationsMap = new Map();

    // --- Core Functions ---

    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    function generateMessageId() {
        return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function saveCurrentSessionId(sessionId) {
        localStorage.setItem('currentSessionId', sessionId);
    }

    function getSavedSessionId() {
        return localStorage.getItem('currentSessionId');
    }
    
    function getInitialSessionId() {
        const urlParams = new URLSearchParams(window.location.search);
        const sessionFromUrl = urlParams.get('session');
        if (sessionFromUrl) {
            window.history.replaceState({}, document.title, window.location.pathname);
            return sessionFromUrl;
        }
        return getSavedSessionId();
    }

    function formatMessageWithCitations(text) {
        const rawHtml = marked.parse(text);
        return rawHtml.replace(/\[(\d+)\]/g, (match, citationNumber) => {
            return ` <a href="#" class="citation-link" data-citation-id="${citationNumber}"><sup>[${citationNumber}]</sup></a>`;
        });
    }

    function addMessage(sender, text, citations = [], messageId = null) {
        const newId = messageId || generateMessageId();
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', `${sender}-message`);
        messageElement.id = newId;

        const messageBubble = document.createElement('div');
        messageBubble.classList.add('message-bubble');
        
        if (sender === 'bot') {
            messageBubble.innerHTML = formatMessageWithCitations(text);
            if (citations.length > 0) {
                messageCitationsMap.set(newId, citations);
            }
        } else {
            messageBubble.textContent = text;
        }
        
        messageElement.appendChild(messageBubble);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        if (sender === 'bot') {
            displayCitations(citations);
        }
        return messageElement;
    }
    
    function displayCitations(citations) {
        referencesContent.innerHTML = '';
        citationCountSpan.textContent = `(${citations.length})`;

        if (citations.length === 0) {
            referencesContent.innerHTML = '<p class="no-references">Citations for responses will appear here.</p>';
            return;
        }

        citations.forEach(citation => {
            const citationItem = document.createElement('div');
            citationItem.classList.add('reference-item');
            citationItem.setAttribute('data-citation-id', citation.id);

            const tempDiv = document.createElement('div');
            tempDiv.textContent = citation.content;
            const sanitizedContent = tempDiv.innerHTML;

            const truncatedContent = sanitizedContent.length > 300 
                ? sanitizedContent.substring(0, 300) + '...' 
                : sanitizedContent;
            
            citationItem.innerHTML = `
                <h3>
                    <span class="ref-id">${citation.id}</span> 
                    <span class="ref-source">${citation.source || 'Unknown Source'}</span>
                </h3>
                <p class="ref-page">${citation.page_number ? `Detail: ${citation.page_number}` : ''}</p>
                <div class="ref-content">${truncatedContent.replace(/\n/g, '<br>')}</div>
                ${citation.source ? `<a href="/documents/${encodeURIComponent(citation.source)}" target="_blank" class="view-pdf-btn">
                    <i class="fas fa-file-pdf"></i> View PDF
                </a>` : ''}
            `;
            referencesContent.appendChild(citationItem);
        });
    }
    
    function handleCitationClick(event) {
        const link = event.target.closest('.citation-link');
        if (!link) return;

        event.preventDefault();
        const citationId = link.dataset.citationId;
        const messageElement = link.closest('.chat-message');
        if (!citationId || !messageElement) return;

        const messageId = messageElement.id;
        const relevantCitations = messageCitationsMap.get(messageId);

        if (relevantCitations) {
            displayCitations(relevantCitations);
            const referenceItem = document.querySelector(`.references-panel .reference-item[data-citation-id="${citationId}"]`);
            if (referenceItem) {
                referenceItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                document.querySelectorAll('.reference-item.highlight').forEach(item => item.classList.remove('highlight'));
                referenceItem.classList.add('highlight');
                setTimeout(() => {
                    referenceItem.classList.remove('highlight');
                }, 2000);
            }
        }
    }

    function showLoading() {
        isLoading = true;
        sendBtn.disabled = true;
        userInput.disabled = true;
    }

    function hideLoading() {
        isLoading = false;
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
    
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '' || isLoading) return;

        addMessage('user', message);
        userInput.value = '';
        showLoading();

        const botMessageElement = addMessage('bot', '');
        const messageBubble = botMessageElement.querySelector('.message-bubble');
        let fullResponseText = "";

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message, session_id: currentSessionId }),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const events = chunk.split('\n\n');
                
                for (const event of events) {
                    if (event.startsWith('data:')) {
                        const dataStr = event.substring(5);
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.type === 'citations') {
                                const citations = data.payload;
                                messageCitationsMap.set(botMessageElement.id, citations);
                                displayCitations(citations);
                            } else if (data.type === 'text') {
                                fullResponseText += data.payload;
                                messageBubble.innerHTML = formatMessageWithCitations(fullResponseText);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } else if (data.type === 'error') {
                                messageBubble.innerHTML = `<p class="error">${data.payload}</p>`;
                            }
                        } catch (e) {
                            // Ignore incomplete JSON chunks
                        }
                    }
                }
            }
            updateSessionPreview(currentSessionId, message);

        } catch (error) {
            console.error('Error sending message:', error);
            messageBubble.innerHTML = 'Sorry, I am unable to connect to the server. Please try again later.';
        } finally {
            hideLoading();
        }
    }

    function startNewChat() {
        currentSessionId = generateSessionId();
        saveCurrentSessionId(currentSessionId);
        messageCitationsMap.clear();
        
        chatMessages.innerHTML = `
            <div class="chat-message bot-message intro-message">
                <div class="message-bubble">
                    Hello! I am the <strong>AI Powered Career Guidance Tool</strong>, an expert on the Refracted Economies Framework. I'm here to provide detailed, comprehensive, and strategic career guidance. How can I assist you today?
                </div>
            </div>
        `;
        displayCitations([]);
        userInput.value = '';
        listSessions();
        updateActiveSessionUI();
        userInput.focus();
    }

    async function loadSession(sessionId) {
        if (isLoading || !sessionId) return;
        
        showLoading(true);
        loadingOverlay.style.display = 'flex';
        currentSessionId = sessionId;
        saveCurrentSessionId(sessionId);
        chatMessages.innerHTML = '';
        messageCitationsMap.clear();
        
        try {
            const response = await fetch(`/api/history?session_id=${sessionId}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();

            if (data.success && data.history) {
                let lastBotCitations = [];
                data.history.forEach(turn => {
                    if (turn.role === 'user' || turn.role === 'model') {
                        const messageText = (turn.parts && turn.parts[0]) || '';
                        const messageCitations = Array.isArray(turn.citations) ? turn.citations : [];
                        const sender = turn.role === 'user' ? 'user' : 'bot';
                        addMessage(sender, messageText, messageCitations);
                        if (sender === 'bot') {
                            lastBotCitations = messageCitations;
                        }
                    }
                });
                displayCitations(lastBotCitations);
            } else {
                addMessage('bot', 'Failed to load chat history.');
            }
        } catch (error) {
            console.error('Error loading history:', error);
            addMessage('bot', 'Failed to load chat history due to a network error.');
        } finally {
            hideLoading();
            loadingOverlay.style.display = 'none';
            updateActiveSessionUI();
        }
    }

    async function listSessions() {
        try {
            const response = await fetch('/api/sessions');
            if (!response.ok) throw new Error('Failed to fetch sessions');
            const data = await response.json();

            sessionList.innerHTML = '';
            if (data.success && data.sessions.length > 0) {
                data.sessions.forEach(session => {
                    const listItem = createSessionListItem(session);
                    sessionList.appendChild(listItem);
                });
            } else {
                sessionList.innerHTML = '<li class="no-sessions-msg">No past sessions found.</li>';
            }
        } catch (error) {
            console.error('Error fetching sessions:', error);
            sessionList.innerHTML = '<li class="no-sessions-msg">Could not load sessions.</li>';
        }
        updateActiveSessionUI();
    }
    
    function createSessionListItem(session) {
        const listItem = document.createElement('li');
        listItem.classList.add('session-item');
        listItem.dataset.sessionId = session.id;
        listItem.innerHTML = `
            <span class="session-item-text">${session.preview}</span>
            <div class="session-menu">
                <button class="session-menu-btn" title="Session Options"><i class="fas fa-ellipsis-v"></i></button>
                <div class="session-menu-dropdown" style="display: none;">
                    <button class="menu-option delete-btn"><i class="fas fa-trash-alt"></i> Delete</button>
                </div>
            </div>
        `;
        listItem.querySelector('.session-item-text').addEventListener('click', () => loadSession(session.id));
        const menuBtn = listItem.querySelector('.session-menu-btn');
        const menuDropdown = listItem.querySelector('.session-menu-dropdown');
        menuBtn.addEventListener('click', e => {
            e.stopPropagation();
            document.querySelectorAll('.session-menu-dropdown').forEach(d => { if (d !== menuDropdown) d.style.display = 'none'; });
            menuDropdown.style.display = menuDropdown.style.display === 'none' ? 'block' : 'none';
        });
        listItem.querySelector('.delete-btn').addEventListener('click', e => { e.stopPropagation(); deleteSession(session.id); });
        
        return listItem;
    }

    async function deleteSession(sessionId) {
        if (!window.confirm('Are you sure you want to delete this chat session? This cannot be undone.')) return;
        try {
            const response = await fetch(`/api/delete_session/${sessionId}`, { method: 'DELETE' });
            const data = await response.json();
            if (data.success) {
                if (currentSessionId === sessionId) {
                    startNewChat();
                }
                listSessions();
            } else {
                alert('Failed to delete session: ' + (data.message || 'Unknown error.'));
            }
        } catch (error) {
            console.error('Error deleting session:', error);
            alert('An error occurred while deleting the session.');
        }
    }
    
    function updateActiveSessionUI() {
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.toggle('active', item.dataset.sessionId === currentSessionId);
        });
    }

    function updateSessionPreview(sessionId, latestUserMessage) {
        let sessionItem = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
        if (!sessionItem) {
             listSessions(); 
             return;
        }
        const previewText = sessionItem.querySelector('.session-item-text');
        if (previewText && (previewText.textContent === 'New Chat Session' || !previewText.textContent)) {
            previewText.textContent = latestUserMessage.substring(0, 50) + (latestUserMessage.length > 50 ? '...' : '');
        }
        sessionList.prepend(sessionItem);
    }
    
    // --- Event Listeners & Initial Load ---
    
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    newChatBtn.addEventListener('click', startNewChat);
    
    chatMessages.addEventListener('click', handleCitationClick);
    
    // Desktop sidebar toggle
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        const isCollapsed = sidebar.classList.contains('collapsed');
        toggleIcon.className = isCollapsed ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
    });

    // **FIX:** Added event listeners for mobile panel toggles
    mobileHistoryBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        overlayBackdrop.style.display = 'block';
    });

    mobileReferencesBtn.addEventListener('click', () => {
        referencesPanel.classList.add('open');
        overlayBackdrop.style.display = 'block';
    });

    overlayBackdrop.addEventListener('click', () => {
        sidebar.classList.remove('open');
        referencesPanel.classList.remove('open');
        overlayBackdrop.style.display = 'none';
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.session-menu')) {
            document.querySelectorAll('.session-menu-dropdown').forEach(d => d.style.display = 'none');
        }
    });

    async function initializeApp() {
        await listSessions();
        const initialSessionId = getInitialSessionId();
        if (initialSessionId) {
            await loadSession(initialSessionId);
        } else {
            startNewChat();
        }
    }
    
    initializeApp();
});
