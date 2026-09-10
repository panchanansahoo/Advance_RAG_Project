/**
 * Advanced RAG Assistant — ChatGPT-Style Frontend (Full Featured)
 * Complete conversation management, document upload, streaming chat,
 * citations, model selector, stop/regenerate, keyboard shortcuts.
 */

const API_BASE = (typeof window !== "undefined" && window.__API_BASE__)
    ? window.__API_BASE__
    : (typeof window !== "undefined" && (window.location.port === "5500" || window.location.port === "3000" || window.location.port === "5173" || window.location.protocol === "file:"))
        ? "http://localhost:8000"
        : "";

async function getAuthHeaders(extra = {}) {
    const headers = { ...extra };
    if (state.supabase) {
        try {
            const { data } = await state.supabase.auth.getSession();
            if (data?.session?.access_token) {
                headers['Authorization'] = `Bearer ${data.session.access_token}`;
            }
        } catch (e) {
            console.warn('Unable to get Supabase session for auth headers', e);
        }
    }
    return headers;
}

// ── State ──────────────────────────────────────────────────
const state = {
    documents: [],
    conversations: [],
    activeConversationId: null,
    isLoading: false,
    searchQuery: '',
    abortController: null,      // For stopping generation
    lastQuery: null,             // For regenerating
    selectedDocIds: [],          // Track which docs to query against
    forceRoute: '',              // Force a specific route (rag, agentic, structured_data, visual)
    supabase: null,
};

// ── DOM Elements ───────────────────────────────────────────
const el = {
    sidebar: document.getElementById('sidebar'),
    sidebarOverlay: document.getElementById('sidebarOverlay'),
    sidebarCloseBtn: document.getElementById('sidebarCloseBtn'),
    sidebarOpenBtn: document.getElementById('sidebarOpenBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    convSearch: document.getElementById('convSearch'),
    conversationList: document.getElementById('conversationList'),
    sidebarDocuments: document.getElementById('sidebarDocuments'),
    docsToggle: document.getElementById('docsToggle'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    documentList: document.getElementById('documentList'),
    docCount: document.getElementById('docCount'),
    chatMessages: document.getElementById('chatMessages'),
    chatMessagesInner: document.getElementById('chatMessagesInner'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    chatForm: document.getElementById('chatForm'),
    queryInput: document.getElementById('queryInput'),
    sendButton: document.getElementById('sendButton'),
    attachBtn: document.getElementById('attachBtn'),
    pendingFiles: document.getElementById('pendingFiles'),
    modelSelector: document.getElementById('modelSelector'),
    evaluationBtn: document.getElementById('evaluationBtn'),
    evaluationPanel: document.getElementById('evaluationPanel'),
    evaluationCloseBtn: document.getElementById('evaluationCloseBtn'),
    runEvaluationBtn: document.getElementById('runEvaluationBtn'),
    runExperimentsBtn: document.getElementById('runExperimentsBtn'),
    evaluationStatus: document.getElementById('evaluationStatus'),
    evaluationMetrics: document.getElementById('evaluationMetrics'),
    evaluationExperiments: document.getElementById('evaluationExperiments'),
};

// ── Initialization ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initUpload();
    initChat();
    initFeatureCards();
    initGlobalDragDrop();
    initKeyboardShortcuts();
    initModelSelector();
    initEvaluation();
    initAuthButtons();
    void initSupabase().then(loadAuthState);
    loadConversations();
    loadDocuments();
});

async function initSupabase() {
    if (state.supabase) return state.supabase;
    try {
        const response = await fetch(`${API_BASE}/api/v1/auth/config`);
        if (!response.ok) {
            console.warn(`Supabase config returned HTTP ${response.status}`);
            return null;
        }
        const config = await response.json();
        const rawUrl = config.supabase_url ? String(config.supabase_url).trim() : '';
        // Strip trailing /rest/v1 or trailing slashes if present
        const cleanUrl = rawUrl.replace(/\/rest\/v1\/?$/, '').replace(/\/+$/, '');
        if (cleanUrl && config.supabase_anon_key && window.supabase) {
            state.supabase = window.supabase.createClient(cleanUrl, config.supabase_anon_key);
            state.supabase.auth.onAuthStateChange(() => loadAuthState());
            return state.supabase;
        }
    } catch (error) {
        console.warn('Supabase Auth is unavailable', error);
    }
    return null;
}

function initAuthButtons() {
    document.querySelectorAll('[data-auth-provider]').forEach((button) => {
        button.addEventListener('click', async () => {
            const provider = button.dataset.authProvider;
            if (!state.supabase) {
                await initSupabase();
            }
            if (state.supabase) {
                const { error } = await state.supabase.auth.signInWithOAuth({
                    provider: provider,
                    options: { redirectTo: window.location.origin }
                });
                if (error) {
                    showToast(`Login failed: ${error.message}`, 'error');
                }
            } else {
                showToast('Supabase login is not configured. Please ensure backend is running with SUPABASE_URL and SUPABASE_ANON_KEY.', 'error');
            }
        });
    });
    document.getElementById('authModalClose')?.addEventListener('click', closeAuthModal);
    document.getElementById('authModal')?.addEventListener('click', (event) => {
        if (event.target.id === 'authModal') closeAuthModal();
    });
    const userPanel = document.getElementById('userPanel');
    userPanel?.addEventListener('click', (event) => {
        if (event.target.closest('[data-auth-provider], #userMenuLogout, #userMenuLogin')) return;
        toggleUserMenu();
    });
    userPanel?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggleUserMenu(); }
    });
    document.getElementById('userMenuLogin')?.addEventListener('click', openAuthModal);
    document.getElementById('userMenuLogout')?.addEventListener('click', logoutUser);
}

async function loadAuthState() {
    try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${API_BASE}/api/v1/auth/me`, { credentials: 'include', headers });
        const session = await response.json();
        const loggedIn = Boolean(session.authenticated);
        const displayName = loggedIn ? (session.name || session.email || 'Account') : 'Guest account';
        const avatarText = loggedIn ? (session.name || session.email || 'U').slice(0, 2).toUpperCase() : 'G';
        document.getElementById('userName').textContent = displayName;
        document.getElementById('userAvatar').textContent = avatarText;
        document.getElementById('userMenuEmail').textContent = loggedIn ? session.email : 'Not signed in';
        const providerEl = document.getElementById('userMenuProvider');
        if (providerEl) {
            providerEl.textContent = loggedIn ? '' : 'Two free questions available';
            providerEl.hidden = loggedIn;
            providerEl.style.display = loggedIn ? 'none' : '';
        }
        const loginBtn = document.getElementById('userMenuLogin');
        if (loginBtn) {
            loginBtn.hidden = loggedIn;
            loginBtn.style.display = loggedIn ? 'none' : '';
        }
        const logoutBtn = document.getElementById('userMenuLogout');
        if (logoutBtn) {
            logoutBtn.hidden = !loggedIn;
            logoutBtn.style.display = loggedIn ? '' : 'none';
        }
        const authActions = document.querySelector('.user-auth-actions');
        if (authActions) {
            authActions.hidden = loggedIn;
            authActions.style.display = loggedIn ? 'none' : 'flex';
            authActions.classList.toggle('hidden', loggedIn);
        }
        if (loggedIn) {
            closeAuthModal();
        }
    } catch (error) {
        console.warn('Unable to load account state', error);
        document.getElementById('userName').textContent = 'Guest account';
        document.getElementById('userAvatar').textContent = 'G';
        document.getElementById('userMenuEmail').textContent = 'Not signed in';
        const providerEl = document.getElementById('userMenuProvider');
        if (providerEl) {
            providerEl.textContent = 'Two free questions available';
            providerEl.hidden = false;
            providerEl.style.display = '';
        }
        const loginBtn = document.getElementById('userMenuLogin');
        if (loginBtn) {
            loginBtn.hidden = false;
            loginBtn.style.display = '';
        }
        const logoutBtn = document.getElementById('userMenuLogout');
        if (logoutBtn) {
            logoutBtn.hidden = true;
            logoutBtn.style.display = 'none';
        }
        const authActions = document.querySelector('.user-auth-actions');
        if (authActions) {
            authActions.hidden = false;
            authActions.style.display = 'flex';
            authActions.classList.remove('hidden');
        }
    }
}

function toggleUserMenu() {
    const panel = document.getElementById('userPanel');
    const menu = document.getElementById('userMenu');
    const expanded = panel.getAttribute('aria-expanded') === 'true';
    panel.setAttribute('aria-expanded', String(!expanded));
    menu.setAttribute('aria-hidden', String(expanded));
    menu.classList.toggle('visible', !expanded);
}

async function logoutUser() {
    if (state.supabase) await state.supabase.auth.signOut();
    await fetch(`${API_BASE}/api/v1/auth/logout`, { method: 'POST', credentials: 'include' });
    toggleUserMenu();
    await loadAuthState();
    showToast('You have been logged out', 'info');
}

function openAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.add('visible');
    modal.setAttribute('aria-hidden', 'false');
    document.getElementById('authModalClose')?.focus();
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (!modal) return;
    modal.classList.remove('visible');
    modal.setAttribute('aria-hidden', 'true');
}

// ── Sidebar Logic ──────────────────────────────────────────
function initSidebar() {
    el.sidebarCloseBtn.addEventListener('click', () => {
        el.sidebar.classList.add('collapsed');
        el.sidebar.classList.remove('mobile-open');
        el.sidebarOverlay.classList.remove('visible');
    });

    el.sidebarOpenBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            el.sidebar.classList.add('mobile-open');
            el.sidebarOverlay.classList.add('visible');
        } else {
            el.sidebar.classList.remove('collapsed');
        }
    });

    el.sidebarOverlay.addEventListener('click', () => {
        el.sidebar.classList.remove('mobile-open');
        el.sidebarOverlay.classList.remove('visible');
    });

    el.newChatBtn.addEventListener('click', () => startNewChat());

    el.convSearch.addEventListener('input', (e) => {
        state.searchQuery = e.target.value.toLowerCase();
        renderConversations();
    });

    el.docsToggle.addEventListener('click', () => {
        el.sidebarDocuments.classList.toggle('collapsed-section');
    });
}

// ── Keyboard Shortcuts ─────────────────────────────────────
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Shift + O → New Chat
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'O') {
            e.preventDefault();
            startNewChat();
        }
        // Ctrl/Cmd + Shift + S → Toggle sidebar
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
            e.preventDefault();
            el.sidebar.classList.toggle('collapsed');
        }
        // Escape → Stop generation
        if (e.key === 'Escape' && state.isLoading) {
            e.preventDefault();
            stopGeneration();
        }
        // Focus input with /
        if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            el.queryInput.focus();
        }
    });
}

// ── Global Drag & Drop ─────────────────────────────────────
function initGlobalDragDrop() {
    let dragCounter = 0;
    const overlay = document.createElement('div');
    overlay.className = 'drag-overlay';
    overlay.innerHTML = `
        <div class="drag-overlay-content">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M12 16V4m0 0l-4 4m4-4l4 4" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20 16v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <p>Drop files to upload</p>
            <span>PDF, DOCX, XLSX, CSV, TXT, MD, Images</span>
        </div>
    `;
    document.body.appendChild(overlay);

    document.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        if (dragCounter === 1) {
            overlay.classList.add('visible');
        }
    });

    document.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            overlay.classList.remove('visible');
        }
    });

    document.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        overlay.classList.remove('visible');
        if (e.dataTransfer.files.length) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });
}

// ── Model Selector ─────────────────────────────────────────
function initModelSelector() {
    const btn = document.getElementById('modelSelectorBtn');
    const dropdown = document.getElementById('modelDropdown');
    const label = document.getElementById('modelLabel');
    if (!btn || !dropdown) return;

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.classList.toggle('visible');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.model-selector')) {
            dropdown.classList.remove('visible');
        }
    });

    // Handle option selection
    dropdown.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const route = option.dataset.route;
            const optLabel = option.dataset.label;

            state.forceRoute = route;
            label.textContent = optLabel;

            // Update active state
            dropdown.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
            option.classList.add('active');

            dropdown.classList.remove('visible');
        });
    });
}

// ── Evaluation Dashboard ──────────────────────────────────
function initEvaluation() {
    if (!el.evaluationBtn || !el.evaluationPanel) return;
    el.evaluationBtn.addEventListener('click', () => {
        el.evaluationPanel.classList.toggle('visible');
        el.evaluationPanel.setAttribute('aria-hidden', String(!el.evaluationPanel.classList.contains('visible')));
        if (el.evaluationPanel.classList.contains('visible')) loadEvaluationResults();
    });
    el.evaluationCloseBtn.addEventListener('click', () => {
        el.evaluationPanel.classList.remove('visible');
        el.evaluationPanel.setAttribute('aria-hidden', 'true');
    });
    el.runEvaluationBtn.addEventListener('click', runEvaluation);
    el.runExperimentsBtn.addEventListener('click', runExperiments);
}

async function loadEvaluationResults() {
    el.evaluationStatus.textContent = 'Loading latest results...';
    try {
        const [resultsResponse, experimentsResponse] = await Promise.all([
            fetch(`${API_BASE}/api/v1/evaluation/results`),
            fetch(`${API_BASE}/api/v1/evaluation/experiments`),
        ]);
        if (!resultsResponse.ok) throw new Error('Evaluation results unavailable');
        renderEvaluationResults(await resultsResponse.json());
        if (experimentsResponse.ok) renderExperimentResults(await experimentsResponse.json());
    } catch (error) {
        el.evaluationStatus.textContent = sanitizeErrorMessage(error.message, 'Evaluation data is currently unavailable.');
    }
}

function renderEvaluationResults(data) {
    const scores = data.aggregate_scores || {};
    const metrics = [
        ['Faithfulness', scores.faithfulness, true], ['Answer relevancy', scores.answer_relevancy, true],
        ['Average latency', scores.avg_latency_ms, false, 'ms'], ['Success rate', scores.success_rate, true],
        ['Total queries', scores.total_queries, false],
    ].filter((metric) => metric[1] !== undefined);
    el.evaluationStatus.textContent = data.status === 'no_results' ? 'No benchmark has been run yet.' : 'Latest benchmark results';
    el.evaluationMetrics.innerHTML = metrics.length ? metrics.map(([label, value, percentage, suffix]) => `
        <div class="evaluation-metric"><span>${label}</span><strong>${percentage ? `${Math.round(Number(value) * 100)}%` : `${Math.round(Number(value))}${suffix || ''}`}</strong></div>
    `).join('') : '<div class="evaluation-empty">Run the benchmark to populate quality and system metrics.</div>';
}

function renderExperimentResults(data) {
    if (!data || data.status === 'no_results') {
        el.evaluationExperiments.innerHTML = '<div class="evaluation-empty">No architecture comparison is available yet.</div>';
        return;
    }
    const rows = data.comparisons || data.results || data.experiments || [];
    el.evaluationExperiments.innerHTML = `<h3>Architecture comparison</h3>${Array.isArray(rows) && rows.length ? rows.map((row) => `<div class="experiment-row"><span>${escapeHtml(row.name || row.architecture || 'Configuration')}</span><strong>${row.score !== undefined ? row.score : row.average_score !== undefined ? row.average_score : 'Ready'}</strong></div>`).join('') : '<div class="evaluation-empty">Comparison report loaded.</div>'}`;
}

async function runEvaluation() {
    setEvaluationBusy(el.runEvaluationBtn, 'Running benchmark...');
    try {
        const response = await fetch(`${API_BASE}/api/v1/evaluation/run`, { method: 'POST' });
        if (!response.ok) throw new Error((await response.json()).detail || 'Unable to start benchmark');
        el.evaluationStatus.textContent = 'Benchmark running in the background...';
        pollEvaluationResults();
    } catch (error) {
        el.evaluationStatus.textContent = sanitizeErrorMessage(error.message);
        setEvaluationBusy(el.runEvaluationBtn, 'Run benchmark', false);
    }
}

async function runExperiments() {
    setEvaluationBusy(el.runExperimentsBtn, 'Comparing...');
    try {
        const response = await fetch(`${API_BASE}/api/v1/evaluation/experiments/run`, { method: 'POST' });
        if (!response.ok) throw new Error((await response.json()).detail || 'Unable to start comparison');
        el.evaluationStatus.textContent = 'Architecture comparison running in the background...';
        setTimeout(async () => { await loadEvaluationResults(); setEvaluationBusy(el.runExperimentsBtn, 'Compare architectures', false); }, 3000);
    } catch (error) {
        el.evaluationStatus.textContent = sanitizeErrorMessage(error.message);
        setEvaluationBusy(el.runExperimentsBtn, 'Compare architectures', false);
    }
}

function pollEvaluationResults(attempt = 0) {
    setTimeout(async () => {
        await loadEvaluationResults();
        if (el.evaluationStatus.textContent.includes('No benchmark') && attempt < 60) return pollEvaluationResults(attempt + 1);
        setEvaluationBusy(el.runEvaluationBtn, 'Run benchmark', false);
    }, 5000);
}

function setEvaluationBusy(button, label, busy = true) {
    button.disabled = busy;
    button.textContent = label;
}

// ── Feature Cards ──────────────────────────────────────────
function initFeatureCards() {
    document.querySelectorAll('.feature-card[data-prompt]').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) {
                el.queryInput.value = prompt;
                el.queryInput.dispatchEvent(new Event('input'));
                el.queryInput.focus();
            }
        });
    });
}

// ── Conversations ──────────────────────────────────────────
async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE}/api/v1/conversations/?limit=50`);
        if (!response.ok) return;
        const data = await response.json();
        state.conversations = data.conversations || [];
        renderConversations();
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
}

function renderConversations() {
    const filtered = state.conversations.filter(c =>
        !state.searchQuery || (c.title || '').toLowerCase().includes(state.searchQuery)
    );

    if (filtered.length === 0) {
        el.conversationList.innerHTML = `
            <div class="empty-state">
                ${state.searchQuery ? 'No matching conversations' : 'No conversations yet.<br>Start a new chat!'}
            </div>
        `;
        return;
    }

    // Group by relative time
    const now = new Date();
    const groups = { today: [], yesterday: [], thisWeek: [], thisMonth: [], older: [] };

    filtered.forEach(c => {
        const created = new Date(c.created_at);
        const diffMs = now - created;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffDays === 0) groups.today.push(c);
        else if (diffDays === 1) groups.yesterday.push(c);
        else if (diffDays < 7) groups.thisWeek.push(c);
        else if (diffDays < 30) groups.thisMonth.push(c);
        else groups.older.push(c);
    });

    let html = '';
    const labels = {
        today: 'Today', yesterday: 'Yesterday', thisWeek: 'Previous 7 Days',
        thisMonth: 'Previous 30 Days', older: 'Older'
    };

    for (const [key, label] of Object.entries(labels)) {
        if (groups[key].length) {
            html += `<div class="conv-section-label">${label}</div>`;
            html += groups[key].map(c => renderConvItem(c)).join('');
        }
    }

    el.conversationList.innerHTML = html;

    // Event delegation for conversation items
    el.conversationList.querySelectorAll('.conv-item').forEach(item => {
        const convId = item.dataset.id;
        item.addEventListener('click', (e) => {
            if (e.target.closest('.conv-action-btn')) return;
            if (e.target.closest('.conv-rename-input')) return;
            loadConversation(convId);
        });
    });

    el.conversationList.querySelectorAll('.conv-rename-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            startRename(btn.dataset.id);
        });
    });

    el.conversationList.querySelectorAll('.conv-delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteConversation(btn.dataset.id);
        });
    });
}

function renderConvItem(c) {
    const isActive = c.id === state.activeConversationId;
    const title = escapeHtml(c.title || 'New Conversation');
    return `
        <div class="conv-item ${isActive ? 'active' : ''}" data-id="${c.id}">
            <svg class="conv-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span class="conv-item-title">${title}</span>
            <div class="conv-item-actions">
                <button class="conv-action-btn conv-rename-btn" data-id="${c.id}" title="Rename">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
                <button class="conv-action-btn conv-delete-btn delete" data-id="${c.id}" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
            </div>
        </div>
    `;
}

function startNewChat() {
    state.activeConversationId = null;
    state.lastQuery = null;
    showWelcomeScreen();
    renderConversations();

    if (window.innerWidth <= 768) {
        el.sidebar.classList.remove('mobile-open');
        el.sidebarOverlay.classList.remove('visible');
    }

    el.queryInput.focus();
}

function showWelcomeScreen() {
    el.chatMessagesInner.innerHTML = `
        <div class="welcome-message" id="welcomeMessage">
            <div class="welcome-logo">
                <img src="logo.svg" alt="Agentic RAG Logo" width="42" height="42">
            </div>
            <h2>Advanced RAG Assistant</h2>
            <p>Upload documents and ask questions. Get evidence-grounded answers with citations from your knowledge base.</p>
            <div class="feature-cards">
                <div class="feature-card" data-prompt="Summarize the key points from the uploaded documents">
                    <span class="feature-emoji">📄</span>
                    <span>Summarize documents</span>
                </div>
                <div class="feature-card" data-prompt="What are the main topics covered in my documents?">
                    <span class="feature-emoji">🔍</span>
                    <span>Find key topics</span>
                </div>
                <div class="feature-card" data-prompt="Compare and contrast the main ideas across documents">
                    <span class="feature-emoji">📌</span>
                    <span>Compare ideas</span>
                </div>
                <div class="feature-card" data-prompt="Analyze the data and provide insights from the uploaded files">
                    <span class="feature-emoji">🤖</span>
                    <span>Analyze data</span>
                </div>
            </div>
        </div>
    `;
    initFeatureCards();
}

async function loadConversation(conversationId) {
    if (state.isLoading) return; // Prevent switching while loading

    state.activeConversationId = conversationId;
    renderConversations();

    // Show loading state in chat area
    el.chatMessagesInner.innerHTML = `
        <div class="message message-assistant" style="padding: 40px 0;">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/api/v1/conversations/${conversationId}`);
        if (!response.ok) throw new Error('Failed to load conversation');
        const conv = await response.json();

        el.chatMessagesInner.innerHTML = '';

        if (conv.messages && conv.messages.length > 0) {
            conv.messages.forEach(msg => {
                if (msg.role === 'user') {
                    appendMessage('user', msg.content);
                } else {
                    appendAssistantMessage(msg.content, msg.citations || []);
                }
            });

            // Set last query for regenerate
            const lastUserMsg = [...conv.messages].reverse().find(m => m.role === 'user');
            if (lastUserMsg) state.lastQuery = lastUserMsg.content;
        } else {
            showWelcomeScreen();
            state.activeConversationId = conversationId;
            renderConversations();
        }

        scrollToBottom();
    } catch (error) {
        console.error('Failed to load conversation:', error);
        showToast('Failed to load conversation', 'error');
        showWelcomeScreen();
    }

    if (window.innerWidth <= 768) {
        el.sidebar.classList.remove('mobile-open');
        el.sidebarOverlay.classList.remove('visible');
    }
}

async function startRename(conversationId) {
    const item = el.conversationList.querySelector(`.conv-item[data-id="${conversationId}"]`);
    if (!item) return;

    const titleSpan = item.querySelector('.conv-item-title');
    const currentTitle = titleSpan.textContent.trim();

    const input = document.createElement('input');
    input.className = 'conv-rename-input';
    input.value = currentTitle;
    titleSpan.replaceWith(input);
    input.focus();
    input.select();

    let finished = false;
    const finishRename = async () => {
        if (finished) return;
        finished = true;

        const newTitle = input.value.trim() || currentTitle;
        const span = document.createElement('span');
        span.className = 'conv-item-title';
        span.textContent = newTitle;
        input.replaceWith(span);

        if (newTitle !== currentTitle) {
            try {
                await fetch(`${API_BASE}/api/v1/conversations/${conversationId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: newTitle }),
                });
                loadConversations();
            } catch (err) {
                showToast('Failed to rename', 'error');
            }
        }
    };

    input.addEventListener('blur', finishRename);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = currentTitle; input.blur(); }
    });
}

async function deleteConversation(conversationId) {
    if (!confirm('Delete this conversation and all messages?')) return;

    try {
        await fetch(`${API_BASE}/api/v1/conversations/${conversationId}`, { method: 'DELETE' });
        if (state.activeConversationId === conversationId) {
            startNewChat();
        }
        loadConversations();
        showToast('Conversation deleted', 'info');
    } catch (error) {
        showToast('Failed to delete conversation', 'error');
    }
}

// ── Smart Error Sanitization ───────────────────────────────
function sanitizeErrorMessage(raw, fallback = 'We encountered a temporary issue. Please try again.') {
    if (!raw) return fallback;

    let str = '';
    if (typeof raw === 'string') {
        str = raw;
    } else if (Array.isArray(raw)) {
        const first = raw[0];
        if (first && typeof first === 'object') {
            const loc = Array.isArray(first.loc) ? first.loc : [];
            const field = loc[loc.length - 1];
            if (field === 'query') return 'Please type a question before submitting.';
            if (field === 'file' || field === 'files') return 'Please select a valid document.';
            return 'Some required information is missing. Please check your input.';
        }
        str = JSON.stringify(raw);
    } else if (typeof raw === 'object') {
        if (raw.detail) return sanitizeErrorMessage(raw.detail, fallback);
        if (raw.message) str = String(raw.message);
        else str = JSON.stringify(raw);
    } else {
        str = String(raw);
    }

    const lower = str.toLowerCase();
    if (lower.includes('not supported') || lower.includes('allowed extensions') || lower.includes('allowed:')) {
        return 'This file type is not supported. Please upload PDF, DOCX, XLSX, CSV, MD, TXT, or image files.';
    }
    if (lower.includes('exceeds') || lower.includes('file size')) {
        return 'This file exceeds the maximum upload size.';
    }
    if (lower.includes('file is empty') || lower.includes('empty file')) {
        return 'The file appears to be empty.';
    }
    if (lower.includes('fetch') || lower.includes('network') || lower.includes('connection') || lower.includes('refused') || lower.includes('offline') || lower.includes('econnrefused')) {
        return 'Service unreachable. Please check your connection.';
    }
    if (lower.includes('timeout') || lower.includes('timed out')) {
        return 'The request timed out. Try a shorter question.';
    }
    if (lower.includes('rate limit') || lower.includes('quota') || lower.includes('429') || lower.includes('too many requests')) {
        return 'High demand. Please wait a moment and try again.';
    }
    if (
        lower.includes('traceback') || lower.includes('exception') ||
        lower.includes('internal server error') || lower.includes('status code') ||
        lower.includes('syntaxerror') || lower.includes('keyerror') ||
        lower.includes('valueerror') || lower.includes('fastapi') ||
        lower.includes('sql') || lower.includes('undefined') ||
        /[{}[\]<>()_\\/]{3,}/.test(str)
    ) {
        return fallback;
    }

    return str.trim() || fallback;
}

function sanitizeBotAnswer(answer) {
    if (!answer || !answer.trim()) {
        return `No matching information was found in the selected documents.\n\n**Suggestions:**\n- Ensure relevant documents are uploaded\n- Try rephrasing your question\n- Upload additional context material`;
    }
    const lower = answer.toLowerCase().trim();
    if (
        lower.startsWith('execution error:') ||
        lower.startsWith('syntax error in generated code:') ||
        lower.startsWith('raw result (summarization failed):') ||
        lower.includes('traceback (most recent call last)')
    ) {
        return 'I encountered an issue processing this data. Please verify your question and try again.';
    }
    if (lower.startsWith('execution timeout:')) {
        return 'Analysis took longer than expected. Try a more focused question.';
    }
    if (lower.includes('an error occurred while generating the final answer')) {
        return 'I analyzed the documents but had trouble formulating the answer. Please try again.';
    }
    return answer;
}

// ── Upload Handling ────────────────────────────────────────
function initUpload() {
    const { uploadZone, fileInput, attachBtn } = el;

    uploadZone.addEventListener('click', () => fileInput.click());
    attachBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(Array.from(e.target.files));
            fileInput.value = '';
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    showToast(`Uploading ${file.name}...`, 'info');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name.replace(/\.[^/.]+$/, ''));

    try {
        const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            let detail = 'Upload failed';
            try { detail = (await response.json()).detail || detail; } catch (e) {}
            throw new Error(detail);
        }

        const doc = await response.json();
        showToast(`${file.name} uploaded successfully!`, 'success');
        pollDocumentStatus(doc.id);
        loadDocuments();
    } catch (error) {
        showToast(sanitizeErrorMessage(error.message, 'Unable to upload file.'), 'error');
    }
}

async function pollDocumentStatus(docId) {
    let attempts = 0;
    const poll = async () => {
        attempts++;
        try {
            const response = await fetch(`${API_BASE}/api/v1/documents/${docId}`);
            if (response.ok) {
                const doc = await response.json();
                if (doc.status === 'completed') {
                    showToast(`${doc.filename} — ${doc.chunk_count} chunks indexed`, 'success');
                    loadDocuments();
                    return;
                } else if (doc.status === 'failed') {
                    showToast(`Processing failed for ${doc.filename}`, 'error');
                    loadDocuments();
                    return;
                }
            }
        } catch (e) {}
        if (attempts < 60) setTimeout(poll, 3000);
    };
    setTimeout(poll, 2000);
}

// ── Document List ──────────────────────────────────────────
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/v1/documents/`);
        if (!response.ok) return;
        const data = await response.json();
        state.documents = data.documents || [];

        // Auto-select all completed documents for querying
        state.selectedDocIds = state.documents
            .filter(d => d.status === 'completed')
            .map(d => d.id);

        renderDocuments();
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function renderDocuments() {
    el.docCount.textContent = state.documents.length;

    if (state.documents.length === 0) {
        el.documentList.innerHTML = `<div class="empty-state">No documents yet</div>`;
        return;
    }

    el.documentList.innerHTML = state.documents.map(doc => {
        const ext = doc.filename.split('.').pop().toLowerCase();
        const iconClass = getIconClass(ext);
        const size = formatFileSize(doc.file_size);
        const isSelected = state.selectedDocIds.includes(doc.id);

        return `
            <div class="doc-item ${isSelected ? 'selected' : ''}" data-id="${doc.id}" onclick="toggleDocSelection('${doc.id}')">
                <div class="doc-icon ${iconClass}">${ext}</div>
                <div class="doc-info">
                    <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="doc-meta">
                        <span>${size}</span>
                        <span>•</span>
                        <span class="doc-status ${doc.status}">${getStatusLabel(doc.status, doc.chunk_count)}</span>
                    </div>
                </div>
                <button class="doc-delete" onclick="deleteDocument('${doc.id}', event)" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
            </div>
        `;
    }).join('');
}

function toggleDocSelection(docId) {
    const idx = state.selectedDocIds.indexOf(docId);
    if (idx >= 0) {
        state.selectedDocIds.splice(idx, 1);
    } else {
        state.selectedDocIds.push(docId);
    }
    renderDocuments();
}

function getIconClass(ext) {
    return { pdf: 'pdf', txt: 'txt', md: 'md', csv: 'csv', xlsx: 'xlsx', docx: 'docx', png: 'img', jpg: 'img', jpeg: 'img' }[ext] || 'txt';
}

function getStatusLabel(status, chunkCount) {
    switch (status) {
        case 'completed': return `✓ ${chunkCount || 0} chunks`;
        case 'processing': return '⟳ Processing...';
        case 'pending': return '◷ Pending';
        case 'failed': return '✗ Failed';
        default: return status;
    }
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function deleteDocument(docId, event) {
    event.stopPropagation();
    if (!confirm('Delete this document and all its data?')) return;

    try {
        await fetch(`${API_BASE}/api/v1/documents/${docId}`, { method: 'DELETE' });
        showToast('Document deleted', 'info');
        state.selectedDocIds = state.selectedDocIds.filter(id => id !== docId);
        loadDocuments();
    } catch (error) {
        showToast('Failed to delete document', 'error');
    }
}

// ── Chat ───────────────────────────────────────────────────
function initChat() {
    const { chatForm, queryInput, sendButton } = el;

    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 180) + 'px';
        updateSendButton();
    });

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (queryInput.value.trim() && !state.isLoading) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query || state.isLoading) return;
        await sendQuery(query);
    });
}

function updateSendButton() {
    if (state.isLoading) {
        el.sendButton.disabled = false; // Can be used to stop
        el.sendButton.classList.add('stop-mode');
        el.sendButton.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
            </svg>
        `;
        el.sendButton.title = 'Stop generating (Esc)';
        el.sendButton.onclick = (e) => { e.preventDefault(); stopGeneration(); };
    } else {
        el.sendButton.classList.remove('stop-mode');
        el.sendButton.disabled = el.queryInput.value.trim().length === 0;
        el.sendButton.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M7 11l5-5 5 5M12 6v13" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        `;
        el.sendButton.title = 'Send message';
        el.sendButton.onclick = null;
    }
}

function stopGeneration() {
    if (state.abortController) {
        state.abortController.abort();
        state.abortController = null;
    }
    state.isLoading = false;
    updateSendButton();
}

async function loginForMoreQuestions() {
    openAuthModal();
    return false;
}

async function sendQuery(query) {
    const { queryInput, chatMessages, chatMessagesInner } = el;

    // Hide welcome
    const welcome = document.getElementById('welcomeMessage');
    if (welcome) welcome.remove();

    // Add user message
    appendMessage('user', query);

    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';
    state.isLoading = true;
    state.lastQuery = query;
    state.abortController = new AbortController();
    updateSendButton();

    // Create assistant message placeholder
    const msgId = 'msg-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.id = msgId;
    msgDiv.className = 'message message-assistant';
    msgDiv.innerHTML = `
        <div class="assistant-header">
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 19l-5-5 5-5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 5l5 5-5 5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M13 3l-2 18" stroke-linecap="round"/>
                </svg>
            </div>
            <span class="assistant-name">RAG Assistant</span>
        </div>
        <div class="message-content"><span class="cursor"></span></div>
    `;
    chatMessagesInner.appendChild(msgDiv);
    scrollToBottom();

    let fullAnswer = '';
    let citations = [];

    try {
        // Build request with document_ids for scoped retrieval
        const requestBody = {
            query,
            conversation_id: state.activeConversationId || undefined,
        };

        // Pass document_ids to scope retrieval to selected documents
        if (state.selectedDocIds.length > 0) {
            requestBody.document_ids = state.selectedDocIds;
        }

        // Pass forced route from model selector
        if (state.forceRoute) {
            requestBody.force_route = state.forceRoute;
        }

        const headers = await getAuthHeaders({ 'Content-Type': 'application/json' });
        let response = await fetch(`${API_BASE}/api/v1/query_stream`, {
            method: 'POST',
            headers,
            credentials: 'include',
            body: JSON.stringify(requestBody),
            signal: state.abortController.signal,
        });

        if (response.status === 401) {
            const loggedIn = await loginForMoreQuestions();
            if (!loggedIn) return;
            const retryHeaders = await getAuthHeaders({ 'Content-Type': 'application/json' });
            response = await fetch(`${API_BASE}/api/v1/query_stream`, {
                method: 'POST',
                headers: retryHeaders,
                credentials: 'include',
                body: JSON.stringify(requestBody),
                signal: state.abortController.signal,
            });
        }

        if (!response.ok) {
            let detail = 'Something went wrong.';
            try { detail = (await response.json()).detail || detail; } catch (e) {}
            throw new Error(detail);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let streamBuffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            streamBuffer += decoder.decode(value, { stream: true });
            const lines = streamBuffer.split('\n');
            streamBuffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.error) throw new Error(data.error);
                    if (data.citations && data.citations.length > 0) {
                        citations = data.citations;
                    }
                    if (data.chunk) fullAnswer += data.chunk;
                    if (data.answer) fullAnswer = data.answer;
                    if (data.conversation_id && !state.activeConversationId) {
                        state.activeConversationId = data.conversation_id;
                    }
                } catch (e) {
                    if (e.name === 'AbortError') throw e;
                    if (e.message && !e.message.includes('JSON')) throw e;
                }
            }

            // Update streaming content
            const contentEl = msgDiv.querySelector('.message-content');
            contentEl.innerHTML = formatMarkdown(fullAnswer) + '<span class="cursor"></span>';
            scrollToBottom();
        }

        const finalLine = streamBuffer.trim();
        if (finalLine) {
            const data = JSON.parse(finalLine);
            if (data.error) throw new Error(data.error);
            if (data.citations && data.citations.length > 0) citations = data.citations;
            if (data.chunk) fullAnswer += data.chunk;
            if (data.answer) fullAnswer = data.answer;
            if (data.conversation_id && !state.activeConversationId) {
                state.activeConversationId = data.conversation_id;
            }
        }

        // Final render
        renderFinalAssistantMsg(msgDiv, fullAnswer, citations);

    } catch (error) {
        if (error.name === 'AbortError') {
            // User stopped generation
            renderFinalAssistantMsg(msgDiv, fullAnswer || '*Generation stopped*', citations);
        } else {
            const friendly = sanitizeErrorMessage(error.message, 'Something went wrong. Please try again.');
            const contentEl = msgDiv.querySelector('.message-content');
            contentEl.innerHTML = `<p>${escapeHtml(friendly)}</p>`;
        }
    } finally {
        state.isLoading = false;
        state.abortController = null;
        updateSendButton();
        loadConversations();
    }
}

function renderFinalAssistantMsg(msgDiv, answer, citations) {
    const cleanAnswer = sanitizeBotAnswer(answer);
    const contentEl = msgDiv.querySelector('.message-content');
    contentEl.innerHTML = formatMarkdown(cleanAnswer);

    // Sources
    if (citations.length > 0) {
        msgDiv.insertAdjacentHTML('beforeend', generateSourcesHtml(citations));
        attachSourcesToggle(msgDiv);
    }

    // Action buttons
    msgDiv.insertAdjacentHTML('beforeend', `
        <div class="message-actions">
            <button class="msg-action-btn" title="Copy" onclick="copyMessage(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
            </button>
            <button class="msg-action-btn" title="Regenerate" onclick="regenerateResponse(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 4v6h6M23 20v-6h-6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
    `);
}

function generateSourcesHtml(citations) {
    if (!citations || citations.length === 0) return '';
    return `
        <div class="sources-container">
            <button class="sources-toggle">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" stroke-linecap="round" stroke-linejoin="round"/>
                    <polyline points="14 2 14 8 20 8" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Sources (${citations.length})
                <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
            <div class="sources-panel">
                ${citations.map((c, i) => `
                    <div class="citation-item">
                        <span class="citation-badge">${i + 1}</span>
                        <div class="citation-details">
                            <div class="citation-source">
                                ${escapeHtml(c.document_name || 'Unknown')}
                                ${c.page_number ? ` — Page ${c.page_number}` : ''}
                                ${c.section ? ` — ${escapeHtml(c.section)}` : ''}
                            </div>
                            <div class="citation-snippet">${escapeHtml(c.content_snippet || '')}</div>
                        </div>
                        <span class="citation-score">${((c.relevance_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function attachSourcesToggle(container) {
    const toggle = container.querySelector('.sources-toggle');
    const panel = container.querySelector('.sources-panel');
    if (toggle && panel) {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('expanded');
            panel.classList.toggle('visible');
        });
    }
}

function appendMessage(role, content) {
    const { chatMessagesInner } = el;

    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${role}`;

    if (role === 'user') {
        msgDiv.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
    }

    chatMessagesInner.appendChild(msgDiv);
    scrollToBottom();
}

function appendAssistantMessage(content, citations = []) {
    const { chatMessagesInner } = el;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-assistant';

    let html = `
        <div class="assistant-header">
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 19l-5-5 5-5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M15 5l5 5-5 5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M13 3l-2 18" stroke-linecap="round"/>
                </svg>
            </div>
            <span class="assistant-name">RAG Assistant</span>
        </div>
        <div class="message-content">${formatMarkdown(content)}</div>
    `;

    if (citations.length > 0) {
        html += generateSourcesHtml(citations);
    }

    html += `
        <div class="message-actions">
            <button class="msg-action-btn" title="Copy" onclick="copyMessage(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
            </button>
            <button class="msg-action-btn" title="Regenerate" onclick="regenerateResponse(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M1 4v6h6M23 20v-6h-6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
    `;

    msgDiv.innerHTML = html;
    chatMessagesInner.appendChild(msgDiv);
    attachSourcesToggle(msgDiv);
    scrollToBottom();
}

// ── Regenerate ─────────────────────────────────────────────
function regenerateResponse(btn) {
    if (state.isLoading || !state.lastQuery) return;

    // Remove last assistant message
    const msgDiv = btn.closest('.message-assistant');
    if (msgDiv) msgDiv.remove();

    // Resend
    sendQuery(state.lastQuery);
}

// ── Copy ───────────────────────────────────────────────────
function copyMessage(btn) {
    const msgDiv = btn.closest('.message-assistant');
    const contentEl = msgDiv.querySelector('.message-content');
    const text = contentEl.textContent || contentEl.innerText;

    navigator.clipboard.writeText(text).then(() => {
        // Visual feedback: change icon briefly
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2"><polyline points="20 6 9 17 4 12" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
        setTimeout(() => { btn.innerHTML = originalHtml; }, 1500);
        showToast('Copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

// ── Scroll ─────────────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
    });
}

// ── Utilities ──────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return '';

    if (typeof marked !== 'undefined') {
        try {
            marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });

            let html = marked.parse(text);

            // GitHub-style callout blockquotes
            html = html.replace(
                /<blockquote>\s*<p>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*<br\s*\/?>\s*/gi,
                (_, type) => {
                    const t = type.toUpperCase();
                    const icons = { NOTE: 'ℹ️', TIP: '💡', IMPORTANT: '❗', WARNING: '⚠️', CAUTION: '🚨' };
                    return `<blockquote class="callout callout-${t.toLowerCase()}"><p><strong>${icons[t] || ''} ${t}</strong><br>`;
                }
            );

            html = html.replace(/<blockquote>\s*<p>\s*💡/g, '<blockquote class="callout callout-tip"><p>💡');

            return html;
        } catch (e) {
            console.warn('marked.parse failed:', e);
        }
    }

    // Fallback
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/<p><\/p>/g, '');
    return html;
}

// ── Toast Notifications ────────────────────────────────────
let toastContainer = null;

function showToast(message, type = 'info') {
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ';
    toast.innerHTML = `<span>${icon}</span> ${escapeHtml(message)}`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
