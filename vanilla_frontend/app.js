/**
 * Advanced RAG Assistant — Frontend Application Logic
 * Handles document upload, chat interaction, and API communication.
 */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────────
const state = {
    documents: [],
    isLoading: false,
};

// ── DOM Elements ───────────────────────────────────────────
const elements = {
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    documentList: document.getElementById('documentList'),
    docCount: document.getElementById('docCount'),
    chatMessages: document.getElementById('chatMessages'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    chatForm: document.getElementById('chatForm'),
    queryInput: document.getElementById('queryInput'),
    sendButton: document.getElementById('sendButton'),
    charCount: document.getElementById('charCount'),
    statusBadge: document.getElementById('statusBadge'),
};

// ── Initialization ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initChat();
    loadDocuments();
});

// ── Upload Handling ────────────────────────────────────────
function initUpload() {
    const { uploadZone, fileInput } = elements;

    // Click to browse
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(Array.from(e.target.files));
            fileInput.value = '';
        }
    });

    // Drag & Drop
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
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const doc = await response.json();
        showToast(`${file.name} uploaded successfully!`, 'success');

        // Start polling for processing status
        pollDocumentStatus(doc.id);
        loadDocuments();
    } catch (error) {
        showToast(`Failed to upload ${file.name}: ${error.message}`, 'error');
    }
}

async function pollDocumentStatus(docId) {
    const maxAttempts = 60;
    let attempts = 0;

    const poll = async () => {
        attempts++;
        try {
            const response = await fetch(`${API_BASE}/api/v1/documents/${docId}`);
            if (response.ok) {
                const doc = await response.json();
                if (doc.status === 'completed') {
                    showToast(`${doc.filename} processed: ${doc.chunk_count} chunks`, 'success');
                    loadDocuments();
                    return;
                } else if (doc.status === 'failed') {
                    showToast(`Processing failed for ${doc.filename}`, 'error');
                    loadDocuments();
                    return;
                }
            }
        } catch (e) {
            // Silently retry
        }

        if (attempts < maxAttempts) {
            setTimeout(poll, 3000);
        }
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
        renderDocuments();
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function renderDocuments() {
    const { documentList, docCount } = elements;
    docCount.textContent = state.documents.length;

    if (state.documents.length === 0) {
        documentList.innerHTML = `
            <div style="text-align: center; padding: 32px 16px; color: var(--text-muted); font-size: 0.85rem;">
                No documents yet.<br>Upload files to get started.
            </div>
        `;
        return;
    }

    documentList.innerHTML = state.documents.map(doc => {
        const ext = doc.filename.split('.').pop().toLowerCase();
        const iconClass = getIconClass(ext);
        const size = formatFileSize(doc.file_size);

        return `
            <div class="doc-item" data-id="${doc.id}">
                <div class="doc-icon ${iconClass}">${ext}</div>
                <div class="doc-info">
                    <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="doc-meta">
                        <span>${size}</span>
                        <span>•</span>
                        <span class="doc-status ${doc.status}">${getStatusLabel(doc.status, doc.chunk_count)}</span>
                    </div>
                </div>
                <button class="doc-delete" onclick="deleteDocument('${doc.id}', event)" title="Delete">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
            </div>
        `;
    }).join('');
}

function getIconClass(ext) {
    const map = {
        pdf: 'pdf', txt: 'txt', md: 'md', csv: 'csv',
        xlsx: 'xlsx', docx: 'docx', png: 'img', jpg: 'img', jpeg: 'img',
    };
    return map[ext] || 'txt';
}

function getStatusLabel(status, chunkCount) {
    switch (status) {
        case 'completed': return `✓ ${chunkCount} chunks`;
        case 'processing': return '⟳ Processing...';
        case 'pending': return '◷ Pending';
        case 'failed': return '✗ Failed';
        default: return status;
    }
}

function formatFileSize(bytes) {
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
        loadDocuments();
    } catch (error) {
        showToast('Failed to delete document', 'error');
    }
}

// ── Chat ───────────────────────────────────────────────────
function initChat() {
    const { chatForm, queryInput, sendButton, charCount } = elements;

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';

        const len = queryInput.value.length;
        charCount.textContent = `${len} / 2000`;
        sendButton.disabled = len === 0 || state.isLoading;
    });

    // Submit on Enter (Shift+Enter for new line)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (queryInput.value.trim() && !state.isLoading) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Form submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query || state.isLoading) return;

        await sendQuery(query);
    });
}

async function sendQuery(query) {
    const { queryInput, sendButton, chatMessages, welcomeMessage, charCount } = elements;

    // Hide welcome
    if (welcomeMessage) welcomeMessage.style.display = 'none';

    // Add user message
    appendMessage('user', query);

    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';
    charCount.textContent = '0 / 2000';
    sendButton.disabled = true;
    state.isLoading = true;
    updateStatus('Thinking...', 'processing');

    // Show typing indicator
    const typingId = showTyping();

    try {
        const response = await fetch(`${API_BASE}/api/v1/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        removeTyping(typingId);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await response.json();
        appendMessage('assistant', data.answer, data.citations);

    } catch (error) {
        removeTyping(typingId);
        appendMessage('assistant', `⚠️ Error: ${error.message}`);
    } finally {
        state.isLoading = false;
        sendButton.disabled = queryInput.value.length === 0;
        updateStatus('Ready', 'ready');
    }
}

function appendMessage(role, content, citations = []) {
    const { chatMessages } = elements;

    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${role}`;

    if (role === 'user') {
        msgDiv.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
    } else {
        let html = `<div class="message-content">${formatMarkdown(content)}</div>`;

        if (citations && citations.length > 0) {
            html += `
                <div class="citations-panel">
                    <div class="citations-header">📌 Sources (${citations.length})</div>
                    ${citations.map((c, i) => `
                        <div class="citation-item">
                            <span class="citation-badge">${i + 1}</span>
                            <div class="citation-details">
                                <div class="citation-source">
                                    ${escapeHtml(c.document_name)}
                                    ${c.page_number ? ` — Page ${c.page_number}` : ''}
                                    ${c.section ? ` — ${escapeHtml(c.section)}` : ''}
                                </div>
                                <div class="citation-snippet">${escapeHtml(c.content_snippet)}</div>
                            </div>
                            <span class="citation-score">${(c.relevance_score * 100).toFixed(0)}%</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        msgDiv.innerHTML = html;
    }

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    const { chatMessages } = elements;
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message message-assistant';
    div.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Utilities ──────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Simple markdown-to-HTML conversion
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Inline code
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists
    html = html.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Paragraphs (double newlines)
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Single newlines within paragraphs
    html = html.replace(/\n/g, '<br>');

    // Clean empty paragraphs
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p><br><\/p>/g, '');

    return html;
}

function updateStatus(text, status) {
    const badge = elements.statusBadge;
    const dot = badge.querySelector('.status-dot');
    badge.childNodes[badge.childNodes.length - 1].textContent = ` ${text}`;

    if (status === 'processing') {
        dot.style.background = 'var(--info)';
    } else {
        dot.style.background = 'var(--success)';
    }
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

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
