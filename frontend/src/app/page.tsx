"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = {
  document_id: string;
  document_name: string;
  chunk_id: string;
  page_number?: number;
  section?: string;
  content_snippet: string;
  relevance_score: number;
};

type Message = {
  role: "user" | "bot";
  content: string;
  citations?: Citation[];
  metadata?: any;
};

type UploadedDoc = {
  id: string;
  filename: string;
  status: string;
  chunk_count: number;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<UploadedDoc[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch existing documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(
          data.documents.map((d: any) => ({
            id: d.id,
            filename: d.filename,
            status: d.status,
            chunk_count: d.chunk_count || 0,
          }))
        );
      }
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const files = Array.from(e.target.files);
    setUploadedFiles(files);
    setIsUploading(true);

    const newDocIds: string[] = [];

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("title", file.name);

        const res = await fetch(`${API_BASE}/api/v1/documents/upload`, {
          method: "POST",
          body: formData,
        });

        if (res.ok) {
          const doc = await res.json();
          newDocIds.push(doc.id);
          setDocuments((prev) => [
            ...prev,
            { id: doc.id, filename: doc.filename, status: doc.status, chunk_count: 0 },
          ]);
        } else {
          const err = await res.json().catch(() => ({ detail: "Upload failed" }));
          console.error(`Upload failed for ${file.name}:`, err.detail);
        }
      } catch (error) {
        console.error(`Upload error for ${file.name}:`, error);
      }
    }

    setSelectedDocIds((prev) => [...prev, ...newDocIds]);
    setIsUploading(false);
    setUploadedFiles([]);

    // Refresh document list after a delay (processing happens in background)
    setTimeout(fetchDocuments, 3000);
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    const query = input;
    setInput("");
    setIsLoading(true);

    try {
      const body: any = { query };
      if (selectedDocIds.length > 0) {
        body.document_ids = selectedDocIds;
      }
      if (conversationId) {
        body.conversation_id = conversationId;
      }

      const res = await fetch(`${API_BASE}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const data = await res.json();

        // Save conversation_id for follow-up queries
        if (data.conversation_id && !conversationId) {
          setConversationId(data.conversation_id);
        }

        const botMsg: Message = {
          role: "bot",
          content: data.answer,
          citations: data.citations,
          metadata: data.retrieval_metadata,
        };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        const err = await res.json().catch(() => ({ detail: "Query failed" }));
        setMessages((prev) => [
          ...prev,
          { role: "bot", content: `Error: ${err.detail || "Failed to process query."}` },
        ]);
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: "Error: Could not connect to the backend. Please make sure the server is running." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDocSelection = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const startNewConversation = () => {
    setMessages([]);
    setConversationId(null);
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "completed": return "text-green-400";
      case "processing": return "text-yellow-400";
      case "failed": return "text-red-400";
      default: return "text-slate-400";
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 h-[calc(100vh-80px)] flex gap-6">

      {/* Document Sidebar */}
      <div className="w-72 flex-shrink-0 glass-panel p-4 flex flex-col gap-4 overflow-hidden">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300">Documents</h2>
          <label className="cursor-pointer">
            <span className="glass-button text-xs py-1 px-2">
              {isUploading ? "Uploading..." : "Upload +"}
            </span>
            <input
              type="file"
              multiple
              className="hidden"
              onChange={handleFileUpload}
              disabled={isUploading}
            />
          </label>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2">
          {documents.length === 0 && (
            <p className="text-xs text-slate-500 italic">No documents uploaded yet.</p>
          )}
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => toggleDocSelection(doc.id)}
              className={`p-2 rounded-lg cursor-pointer transition-all text-xs border ${
                selectedDocIds.includes(doc.id)
                  ? "border-blue-500/50 bg-blue-900/20"
                  : "border-slate-700/50 bg-slate-800/30 hover:border-slate-600/50"
              }`}
            >
              <div className="font-medium text-slate-300 truncate">{doc.filename}</div>
              <div className="flex items-center justify-between mt-1">
                <span className={`${statusColor(doc.status)} text-[10px] uppercase tracking-wider`}>
                  {doc.status}
                </span>
                {doc.chunk_count > 0 && (
                  <span className="text-slate-500 text-[10px]">{doc.chunk_count} chunks</span>
                )}
              </div>
            </div>
          ))}
        </div>

        <button
          onClick={startNewConversation}
          className="glass-button text-xs py-2 w-full"
        >
          New Conversation
        </button>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col gap-4">
        {/* File upload indicator */}
        {uploadedFiles.length > 0 && (
          <div className="glass-panel p-3 flex items-center gap-2">
            {uploadedFiles.map((f) => (
              <span key={f.name} className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300">
                {f.name}
              </span>
            ))}
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 glass-panel p-6 flex flex-col overflow-hidden relative">
          <div className="flex-1 overflow-y-auto space-y-6 pb-4 pr-2">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4 opacity-50">
                <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center shadow-inner">
                  🤖
                </div>
                <p>Ask anything. I will search the documents and verify the answer.</p>
                {selectedDocIds.length > 0 && (
                  <p className="text-xs">
                    {selectedDocIds.length} document{selectedDocIds.length > 1 ? "s" : ""} selected for search.
                  </p>
                )}
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`p-4 ${msg.role === "user" ? "message-user" : "message-bot"}`}>
                <div className="text-xs font-semibold mb-2 opacity-60 uppercase tracking-wider">
                  {msg.role === "user" ? "You" : "Agent"}
                </div>
                <div className="text-sm markdown-body whitespace-pre-wrap">{msg.content}</div>

                {msg.metadata && (
                  <div className="mt-4 pt-4 border-t border-slate-700/50 flex flex-wrap gap-2">
                    <span className="text-[10px] bg-blue-900/40 text-blue-300 px-2 py-1 rounded border border-blue-800">
                      Route: {msg.metadata.route}
                    </span>
                    {msg.metadata.steps && (
                      <span className="text-[10px] bg-purple-900/40 text-purple-300 px-2 py-1 rounded border border-purple-800">
                        Steps: {msg.metadata.steps}
                      </span>
                    )}
                    {msg.metadata.chunks_retrieved !== undefined && (
                      <span className="text-[10px] bg-emerald-900/40 text-emerald-300 px-2 py-1 rounded border border-emerald-800">
                        Chunks: {msg.metadata.chunks_retrieved}
                      </span>
                    )}
                    {msg.metadata.conversation_context_used && (
                      <span className="text-[10px] bg-amber-900/40 text-amber-300 px-2 py-1 rounded border border-amber-800">
                        Memory Active
                      </span>
                    )}
                  </div>
                )}

                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 flex flex-col gap-2">
                    <span className="text-xs font-semibold text-slate-400">Sources:</span>
                    {msg.citations.map((c, i) => (
                      <div key={i} className="text-xs bg-slate-900/50 p-2 rounded border border-slate-700">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-semibold text-slate-300">{c.document_name}</span>
                          <div className="flex items-center gap-2">
                            {c.page_number && (
                              <span className="text-slate-500">p. {c.page_number}</span>
                            )}
                            <span className="text-slate-500">
                              {Math.round(c.relevance_score * 100)}%
                            </span>
                          </div>
                        </div>
                        <span className="text-slate-400 italic">"{c.content_snippet}"</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="message-bot p-4 animate-pulse">
                <div className="flex gap-2 items-center text-sm text-slate-400">
                  <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
                  Agent is thinking, planning, and searching...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <form onSubmit={handleSend} className="mt-4 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="glass-input pr-24"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="absolute right-2 top-2 glass-button px-4 py-1.5 text-sm"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
