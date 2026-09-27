"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";

type Message = {
  role: "assistant" | "user";
  text: string;
  sources?: string[];
};

const starterMessages: Message[] = [
  {
    role: "assistant",
    text: "Hi there. Upload a PDF and I’ll help you find the signal inside it. Ask me anything about the document once it’s indexed.",
  },
];

const demoAnswer =
  "I’m ready to search your document. The FastAPI service will return a grounded answer here once it is running on port 8000.";

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("No document uploaded");
  const [uploadError, setUploadError] = useState("");
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [question, setQuestion] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);

  async function uploadFile(file: File) {
    if (file.type !== "application/pdf") {
      setMessages((current) => [...current, { role: "assistant", text: "Please choose a PDF file so I can index it." }]);
      return;
    }

    setIsUploading(true);
    setUploadError("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/upload", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "The document could not be indexed.");
      }
      setFileName(file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not connect to the document service.";
      setUploadError(message);
      setMessages((current) => [...current, { role: "assistant", text: `Could not upload ${file.name}: ${message}` }]);
    } finally {
      setIsUploading(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) uploadFile(file);
    event.target.value = "";
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isSending) return;

    setMessages((current) => [...current, { role: "user", text: trimmedQuestion }]);
    setQuestion("");
    setIsSending(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion }),
      });
      if (!response.ok) throw new Error("API unavailable");
      const data = await response.json();
      setMessages((current) => [...current, { role: "assistant", text: data.answer, sources: ["Uploaded document"] }]);
    } catch {
      setMessages((current) => [...current, { role: "assistant", text: demoAnswer }]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">✦</span><span>context<span className="brand-dot">.</span></span></div>
        <div className="topbar-status"><span className="status-dot" /> Local workspace <span className="topbar-divider" /> <span className="avatar">VK</span></div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading"><span>YOUR LIBRARY</span><button className="icon-button" aria-label="Add a document" onClick={() => inputRef.current?.click()}>＋</button></div>
          <button className="new-chat" onClick={() => setMessages(starterMessages)}><span>＋</span> New conversation</button>
          <div className="library-list">
            <p className="list-label">DOCUMENTS <span>{fileName === "No document uploaded" ? "0" : "1"}</span></p>
            {fileName !== "No document uploaded" ? (
              <div className="document-item active"><span className="pdf-icon">PDF</span><span className="document-copy"><strong>{fileName}</strong><small>{isUploading ? "Indexing another document..." : uploadError ? "Previous document still active" : "Ready to chat"}</small></span><span className="more">•••</span></div>
            ) : <div className="empty-library">Your documents will appear here.</div>}
          </div>
          <div className="sidebar-footer"><div className="plan-row"><span>FREE PLAN</span><span>0 / 3 docs</span></div><div className="plan-track"><span /></div><button className="upgrade-button">Upgrade workspace <span>↗</span></button></div>
        </aside>

        <section className="chat-panel">
          <div className="chat-header"><div><p className="eyebrow">DOCUMENT Q&A</p><h1>Ask your documents.</h1></div><div className="header-actions"><button className="quiet-button">⌘ Share</button><button className="round-button" aria-label="More options">•••</button></div></div>

          <div className="conversation">
            {messages.map((message, index) => <div className={`message-row ${message.role}`} key={`${message.role}-${index}`}><div className="message-avatar">{message.role === "assistant" ? "✦" : "VK"}</div><div className="message-content"><span className="message-author">{message.role === "assistant" ? "Context AI" : "You"}</span><p>{message.text}</p>{message.sources && <div className="source-list">{message.sources.map((source) => <span className="source-chip" key={source}>⌁ {source}</span>)}</div>}</div></div>)}
            {isSending && <div className="message-row assistant"><div className="message-avatar">✦</div><div className="message-content"><span className="message-author">Context AI</span><p className="typing">Thinking<span>.</span><span>.</span><span>.</span></p></div></div>}
          </div>

          <div className="composer-wrap"><div className="suggestions"><button onClick={() => setQuestion("What is this document about?")}>What is this document about?</button><button onClick={() => setQuestion("Summarize the key points")}>Summarize the key points</button></div><form className="composer" onSubmit={handleSubmit}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask anything about your documents..." rows={1} /><div className="composer-controls"><button type="button" className="attach-button" aria-label="Attach a PDF" onClick={() => inputRef.current?.click()}>＋</button><span>Press Enter to send</span><button className="send-button" aria-label="Send message" disabled={!question.trim() || isSending}>↑</button></div></form><p className="disclaimer">Context AI can make mistakes. Check important information against the original document.</p></div>
        </section>

        <input ref={inputRef} className="visually-hidden" type="file" accept="application/pdf" onChange={handleFileChange} />
        <aside className="right-rail"><div className="upload-card"><div className="upload-icon">↥</div><h2>Bring your knowledge.</h2><p>Upload a PDF and start asking questions in seconds.</p><button onClick={() => inputRef.current?.click()}>{isUploading ? "Indexing..." : "Upload a PDF"}</button><span className="upload-note">PDF up to 20MB</span></div><div className="tip-card"><span className="tip-kicker">QUICK TIP</span><p>Ask specific questions for more useful answers. Context AI cites the source whenever it can.</p></div></aside>
      </main>
    </div>
  );
}
