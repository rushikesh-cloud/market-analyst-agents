"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

type TabKey = "ingest" | "web" | "technical" | "fundamental" | "supervisor" | "chat";

type Source = { title?: string; url?: string; content?: string; score?: number };

type IngestedDoc = {
  id: string;
  company: string;
  ticker?: string | null;
  year?: string | null;
  doc_type?: string;
  collection_name: string;
  chunks_stored: number;
  source_path: string;
};

type ChatSession = {
  id: string;
  title: string;
  symbol?: string | null;
  company?: string | null;
  created_at: string;
  updated_at: string;
};

type ChatMessage = {
  id: number;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

const tabs: { key: TabKey; label: string }[] = [
  { key: "ingest", label: "PDF Ingestion" },
  { key: "web", label: "Web Search Agent" },
  { key: "technical", label: "Technical Agent" },
  { key: "fundamental", label: "Fundamental Agent" },
  { key: "supervisor", label: "Supervisor Agent" },
  { key: "chat", label: "Supervisor Chat" },
];

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.message || data?.detail || data?.error || "Request failed");
  }
  return data;
}

function mdSources(sources: Source[]): string {
  if (!sources.length) return "No sources returned.";
  return sources
    .map((s, i) => `${i + 1}. [${s.title || s.url || "Source"}](${s.url || "#"})${s.score != null ? ` (score: ${s.score})` : ""}`)
    .join("\n");
}

export default function HomePage() {
  const [tab, setTab] = useState<TabKey>("ingest");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [docs, setDocs] = useState<IngestedDoc[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("");

  const [ingestForm, setIngestForm] = useState({ company: "APPLE", ticker: "AAPL", year: "2025", collection: "fundamental_docs" });
  const [ingestFile, setIngestFile] = useState<File | null>(null);
  const [ingestResult, setIngestResult] = useState<any>(null);

  const [webQuery, setWebQuery] = useState("AAPL latest earnings and guidance");
  const [webResult, setWebResult] = useState<{ query: string; answer: string; sources: Source[] } | null>(null);

  const [techForm, setTechForm] = useState({ symbol: "AAPL", period: "3mo", interval: "1d" });
  const [techResult, setTechResult] = useState<any>(null);

  const [fundForm, setFundForm] = useState({ company: "APPLE", question: "What are the key balance sheet risks?", mode: "qa", collection: "fundamental_docs", top_k: 8 });
  const [fundResult, setFundResult] = useState<any>(null);

  const [supForm, setSupForm] = useState({
    symbol: "AAPL",
    company: "APPLE",
    fundamental_question: "Assess fundamentals and near-term risks",
    news_query: "AAPL latest product and regulation news",
    technical_period: "3mo",
    technical_interval: "1d",
    collection: "fundamental_docs",
    top_k: 8,
  });
  const [supResult, setSupResult] = useState<any>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [chatSessionId, setChatSessionId] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("Analyze AAPL for next 6 months.");
  const [newSessionTitle, setNewSessionTitle] = useState("AAPL Strategy Session");
  const [chatContext, setChatContext] = useState({ ticker: "AAPL", company: "APPLE" });

  async function refreshDocs() {
    const data = await apiJson<IngestedDoc[]>("/agents/ingested-docs");
    setDocs(data);
    if (!selectedDocId && data.length) {
      setSelectedDocId(data[0].id);
    }
  }

  async function refreshChatSessions() {
    const data = await apiJson<ChatSession[]>("/agents/supervisor-chat/sessions");
    setChatSessions(data);
    if (!chatSessionId && data.length) {
      setChatSessionId(data[0].id);
    }
  }

  async function refreshChatHistory(sessionId: string) {
    if (!sessionId) {
      setChatHistory([]);
      return;
    }
    const data = await apiJson<ChatMessage[]>(`/agents/supervisor-chat/sessions/${sessionId}/messages`);
    setChatHistory(data);
  }

  useEffect(() => {
    refreshDocs().catch(() => {});
    refreshChatSessions().catch(() => {});
  }, []);

  useEffect(() => {
    refreshChatHistory(chatSessionId).catch(() => {});
  }, [chatSessionId]);

  const selectedDoc = useMemo(() => docs.find((d) => d.id === selectedDocId), [docs, selectedDocId]);

  const applySelectedDoc = () => {
    if (!selectedDoc) return;
    setIngestForm((p) => ({ ...p, company: selectedDoc.company, ticker: selectedDoc.ticker || "", year: selectedDoc.year || "", collection: selectedDoc.collection_name }));
    setTechForm((p) => ({ ...p, symbol: selectedDoc.ticker || p.symbol || selectedDoc.company }));
    setFundForm((p) => ({ ...p, company: selectedDoc.company, collection: selectedDoc.collection_name }));
    setSupForm((p) => ({ ...p, symbol: selectedDoc.ticker || p.symbol, company: selectedDoc.company, collection: selectedDoc.collection_name }));
    setWebQuery(`${selectedDoc.ticker || selectedDoc.company} latest company news catalysts and risks`);
    setChatContext({
      ticker: (selectedDoc.ticker || selectedDoc.company || "").toUpperCase(),
      company: (selectedDoc.company || selectedDoc.ticker || "").toUpperCase(),
    });
  };

  const run = async (fn: () => Promise<void>) => {
    try {
      setLoading(true);
      setError(null);
      await fn();
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const ingestMd = ingestResult
    ? `### Ingestion Completed\n- **Company:** ${ingestResult.company}\n- **Ticker:** ${ingestResult.ticker || "-"}\n- **Collection:** ${ingestResult.collection_name}\n- **Chunks Stored:** ${ingestResult.chunks_stored}\n- **Source:** ${ingestResult.source_path}`
    : "No ingestion response yet.";

  const webMd = webResult
    ? `### Query\n${webResult.query}\n\n### Answer\n${webResult.answer}\n\n### Sources\n${mdSources(webResult.sources)}`
    : "No web-search response yet.";

  const techMd = techResult
    ? `### Technical Summary (${techResult.symbol})\n${techResult.summary}\n\n### Latest Values\n${Object.entries(techResult.latest_values || {}).map(([k, v]) => `- **${k}:** ${v ?? "n/a"}`).join("\n")}`
    : "No technical response yet.";

  const fundMd = fundResult
    ? `### Fundamental (${fundResult.company})\n${fundResult.answer}\n\n### Sources\n${(fundResult.sources || []).map((s: any, i: number) => `${i + 1}. ${s.source_path || "-"} (year: ${s.year || "-"}, ticker: ${s.ticker || "-"})`).join("\n") || "No sources."}`
    : "No fundamental response yet.";

  const supMd = supResult
    ? `### Final Thesis\n${supResult.synthesis.final_thesis}\n\n### Rating\n- **Score (6m):** ${supResult.synthesis.investment_rating_6m ?? "n/a"}\n- **Stance:** ${supResult.synthesis.stance}\n\n### Technical\n${supResult.synthesis.technical_section}\n\n### Fundamental\n${supResult.synthesis.fundamental_section}\n\n### News\n${supResult.synthesis.news_section}\n\n### Risks\n${(supResult.synthesis.risks || []).map((r: string) => `- ${r}`).join("\n")}`
    : "No supervisor response yet.";

  const chatSession = chatSessions.find((s) => s.id === chatSessionId) || null;

  const renderMain = () => {
    if (tab === "ingest") {
      return (
        <div className="panel">
          <h2>Upload PDF to Vector DB</h2>
          <form onSubmit={(e) => run(async () => {
            e.preventDefault();
            if (!ingestFile) throw new Error("Select a PDF file first.");
            const fd = new FormData();
            fd.append("company", ingestForm.company);
            fd.append("ticker", ingestForm.ticker);
            fd.append("year", ingestForm.year);
            fd.append("collection", ingestForm.collection);
            fd.append("file", ingestFile);
            const response = await fetch(`${API_BASE}/agents/ingest`, { method: "POST", body: fd });
            const data = await response.json();
            if (!response.ok) throw new Error(data?.message || data?.detail || "Ingestion failed");
            setIngestResult(data);
            await refreshDocs();
          })}>
            <div className="grid2">
              <div><div className="label">Company</div><input className="input" value={ingestForm.company} onChange={(e) => setIngestForm((p) => ({ ...p, company: e.target.value }))} /></div>
              <div><div className="label">Ticker</div><input className="input" value={ingestForm.ticker} onChange={(e) => setIngestForm((p) => ({ ...p, ticker: e.target.value }))} /></div>
              <div><div className="label">Year</div><input className="input" value={ingestForm.year} onChange={(e) => setIngestForm((p) => ({ ...p, year: e.target.value }))} /></div>
              <div><div className="label">Collection</div><input className="input" value={ingestForm.collection} onChange={(e) => setIngestForm((p) => ({ ...p, collection: e.target.value }))} /></div>
            </div>
            <div style={{ marginTop: 12 }}><div className="label">PDF File</div><input className="fileInput" type="file" accept=".pdf" onChange={(e) => setIngestFile(e.target.files?.[0] || null)} /></div>
            <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Uploading..." : "Upload + Ingest"}</button></div>
          </form>
          <hr className="hr" />
          <div className="resultBlock"><h3>Ingestion Result</h3><ReactMarkdown>{ingestMd}</ReactMarkdown></div>
          <div className="resultBlock">
            <h3>All Ingested Documents</h3>
            <div className="tableWrap">
              <table className="table">
                <thead><tr><th>Collection</th><th>Company</th><th>Ticker</th><th>Year</th><th>Type</th><th>Chunks</th><th>Source Path</th><th>Action</th></tr></thead>
                <tbody>
                  {docs.map((d) => (
                    <tr key={d.id}>
                      <td>{d.collection_name}</td><td>{d.company}</td><td>{d.ticker || "-"}</td><td>{d.year || "-"}</td><td>{d.doc_type || "-"}</td><td>{d.chunks_stored}</td><td>{d.source_path}</td>
                      <td><button className="btn btnDanger" onClick={() => run(async () => {
                        await fetch(`${API_BASE}/agents/ingested-docs/${d.id}`, { method: "DELETE" });
                        await refreshDocs();
                      })}>Delete</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      );
    }

    if (tab === "web") {
      return (
        <div className="panel">
          <h2>Web Search Agent</h2>
          <form onSubmit={(e) => run(async () => { e.preventDefault(); setWebResult(await apiJson("/agents/web-search", { method: "POST", body: JSON.stringify({ query: webQuery }) })); })}>
            <div className="label">Market Query</div>
            <textarea className="textarea" value={webQuery} onChange={(e) => setWebQuery(e.target.value)} />
            <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Running..." : "Run Web Search"}</button></div>
          </form>
          <div className="resultBlock"><h3>Result</h3><ReactMarkdown>{webMd}</ReactMarkdown></div>
        </div>
      );
    }

    if (tab === "technical") {
      return (
        <div className="panel">
          <h2>Technical Agent</h2>
          <form onSubmit={(e) => run(async () => { e.preventDefault(); setTechResult(await apiJson("/agents/technical", { method: "POST", body: JSON.stringify(techForm) })); })}>
            <div className="grid2">
              <div><div className="label">Symbol</div><input className="input" value={techForm.symbol} onChange={(e) => setTechForm((p) => ({ ...p, symbol: e.target.value }))} /></div>
              <div><div className="label">Period</div><input className="input" value={techForm.period} onChange={(e) => setTechForm((p) => ({ ...p, period: e.target.value }))} /></div>
              <div><div className="label">Interval</div><input className="input" value={techForm.interval} onChange={(e) => setTechForm((p) => ({ ...p, interval: e.target.value }))} /></div>
            </div>
            <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Running..." : "Run Technical"}</button></div>
          </form>
          <div className="resultBlock"><h3>Result</h3><ReactMarkdown>{techMd}</ReactMarkdown></div>
        </div>
      );
    }

    if (tab === "fundamental") {
      return (
        <div className="panel">
          <h2>Fundamental Agent</h2>
          <form onSubmit={(e) => run(async () => { e.preventDefault(); setFundResult(await apiJson("/agents/fundamental", { method: "POST", body: JSON.stringify(fundForm) })); })}>
            <div className="grid2">
              <div><div className="label">Company</div><input className="input" value={fundForm.company} onChange={(e) => setFundForm((p) => ({ ...p, company: e.target.value }))} /></div>
              <div><div className="label">Collection</div><input className="input" value={fundForm.collection} onChange={(e) => setFundForm((p) => ({ ...p, collection: e.target.value }))} /></div>
              <div><div className="label">Mode</div><select className="select" value={fundForm.mode} onChange={(e) => setFundForm((p) => ({ ...p, mode: e.target.value }))}><option value="auto">auto</option><option value="general">general</option><option value="qa">qa</option></select></div>
              <div><div className="label">Top K</div><input className="input" type="number" value={fundForm.top_k} onChange={(e) => setFundForm((p) => ({ ...p, top_k: Number(e.target.value) }))} /></div>
            </div>
            <div style={{ marginTop: 10 }}><div className="label">Question</div><textarea className="textarea" value={fundForm.question} onChange={(e) => setFundForm((p) => ({ ...p, question: e.target.value }))} /></div>
            <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Running..." : "Run Fundamental"}</button></div>
          </form>
          <div className="resultBlock"><h3>Result</h3><ReactMarkdown>{fundMd}</ReactMarkdown></div>
        </div>
      );
    }

    if (tab === "supervisor") {
      return (
      <div className="panel">
        <h2>Supervisor Agent</h2>
        <form onSubmit={(e) => run(async () => { e.preventDefault(); setSupResult(await apiJson("/agents/supervisor", { method: "POST", body: JSON.stringify(supForm) })); })}>
          <div className="grid2">
            <div><div className="label">Symbol</div><input className="input" value={supForm.symbol} onChange={(e) => setSupForm((p) => ({ ...p, symbol: e.target.value }))} /></div>
            <div><div className="label">Company</div><input className="input" value={supForm.company} onChange={(e) => setSupForm((p) => ({ ...p, company: e.target.value }))} /></div>
            <div><div className="label">Technical Period</div><input className="input" value={supForm.technical_period} onChange={(e) => setSupForm((p) => ({ ...p, technical_period: e.target.value }))} /></div>
            <div><div className="label">Technical Interval</div><input className="input" value={supForm.technical_interval} onChange={(e) => setSupForm((p) => ({ ...p, technical_interval: e.target.value }))} /></div>
            <div><div className="label">Collection</div><input className="input" value={supForm.collection} onChange={(e) => setSupForm((p) => ({ ...p, collection: e.target.value }))} /></div>
            <div><div className="label">Top K</div><input className="input" type="number" value={supForm.top_k} onChange={(e) => setSupForm((p) => ({ ...p, top_k: Number(e.target.value) }))} /></div>
          </div>
          <div style={{ marginTop: 10 }}><div className="label">Fundamental Focus</div><textarea className="textarea" value={supForm.fundamental_question} onChange={(e) => setSupForm((p) => ({ ...p, fundamental_question: e.target.value }))} /></div>
          <div style={{ marginTop: 10 }}><div className="label">News Focus</div><textarea className="textarea" value={supForm.news_query} onChange={(e) => setSupForm((p) => ({ ...p, news_query: e.target.value }))} /></div>
          <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Running..." : "Run Supervisor"}</button></div>
        </form>
        <div className="resultBlock"><h3>Final Synthesis</h3><ReactMarkdown>{supMd}</ReactMarkdown></div>
        {supResult?.news?.sources?.length ? (
          <div className="resultBlock"><h3>Supervisor News Sources</h3><ReactMarkdown>{mdSources(supResult.news.sources)}</ReactMarkdown></div>
        ) : null}
      </div>
      );
    }

    return (
      <div className="panel">
        <h2>Supervisor Chat</h2>
        <div className="grid2">
          <div>
            <div className="label">New Session Title</div>
            <input className="input" value={newSessionTitle} onChange={(e) => setNewSessionTitle(e.target.value)} />
          </div>
          <div style={{ display: "flex", alignItems: "end" }}>
            <button className="btn" onClick={() => run(async () => {
              const created = await apiJson<ChatSession>("/agents/supervisor-chat/sessions", {
                method: "POST",
                body: JSON.stringify({
                  title: newSessionTitle,
                  symbol: chatContext.ticker,
                  company: chatContext.company,
                }),
              });
              await refreshChatSessions();
              setChatSessionId(created.id);
            })}>Create Session</button>
          </div>
        </div>

        <div style={{ marginTop: 12 }} className="grid2">
          <div>
            <div className="label">Session</div>
            <select className="select" value={chatSessionId} onChange={(e) => setChatSessionId(e.target.value)}>
              {chatSessions.map((s) => (
                <option key={s.id} value={s.id}>{`${s.title} (${s.symbol || s.company || "-"})`}</option>
              ))}
            </select>
          </div>
          <div>
            <div className="label">Session Context</div>
            <div className="input" style={{ height: 48 }}>{chatSession ? `${chatSession.symbol || "-"} / ${chatSession.company || "-"}` : "No session selected"}</div>
          </div>
        </div>
        <div style={{ marginTop: 12 }} className="grid2">
          <div>
            <div className="label">Chat Ticker</div>
            <input className="input" value={chatContext.ticker} onChange={(e) => setChatContext((p) => ({ ...p, ticker: e.target.value.toUpperCase() }))} />
          </div>
          <div>
            <div className="label">Chat Company</div>
            <input className="input" value={chatContext.company} onChange={(e) => setChatContext((p) => ({ ...p, company: e.target.value.toUpperCase() }))} />
          </div>
        </div>

        <form onSubmit={(e) => run(async () => {
          e.preventDefault();
          if (!chatSessionId) throw new Error("Create or select a chat session first.");
          await apiJson("/agents/supervisor-chat/message", {
            method: "POST",
            body: JSON.stringify({
              session_id: chatSessionId,
              message: chatInput,
              symbol: chatContext.ticker,
              company: chatContext.company,
            }),
          });
          setChatInput("");
          await refreshChatHistory(chatSessionId);
          await refreshChatSessions();
        })}>
          <div style={{ marginTop: 12 }}><div className="label">Message</div><textarea className="textarea" value={chatInput} onChange={(e) => setChatInput(e.target.value)} /></div>
          <div style={{ marginTop: 12 }}><button className="btn" disabled={loading}>{loading ? "Sending..." : "Send"}</button></div>
        </form>

        <div className="resultBlock">
          <h3>Chat History</h3>
          {chatHistory.length ? chatHistory.map((m) => (
            <div key={m.id} style={{ borderTop: "1px dashed #d7d0bf", paddingTop: 10, marginTop: 10 }}>
              <div className="codePill">{m.role.toUpperCase()}</div>
              <div style={{ marginTop: 8 }}>
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>
            </div>
          )) : <p>No chat messages yet.</p>}
        </div>
      </div>
    );
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <h1 className="brand">Market Analyst</h1>
        <p className="subtitle">All agents + vector document controls</p>

        {tabs.map((t) => (
          <button key={t.key} className={`navBtn ${tab === t.key ? "active" : ""}`} onClick={() => setTab(t.key)}>{t.label}</button>
        ))}

        <div className="sideSectionTitle">Ingested Documents</div>
        <select className="select" value={selectedDocId} onChange={(e) => setSelectedDocId(e.target.value)}>
          {docs.map((d) => (
            <option key={d.id} value={d.id}>{`[${d.ticker || d.company}] ${d.year || "-"} - ${d.source_path}`}</option>
          ))}
        </select>
        <div style={{ marginTop: 8 }}>
          <button className="btn" style={{ width: "100%" }} onClick={applySelectedDoc}>Use Selected Doc In Agent Forms</button>
        </div>

        <div style={{ marginTop: 18 }}>
          <span className="codePill">API base: {API_BASE}</span>
        </div>
      </aside>

      <section className="content">
        {error ? <div className="error">{error}</div> : null}
        {renderMain()}
      </section>
    </main>
  );
}
