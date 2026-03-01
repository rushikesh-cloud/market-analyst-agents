"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type TabId = "ingest" | "web" | "technical" | "fundamental" | "supervisor";
type SourceDoc = { company?: string | null; ticker?: string | null; year?: string | null; doc_type?: string | null; source_path?: string | null; chunk_index?: number | null };
type TechnicalResponse = { symbol: string; image_path: string; summary: string; latest_values: Record<string, number> };
type FundamentalResponse = { mode: string; company: string; answer: string; sources: SourceDoc[] };
type SupervisorResponse = { symbol: string; company: string; technical: TechnicalResponse; fundamental: FundamentalResponse; news: { query: string; answer: string }; synthesis: { investment_rating_6m?: number | null; stance: string; technical_section: string; fundamental_section: string; news_section: string; risks: string[]; final_thesis: string } };
type IngestionResponse = { company: string; ticker?: string | null; source_path: string; chunks_stored: number; collection_name: string; markdown_path?: string | null };
type IngestedDoc = { collection_name: string; source_path: string; company?: string | null; ticker?: string | null; doc_type?: string | null; year?: string | null; chunks_stored: number };
type StreamEvent = { event: string; data: unknown };
type InvokeMode = "direct" | "stream";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";
const tabs: Array<{ id: TabId; label: string }> = [
  { id: "ingest", label: "PDF Ingestion" },
  { id: "web", label: "Web Search Agent" },
  { id: "technical", label: "Technical Agent" },
  { id: "fundamental", label: "Fundamental Agent" },
  { id: "supervisor", label: "Supervisor Agent" },
];

async function req<T>(path: string, method: "GET" | "POST" | "DELETE" = "GET", payload?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.detail ? JSON.stringify(data.detail) : `Request failed (${res.status})`);
  return data as T;
}

async function runNdjsonStream(
  path: string,
  payload: unknown,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ? JSON.stringify(data.detail) : `Request failed (${res.status})`);
  }
  if (!res.body) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed) as StreamEvent);
      } catch {
        onEvent({ event: "stream", data: trimmed });
      }
    }
  }
  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()) as StreamEvent);
    } catch {
      onEvent({ event: "stream", data: buffer.trim() });
    }
  }
}

function keyOf(doc: IngestedDoc): string {
  return `${doc.collection_name}::${doc.source_path}`;
}

function chartUrl(path: string): string | null {
  const v = path.replace(/\\/g, "/");
  const i = v.indexOf("data/");
  return i === -1 ? null : `${API_BASE}/static/${v.slice(i + 5)}`;
}

function synthesisMd(s: SupervisorResponse["synthesis"]): string {
  return [
    "## Final Investment View",
    `- **Stance:** ${s.stance || "N/A"}`,
    `- **6M Rating:** ${s.investment_rating_6m ?? "N/A"} / 10`,
    "",
    "## Technical",
    s.technical_section || "_No technical section provided._",
    "",
    "## Fundamental",
    s.fundamental_section || "_No fundamental section provided._",
    "",
    "## News",
    s.news_section || "_No news section provided._",
    "",
    "## Risks",
    ...(s.risks?.length ? s.risks.map((r) => `- ${r}`) : ["- No explicit risks returned"]),
    "",
    "## Final Thesis",
    s.final_thesis || "_No final thesis provided._",
  ].join("\n");
}

function webAnswer(payload: unknown): string {
  const messages = (payload as { messages?: Array<{ content?: unknown }> } | null)?.messages;
  if (!Array.isArray(messages)) return "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const c = messages[i]?.content;
    if (typeof c === "string" && c.trim()) return c;
  }
  return "";
}

export default function HomePage() {
  const [tab, setTab] = useState<TabId>("ingest");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [ingCompany, setIngCompany] = useState("APPLE");
  const [ingTicker, setIngTicker] = useState("AAPL");
  const [ingYear, setIngYear] = useState("");
  const [ingCollection, setIngCollection] = useState("fundamental_docs");
  const [ingFile, setIngFile] = useState<File | null>(null);
  const [ingRes, setIngRes] = useState<IngestionResponse | null>(null);

  const [docs, setDocs] = useState<IngestedDoc[]>([]);
  const [selectedDocId, setSelectedDocId] = useState("");

  const [webQuery, setWebQuery] = useState("Apple latest company news catalysts risks");
  const [webRes, setWebRes] = useState<unknown>(null);
  const [webEvents, setWebEvents] = useState<StreamEvent[]>([]);
  const [webMode, setWebMode] = useState<InvokeMode>("stream");

  const [sym, setSym] = useState("AAPL");
  const [period, setPeriod] = useState("3mo");
  const [interval, setInterval] = useState("1d");
  const [techRes, setTechRes] = useState<TechnicalResponse | null>(null);
  const [techEvents, setTechEvents] = useState<StreamEvent[]>([]);
  const [techMode, setTechMode] = useState<InvokeMode>("stream");

  const [fundCompany, setFundCompany] = useState("APPLE");
  const [fundMode, setFundMode] = useState<"auto" | "general" | "qa">("auto");
  const [fundQ, setFundQ] = useState("Summarize revenue, margin, cash flow strength and key risks.");
  const [fundTopK, setFundTopK] = useState(8);
  const [fundRes, setFundRes] = useState<FundamentalResponse | null>(null);
  const [fundEvents, setFundEvents] = useState<StreamEvent[]>([]);
  const [fundModeType, setFundModeType] = useState<InvokeMode>("stream");

  const [supSym, setSupSym] = useState("AAPL");
  const [supCompany, setSupCompany] = useState("APPLE");
  const [supFQ, setSupFQ] = useState("Evaluate financial strength and key downside risks.");
  const [supNQ, setSupNQ] = useState("Apple latest company news catalysts risks");
  const [supRes, setSupRes] = useState<SupervisorResponse | null>(null);
  const [supEvents, setSupEvents] = useState<StreamEvent[]>([]);
  const [supMode, setSupMode] = useState<InvokeMode>("stream");

  const selectedDoc = useMemo(() => docs.find((d) => keyOf(d) === selectedDocId) ?? null, [docs, selectedDocId]);
  const fundamentalDocOptions = useMemo(() => {
    const seen = new Set<string>();
    return docs
      .map((d) => {
        const value = (d.company || d.ticker || "").toUpperCase();
        if (!value || seen.has(value)) return null;
        seen.add(value);
        return {
          value,
          label: `[${d.ticker || value}] ${d.company || value}${d.year ? ` (${d.year})` : ""}`,
        };
      })
      .filter((v): v is { value: string; label: string } => Boolean(v));
  }, [docs]);
  const supervisorCompanyOptions = useMemo(() => {
    const seen = new Set<string>();
    return docs
      .map((d) => (d.company || d.ticker || "").toUpperCase())
      .filter((v) => {
        if (!v || seen.has(v)) return false;
        seen.add(v);
        return true;
      });
  }, [docs]);
  const supMd = useMemo(() => (supRes ? synthesisMd(supRes.synthesis) : ""), [supRes]);
  const webMd = useMemo(() => webAnswer(webRes), [webRes]);

  async function refreshDocs(collection = "fundamental_docs") {
    const data = await req<{ items: IngestedDoc[] }>(`/documents/ingested?collection=${encodeURIComponent(collection)}`);
    setDocs(data.items);
    if (!selectedDocId && data.items.length) setSelectedDocId(keyOf(data.items[0]));
  }

  useEffect(() => {
    refreshDocs().catch((e) => setError(e instanceof Error ? e.message : "Unable to fetch docs"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!fundamentalDocOptions.length) return;
    if (!fundamentalDocOptions.some((opt) => opt.value === fundCompany)) {
      setFundCompany(fundamentalDocOptions[0].value);
    }
  }, [fundCompany, fundamentalDocOptions]);

  useEffect(() => {
    if (supervisorCompanyOptions.length && !supervisorCompanyOptions.includes(supCompany)) {
      setSupCompany(supervisorCompanyOptions[0]);
    }
  }, [supCompany, supervisorCompanyOptions]);

  function applyDoc() {
    if (!selectedDoc) return;
    const ticker = (selectedDoc.ticker || selectedDoc.company || "").toUpperCase();
    const company = (selectedDoc.company || selectedDoc.ticker || "").toUpperCase();
    if (ticker) {
      setSym(ticker);
      setSupSym(ticker);
      setIngTicker(ticker);
    }
    if (company) {
      setFundCompany(company);
      setSupCompany(company);
      setIngCompany(company);
      setWebQuery(`${company} latest company news catalysts risks`);
    }
  }

  async function onIngest(e: FormEvent) {
    e.preventDefault();
    if (!ingFile) return setError("Please attach a PDF file.");
    setLoading(true); setError(null); setIngRes(null);
    try {
      const f = new FormData();
      f.append("company", ingCompany.trim().toUpperCase());
      f.append("ticker", ingTicker.trim().toUpperCase());
      f.append("doc_type", "annual_report");
      f.append("collection", ingCollection.trim());
      if (ingYear.trim()) f.append("year", ingYear.trim());
      f.append("file", ingFile);
      const r = await fetch(`${API_BASE}/agents/ingest`, { method: "POST", body: f });
      const data = (await r.json()) as IngestionResponse | { detail?: unknown };
      if (!r.ok) throw new Error("detail" in data ? JSON.stringify(data.detail) : `Request failed (${r.status})`);
      setIngRes(data as IngestionResponse);
      await refreshDocs(ingCollection.trim() || "fundamental_docs");
    } catch (err) { setError(err instanceof Error ? err.message : "Ingestion failed."); }
    finally { setLoading(false); }
  }

  async function onDeleteDoc(doc: IngestedDoc) {
    setLoading(true); setError(null);
    try {
      await req("/documents/ingested", "DELETE", { collection: doc.collection_name, source_path: doc.source_path });
      await refreshDocs(doc.collection_name);
    } catch (err) { setError(err instanceof Error ? err.message : "Delete failed."); }
    finally { setLoading(false); }
  }

  async function onWeb(e: FormEvent) {
    e.preventDefault(); setLoading(true); setError(null); setWebRes(null);
    setWebEvents([]);
    try {
      if (webMode === "direct") {
        const data = await req<{ result: unknown }>("/agents/web-search", "POST", { messages: [webQuery.trim()] });
        setWebRes(data.result);
      } else {
        await runNdjsonStream("/agents/web-search/stream", { messages: [webQuery.trim()] }, (evt) => {
          setWebEvents((prev) => [...prev, evt]);
          if (evt.event === "final") {
            const result = (evt.data as { result?: unknown } | null)?.result;
            if (result !== undefined) setWebRes(result);
          }
        });
      }
    }
    catch (err) { setError(err instanceof Error ? err.message : "Web failed."); }
    finally { setLoading(false); }
  }
  async function onTech(e: FormEvent) {
    e.preventDefault(); setLoading(true); setError(null); setTechRes(null);
    setTechEvents([]);
    try {
      const payload = { symbol: sym.trim().toUpperCase(), period: period.trim(), interval: interval.trim() };
      if (techMode === "direct") {
        setTechRes(await req<TechnicalResponse>("/agents/technical", "POST", payload));
      } else {
        await runNdjsonStream("/agents/technical/stream", payload, (evt) => {
          setTechEvents((prev) => [...prev, evt]);
          if (evt.event === "final") {
            const result = (evt.data as { result?: TechnicalResponse } | null)?.result;
            if (result) setTechRes(result);
          }
        });
      }
    }
    catch (err) { setError(err instanceof Error ? err.message : "Technical failed."); }
    finally { setLoading(false); }
  }
  async function onFund(e: FormEvent) {
    e.preventDefault(); setLoading(true); setError(null); setFundRes(null);
    setFundEvents([]);
    try {
      const payload = {
        company: fundCompany.trim().toUpperCase(),
        mode: fundMode,
        question: fundMode === "qa" || fundMode === "auto" ? fundQ.trim() || null : null,
        collection: "fundamental_docs",
        top_k: fundTopK,
      };
      if (fundModeType === "direct") {
        setFundRes(await req<FundamentalResponse>("/agents/fundamental", "POST", payload));
      } else {
        await runNdjsonStream("/agents/fundamental/stream", payload, (evt) => {
          setFundEvents((prev) => [...prev, evt]);
          if (evt.event === "final") {
            const result = (evt.data as { result?: FundamentalResponse } | null)?.result;
            if (result) setFundRes(result);
          }
        });
      }
    } catch (err) { setError(err instanceof Error ? err.message : "Fundamental failed."); }
    finally { setLoading(false); }
  }
  async function onSup(e: FormEvent) {
    e.preventDefault(); setLoading(true); setError(null); setSupRes(null);
    setSupEvents([]);
    try {
      const payload = {
        symbol: supSym.trim().toUpperCase(),
        company: supCompany.trim().toUpperCase(),
        fundamental_question: supFQ.trim() || null,
        news_query: supNQ.trim() || null,
        technical_period: "3mo",
        technical_interval: "1d",
        collection: "fundamental_docs",
        top_k: 8,
      };
      if (supMode === "direct") {
        setSupRes(await req<SupervisorResponse>("/agents/supervisor", "POST", payload));
      } else {
        await runNdjsonStream("/agents/supervisor/stream", payload, (evt) => {
          setSupEvents((prev) => [...prev, evt]);
          if (evt.event === "final") {
            const result = (evt.data as { result?: SupervisorResponse } | null)?.result;
            if (result) setSupRes(result);
          }
        });
      }
    } catch (err) { setError(err instanceof Error ? err.message : "Supervisor failed."); }
    finally { setLoading(false); }
  }

  return (
    <main className="app">
      <div className="shell">
        <aside className="sidebar">
          <h1 className="brand">Market Analyst</h1>
          <p className="subtitle">All agents + vector document controls</p>
          <div className="tabs">{tabs.map((t) => <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>)}</div>
          <div className="row" style={{ marginTop: 14 }}>
            <label>Ingested Documents</label>
            <select value={selectedDocId} onChange={(e) => setSelectedDocId(e.target.value)}>
              <option value="">Select document</option>
              {docs.map((d) => <option key={keyOf(d)} value={keyOf(d)}>[{d.ticker || d.company || "N/A"}] {d.year || "NA"} - {d.source_path}</option>)}
            </select>
          </div>
          <button type="button" onClick={applyDoc} disabled={!selectedDocId || loading}>Use Selected Doc In Agent Forms</button>
          <p className="muted" style={{ marginTop: 18 }}>API base: {API_BASE}</p>
        </aside>

        <section className="content">
          {tab === "ingest" && (
            <div className="card">
              <h2>Upload PDF to Vector DB</h2>
              <form onSubmit={onIngest}>
                <div className="grid">
                  <div className="row"><label>Company</label><input value={ingCompany} onChange={(e) => setIngCompany(e.target.value)} /></div>
                  <div className="row"><label>Ticker</label><input value={ingTicker} onChange={(e) => setIngTicker(e.target.value)} /></div>
                </div>
                <div className="grid">
                  <div className="row"><label>Year</label><input value={ingYear} onChange={(e) => setIngYear(e.target.value)} placeholder="2025" /></div>
                  <div className="row"><label>Collection</label><input value={ingCollection} onChange={(e) => setIngCollection(e.target.value)} /></div>
                </div>
                <div className="row"><label>PDF File</label><input type="file" accept="application/pdf" onChange={(e) => setIngFile(e.target.files?.[0] ?? null)} /></div>
                <button type="submit" disabled={loading}>{loading ? "Uploading..." : "Upload + Ingest"}</button>
              </form>
              {ingRes && <div className="result"><span className="pill">company: {ingRes.company}</span><span className="pill">ticker: {ingRes.ticker || "-"}</span><span className="pill">chunks: {ingRes.chunks_stored}</span></div>}
              <div className="result">
                <h3>All Ingested Documents</h3>
                <table className="table"><thead><tr><th>Collection</th><th>Company</th><th>Ticker</th><th>Year</th><th>Type</th><th>Chunks</th><th>Source Path</th><th>Action</th></tr></thead>
                  <tbody>{docs.map((d) => <tr key={keyOf(d)}><td>{d.collection_name}</td><td>{d.company || "-"}</td><td>{d.ticker || "-"}</td><td>{d.year || "-"}</td><td>{d.doc_type || "-"}</td><td>{d.chunks_stored}</td><td>{d.source_path}</td><td><button type="button" style={{ background: "#a4412e" }} onClick={() => onDeleteDoc(d)} disabled={loading}>Delete</button></td></tr>)}</tbody>
                </table>
              </div>
            </div>
          )}

          {tab === "web" && (
            <div className="card">
              <h2>Web Search Agent</h2>
              <form onSubmit={onWeb}>
                <div className="row"><label>Execution Mode</label><select value={webMode} onChange={(e) => setWebMode(e.target.value as InvokeMode)}><option value="direct">Direct Invoke</option><option value="stream">Streaming Invoke</option></select></div>
                <div className="row"><label>Query</label><textarea value={webQuery} onChange={(e) => setWebQuery(e.target.value)} /></div>
                <button type="submit" disabled={loading}>{loading ? "Running..." : "Run Web Search Agent"}</button>
              </form>
              {webMode === "stream" && webEvents.length > 0 && <div className="result"><h3>Live Stream Events</h3><pre className="raw">{JSON.stringify(webEvents, null, 2)}</pre></div>}
              {webRes !== null && <div className="result"><h3>Answer</h3><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{webMd || "No answer extracted; inspect raw payload."}</ReactMarkdown></div><pre className="raw">{JSON.stringify(webRes, null, 2)}</pre></div>}
            </div>
          )}

          {tab === "technical" && (
            <div className="card">
              <h2>Technical Agent</h2>
              <form onSubmit={onTech}>
                <div className="row"><label>Execution Mode</label><select value={techMode} onChange={(e) => setTechMode(e.target.value as InvokeMode)}><option value="direct">Direct Invoke</option><option value="stream">Streaming Invoke</option></select></div>
                <div className="grid"><div className="row"><label>Symbol</label><input value={sym} onChange={(e) => setSym(e.target.value)} /></div><div className="row"><label>Period</label><input value={period} onChange={(e) => setPeriod(e.target.value)} /></div></div>
                <div className="row"><label>Interval</label><input value={interval} onChange={(e) => setInterval(e.target.value)} /></div>
                <button type="submit" disabled={loading}>{loading ? "Running..." : "Run Technical Agent"}</button>
              </form>
              {techMode === "stream" && techEvents.length > 0 && <div className="result"><h3>Live Stream Events</h3><pre className="raw">{JSON.stringify(techEvents, null, 2)}</pre></div>}
              {techRes && <div className="result"><span className="pill">symbol: {techRes.symbol}</span><div className="two-col"><div><h3>Summary</h3><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{techRes.summary}</ReactMarkdown></div></div><div><h3>Indicators</h3><table className="table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{Object.entries(techRes.latest_values).map(([k, v]) => <tr key={k}><td>{k}</td><td>{String(v)}</td></tr>)}</tbody></table></div></div>{chartUrl(techRes.image_path) && <div style={{ marginTop: 12 }}><img src={chartUrl(techRes.image_path) as string} alt="Technical chart" style={{ maxWidth: "100%", border: "1px solid var(--line)", borderRadius: 10 }} /></div>}</div>}
            </div>
          )}

          {tab === "fundamental" && (
            <div className="card">
              <h2>Fundamental Agent</h2>
              <form onSubmit={onFund}>
                <div className="row"><label>Execution Mode</label><select value={fundModeType} onChange={(e) => setFundModeType(e.target.value as InvokeMode)}><option value="direct">Direct Invoke</option><option value="stream">Streaming Invoke</option></select></div>
                <div className="grid"><div className="row"><label>Company / Ticker</label>{fundamentalDocOptions.length > 0 ? <select value={fundCompany} onChange={(e) => setFundCompany(e.target.value)}>{fundamentalDocOptions.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}</select> : <input value={fundCompany} onChange={(e) => setFundCompany(e.target.value)} />}</div><div className="row"><label>Mode</label><select value={fundMode} onChange={(e) => setFundMode(e.target.value as "auto" | "general" | "qa")}><option value="auto">auto</option><option value="general">general</option><option value="qa">qa</option></select></div></div>
                <div className="row"><label>Question</label><textarea value={fundQ} onChange={(e) => setFundQ(e.target.value)} /></div>
                <div className="row"><label>Top K</label><input type="number" min={1} max={25} value={fundTopK} onChange={(e) => setFundTopK(Number(e.target.value))} /></div>
                <button type="submit" disabled={loading}>{loading ? "Running..." : "Run Fundamental Agent"}</button>
              </form>
              {fundModeType === "stream" && fundEvents.length > 0 && <div className="result"><h3>Live Stream Events</h3><pre className="raw">{JSON.stringify(fundEvents, null, 2)}</pre></div>}
              {fundRes && <div className="result"><span className="pill">mode: {fundRes.mode}</span><span className="pill">company: {fundRes.company}</span><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{fundRes.answer}</ReactMarkdown></div><table className="table"><thead><tr><th>Company</th><th>Ticker</th><th>Year</th><th>Doc</th><th>Chunk</th><th>Source Path</th></tr></thead><tbody>{fundRes.sources.map((s, i) => <tr key={`${s.source_path}-${s.chunk_index}-${i}`}><td>{s.company || "-"}</td><td>{s.ticker || "-"}</td><td>{s.year || "-"}</td><td>{s.doc_type || "-"}</td><td>{s.chunk_index || "-"}</td><td>{s.source_path || "-"}</td></tr>)}</tbody></table></div>}
            </div>
          )}

          {tab === "supervisor" && (
            <div className="card">
              <h2>Supervisor Agent</h2>
              <form onSubmit={onSup}>
                <div className="row"><label>Execution Mode</label><select value={supMode} onChange={(e) => setSupMode(e.target.value as InvokeMode)}><option value="direct">Direct Invoke</option><option value="stream">Streaming Invoke</option></select></div>
                <div className="grid"><div className="row"><label>Symbol</label><input value={supSym} onChange={(e) => setSupSym(e.target.value)} /></div><div className="row"><label>Company / Ticker</label>{supervisorCompanyOptions.length > 0 ? <select value={supCompany} onChange={(e) => setSupCompany(e.target.value)}>{supervisorCompanyOptions.map((v) => <option key={v} value={v}>{v}</option>)}</select> : <input value={supCompany} onChange={(e) => setSupCompany(e.target.value)} />}</div></div>
                <div className="row"><label>Fundamental Focus</label><textarea value={supFQ} onChange={(e) => setSupFQ(e.target.value)} /></div>
                <div className="row"><label>News Focus</label><textarea value={supNQ} onChange={(e) => setSupNQ(e.target.value)} /></div>
                <button type="submit" disabled={loading}>{loading ? "Running..." : "Run Supervisor"}</button>
              </form>
              {supMode === "stream" && supEvents.length > 0 && <div className="result"><h3>Live Stream Events</h3><pre className="raw">{JSON.stringify(supEvents, null, 2)}</pre></div>}
              {supRes && <div className="result"><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{supMd}</ReactMarkdown></div><table className="table"><thead><tr><th>Company</th><th>Ticker</th><th>Year</th><th>Doc</th><th>Chunk</th><th>Source Path</th></tr></thead><tbody>{supRes.fundamental.sources.map((s, i) => <tr key={`${s.source_path}-${s.chunk_index}-${i}`}><td>{s.company || "-"}</td><td>{s.ticker || "-"}</td><td>{s.year || "-"}</td><td>{s.doc_type || "-"}</td><td>{s.chunk_index || "-"}</td><td>{s.source_path || "-"}</td></tr>)}</tbody></table></div>}
            </div>
          )}

          {error && <p className="error">Error: {error}</p>}
        </section>
      </div>
    </main>
  );
}
