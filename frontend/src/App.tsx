import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight, ArrowUp, BookOpen, Check, ChevronDown, FileText, History,
  Library, Menu, PanelRightClose, PanelRightOpen, Plus, Search, Settings2,
  ShieldCheck, Upload, X,
} from "lucide-react";
import type { AnswerJob, AnswerResult, Citation, Paper, QuestionRecord, SearchResult } from "./api/contracts";
import { ApiFailure, createPendingStore, createRequestId, runRequest } from "./api/client";
import type { PendingRequest } from "./api/client";
import { useBackend } from "./api/useBackend";
import AnswerView from "./components/AnswerView";
import PdfReader from "./components/PdfReader";
import { answerDemo, DATASET_ID, exampleQueries, papers as demoPapers, searchDemoPapers } from "./mocks/demo";
import { appendHistory, clearHistory, loadHistory, saveHistory } from "./storage/history";

type View = "workspace" | "library" | "import";
type Mode = "answer" | "search";
const EMPTY_PAPERS: Paper[] = [];

export default function App() {
  const demoMode = new URLSearchParams(window.location.search).get("demo") === "1";
  const backend = useBackend(demoMode);
  const papers = demoMode ? demoPapers : backend.data?.papers ?? EMPTY_PAPERS;
  const datasetId = demoMode ? DATASET_ID : backend.data?.health.dataset_id ?? "";
  const sessionId = backend.data?.health.session_id ?? "";
  const historyContext = useMemo(() => ({ datasetId, papers, source: demoMode ? "demo" as const : "api" as const, sessionId }),
    [datasetId, papers, demoMode, sessionId]);
  const contextKey = `${datasetId}:${sessionId}`;
  const [historyReadyKey, setHistoryReadyKey] = useState(demoMode ? contextKey : "");
  const [view, setView] = useState<View>("workspace");
  const [mode, setMode] = useState<Mode>("answer");
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<string[]>([]);
  const [submittedScope, setSubmittedScope] = useState<string[]>([]);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [history, setHistory] = useState(() => demoMode ? loadHistory() : []);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [searchResults, setSearchResults] = useState<Paper[] | null>(null);
  const [error, setError] = useState("");
  const [storageWarning, setStorageWarning] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [activePaper, setActivePaper] = useState<Paper | null>(null);
  const [page, setPage] = useState(1);
  const [readerOpen, setReaderOpen] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [waitingForCurrent, setWaitingForCurrent] = useState(false);
  const help = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const latestRequest = useRef(0);
  const pending = useRef<PendingRequest | null>(null);
  const inFlight = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const recoveredContext = useRef("");
  const [jobState, setJobState] = useState<AnswerJob["state"] | "submitting">("submitting");
  const [checkpoint] = useState(() => createPendingStore());
  useEffect(() => {
    if (!datasetId) return;
    setHistory(loadHistory(undefined, historyContext)); setHistoryReadyKey(contextKey);
  }, [historyContext, datasetId, contextKey]);
  useEffect(() => {
    if (demoMode && datasetId && historyReadyKey === contextKey)
      setStorageWarning(!saveHistory(history, undefined, historyContext));
  }, [history, historyContext, historyReadyKey, datasetId, contextKey, demoMode]);
  useEffect(() => {
    if (demoMode || !datasetId) return;
    const refreshHistory = () => setHistory(loadHistory(undefined, historyContext));
    window.addEventListener("storage", refreshHistory);
    return () => window.removeEventListener("storage", refreshHistory);
  }, [demoMode, datasetId, historyContext]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (demoMode || !backend.data || recoveredContext.current === contextKey) return;
    recoveredContext.current = contextKey;
    const intent = checkpoint.read();
    if (!intent) return;
    if (intent.dataset_id !== datasetId || intent.session_id !== sessionId || Date.now() - intent.created_at > 6 * 3600_000 ||
      intent.paper_ids.some(id => !papers.some(p => p.paper_id === id))) {
      if (!checkpoint.clear(intent.key)) setStorageWarning(true);
      setError("之前的任务属于其他资料库、会话或已超过恢复期限，未自动重新提交。"); return;
    }
    pending.current = intent; setMode(intent.kind === "search" ? "search" : "answer");
    setSubmittedQuery(intent.query); setSubmittedScope(intent.paper_ids); setWaitingForCurrent(true);
    void executeLive(intent, ++latestRequest.current);
  }, [backend.data, contextKey, datasetId, sessionId, demoMode, papers, checkpoint]);

  const examples = demoMode ? exampleQueries : papers.slice(0, 3).map(paper => ({
    paper_id: paper.paper_id, label: "理解论文方法", query: `请概括《${paper.title}》的核心方法。`,
  }));
  const filterLibrary = (query: string) => demoMode ? searchDemoPapers(query) : papers.filter(paper =>
    [paper.title, paper.canonical_title, paper.summary, paper.year, paper.venue, ...paper.authors, ...paper.tags].join(" ").toLowerCase().includes(query.trim().toLowerCase()));

  async function executeLive(intent: PendingRequest, request: number) {
    if (inFlight.current) return;
    inFlight.current = true; setBusy(true); setError(""); setJobState("submitting");
    const abort = new AbortController(); controller.current = abort; pending.current = intent;
    if (!checkpoint.write(intent)) setStorageWarning(true);
    try {
      const job = await runRequest(intent, progress => {
        intent.job_id = progress.job_id; setJobState(progress.state);
        if (!checkpoint.write(intent)) setStorageWarning(true);
      }, abort.signal);
      if (job.state !== "completed" || !job.result) {
        pending.current = null; if (!checkpoint.clear(intent.key)) setStorageWarning(true);
        throw new ApiFailure(job.error?.message || "任务未能完成，请重新提问。");
      }
      if (intent.kind === "answer") {
        const answer = job.result as AnswerResult;
        const record: QuestionRecord = { id: job.job_id, query: intent.query, paper_ids: intent.paper_ids,
          result: answer, dataset_id: datasetId, created_at: new Date().toISOString() };
        const saved = appendHistory(record, historyContext);
        if (!saved) setStorageWarning(true);
        setHistory(items => saved ? loadHistory(undefined, historyContext) :
          [record, ...items.filter(item => item.id !== record.id)].slice(0, 20));
        if (request === latestRequest.current) {
          setResult(answer); setActiveId(record.id); setQuestion("");
          if (answer.citations[0]) openCitation(answer.citations[0]);
        }
      } else if (request === latestRequest.current) {
        setSearchResults((job.result as SearchResult).papers); setQuestion("");
      }
      pending.current = null; if (!checkpoint.clear(intent.key)) setStorageWarning(true);
    } catch (reason) {
      if (abort.signal.aborted) return;
      if (reason instanceof ApiFailure && [404, 409, 422, 429].includes(reason.status)) {
        pending.current = null; if (!checkpoint.clear(intent.key)) setStorageWarning(true);
      }
      setError(reason instanceof Error ? reason.message : "请求未能完成。");
    } finally { inFlight.current = false; setBusy(false); setWaitingForCurrent(false); }
  }
  function resumeTask() {
    const intent = pending.current;
    if (!intent || inFlight.current) return;
    changeView("workspace"); setSubmittedQuery(intent.query); setSubmittedScope(intent.paper_ids);
    setWaitingForCurrent(true);
    setResult(null); setSearchResults(null); setMode(intent.kind === "search" ? "search" : "answer");
    resetWorkspacePosition(); void executeLive(intent, ++latestRequest.current);
  }

  function resetWorkspacePosition(focusInput = false) {
    requestAnimationFrame(() => {
      if (focusInput) input.current?.focus({ preventScroll: true });
      for (const selector of [".workspace", ".conversation-scroll"]) {
        const element = document.querySelector(selector);
        if (element) element.scrollTop = 0;
      }
    });
  }
  function changeView(next: View) { setView(next); setMobileNavOpen(false); setScopeOpen(false); }
  function openCitation(citation: Citation, reveal = false) {
    const paper = papers.find(item => item.paper_id === citation.paper_id);
    if (!paper || !paper.files.some(file => file.file_id === citation.file_id)) {
      setError("这条引用的原文文件没有找到。"); return;
    }
    setActiveCitation(citation); setActivePaper(paper); setPage(citation.page_number); setReaderOpen(true);
    if (reveal && window.matchMedia("(max-width: 650px)").matches)
      requestAnimationFrame(() => document.querySelector(".reader-panel")?.scrollIntoView({ block: "start", behavior: "smooth" }));
  }
  function newQuestion() {
    latestRequest.current += 1; if (demoMode) setBusy(false); changeView("workspace"); setQuestion("");
    setWaitingForCurrent(false);
    setScope([]); setSubmittedScope([]); setMode("answer");
    setResult(null); setSubmittedQuery(""); setSearchResults(null); setError(""); setActiveId(null);
    setActivePaper(null); setActiveCitation(null); setPage(1);
    resetWorkspacePosition(true);
  }
  function readPaper(paper: Paper) {
    // Opening a source starts a new workspace intent; an older submission must not take it over.
    newQuestion(); setActivePaper(paper); setReaderOpen(true);
    if (window.matchMedia("(max-width: 650px)").matches)
      requestAnimationFrame(() => document.querySelector(".reader-panel")?.scrollIntoView({ block: "start", behavior: "smooth" }));
  }
  function askPaper(paper: Paper) {
    newQuestion(); setScope([paper.paper_id]);
    setQuestion(demoMode ? exampleQueries.find(item => item.paper_id === paper.paper_id)?.query || "" : `请概括《${paper.title}》的核心方法。`);
  }
  function showRecord(record: QuestionRecord) {
    latestRequest.current += 1; if (demoMode) setBusy(false); changeView("workspace"); setMode("answer");
    setWaitingForCurrent(false);
    setQuestion(""); setScope(record.paper_ids); setSubmittedScope(record.paper_ids); setSubmittedQuery(record.query);
    setResult(record.result); setActiveId(record.id); setError(""); setSearchResults(null);
    setActivePaper(null); setActiveCitation(null); setPage(1);
    resetWorkspacePosition();
  }
  async function submit(override?: string) {
    const query = (override ?? question).trim();
    if (!query || busy) return;
    if (!demoMode && (!backend.data || inFlight.current)) return;
    if (!demoMode && pending.current) { setError("还有任务未确认结果，请先恢复任务，避免重复调用。"); return; }
    if (query.length > 2000) { setError("问题请控制在 2000 字以内。"); return; }
    let key: string;
    try { key = createRequestId(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "无法创建请求编号，请重试。"); return; }
    const request = ++latestRequest.current;
    setBusy(true); setWaitingForCurrent(true); setError(""); setScopeOpen(false); setSubmittedQuery(query); setSubmittedScope([...scope]);
    setResult(null); setSearchResults(null); setActiveId(null); setActiveCitation(null); setActivePaper(null);
    resetWorkspacePosition();
    if (!demoMode) {
      const intent: PendingRequest = { key, job_id: null, kind: mode === "answer" ? "answer" : "search",
        query, paper_ids: [...scope], dataset_id: datasetId, session_id: sessionId, created_at: Date.now() };
      await executeLive(intent, request); return;
    }
    // UI loading only: no network call, RAG task or provider verification occurs.
    await new Promise(resolve => setTimeout(resolve, 150));
    if (request !== latestRequest.current) return;
    try {
      if (mode === "search") {
        setSearchResults(searchDemoPapers(query).filter(p => !scope.length || scope.includes(p.paper_id)));
      } else {
        const answer = answerDemo({ query, paper_ids: scope });
        setResult(answer);
        const record: QuestionRecord = {
          id: key, query, paper_ids: [...scope], result: answer,
          created_at: new Date().toISOString(), dataset_id: DATASET_ID,
        };
        setHistory(items => [record, ...items].slice(0, 20)); setActiveId(record.id);
        if (answer.citations[0]) openCitation(answer.citations[0]);
      }
      setQuestion("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "界面示例无法加载。"); }
    finally { if (request === latestRequest.current) { setBusy(false); setWaitingForCurrent(false); } }
  }
  const labelForScope = (ids: string[]) => ids.length === 0 ? (demoMode ? "全部示例论文" : "全部论文") : ids.length === 1
    ? papers.find(p => p.paper_id === ids[0])?.title.split("：")[0] || "指定论文" : `已选 ${ids.length} 篇论文`;
  const scopeLabel = labelForScope(scope);

  return <div className={`app-shell ${mobileNavOpen ? "nav-open" : ""}`}>
    {mobileNavOpen && <button className="nav-scrim" aria-label="关闭导航" onClick={() => setMobileNavOpen(false)} />}
    <aside className="sidebar">
      <a className="wordmark" href="#" onClick={event => { event.preventDefault(); newQuestion(); }}>
        <span className="brand-tile"><BookOpen size={23} strokeWidth={1.8} /></span>
        <span><strong>ScholarRAG</strong><small>论文知识工作台</small></span>
      </a>
      <button className="new-question" onClick={newQuestion}><Plus size={17} />新建问题<span>＋</span></button>
      <nav aria-label="主导航">
        <button className={view === "workspace" ? "active" : ""} onClick={() => changeView("workspace")}><BookOpen size={18} />问答工作台</button>
        <button className={view === "library" ? "active" : ""} onClick={() => changeView("library")}><Library size={18} />论文库<span className="nav-count">{papers.length}</span></button>
        <button className={view === "import" ? "active" : ""} onClick={() => changeView("import")}><Upload size={18} />论文入库</button>
      </nav>
      <div className="history-heading"><span>最近的问题</span><History size={13} /></div>
      <div className="history-list">
        {history.length ? history.map(record => <button key={record.id}
          className={activeId === record.id ? "selected" : ""} onClick={() => showRecord(record)}
          title={record.query}><span className="history-dot" /><span>{record.query}</span></button>) :
          <p className="history-empty">你的问题记录会保存在<br />当前浏览器中。</p>}
      </div>
      {history.length > 0 && <button className="clear-history" onClick={() => {
        if (!clearHistory(historyContext)) setStorageWarning(true);
        setHistory([]); setActiveId(null);
      }}>{demoMode ? "清空示例记录" : "清空本地记录"}</button>}
      <div className="sidebar-bottom"><div className="dataset-mark"><span className="dataset-dot" /><div><strong>{demoMode ? "公开示例资料库" : backend.data?.health.label || "正在连接资料库"}</strong><small>{demoMode ? "3 份原创合成文档" : `${papers.length} 篇论文 · 只读检索`}</small></div></div>
        <button className="help-button" onClick={() => help.current?.showModal()}><Settings2 size={16} />使用说明<ArrowRight size={14} /></button>
      </div>
    </aside>

    <div className="main-shell">
      <header className="topbar">
        <div><button className="icon-button mobile-menu" aria-label="打开导航" onClick={() => setMobileNavOpen(true)}><Menu size={19} /></button>
          <span className="topbar-product">论文助手</span><span className="breadcrumb-separator">/</span>
          <strong>{view === "workspace" ? "问答工作台" : view === "library" ? "论文库" : "论文入库"}</strong>
        </div>
        <div><span className="demo-status"><span />{demoMode ? "界面演示" : backend.data ? "真实 RAG" : "连接服务"}</span>
          {view === "workspace" && <button className="icon-button" aria-label={readerOpen ? "收起原文阅读区" : "展开原文阅读区"} onClick={() => setReaderOpen(value => !value)}>
            {readerOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}</button>}
        </div>
      </header>
      <div className="demo-banner"><span>{demoMode ? "DEMO" : "RAG"}</span><p>{demoMode ?
        "当前是前端原型：答案为预设示例，未调用模型；右侧 PDF 为真实合成文件。" :
        backend.data ? `已连接${backend.data.health.label}。检索与回答使用真实服务，核验按服务端配置执行。` : "正在连接后端并读取论文库，请稍候…"}</p></div>
      {!demoMode && backend.error && <div className="connection-error" role="alert">{backend.error}<button onClick={backend.retry}>重新连接</button></div>}
      {error && !submittedQuery && <div className="connection-error" role="alert">{error}</div>}
      {!demoMode && pending.current && !busy && <div className="task-recovery" role="status">有一个任务尚未确认结果。<button onClick={resumeTask}>恢复任务</button></div>}
      {storageWarning && <p className="storage-warning" role="status">浏览器无法保存记录，关闭页面后记录或任务恢复信息可能丢失。</p>}

      {view === "workspace" ? <div className={`workspace ${readerOpen ? "" : "reader-collapsed"}`}>
        <main className="conversation">
          <div className="conversation-scroll">
            {!submittedQuery ? <div className="welcome">
              <span className="welcome-symbol"><BookOpen size={27} strokeWidth={1.5} /></span>
              <p className="eyebrow">YOUR PAPERS, WITH EVIDENCE</p>
              <h1>从一个问题，<br />找到论文里的依据。</h1>
              <p className="welcome-description">找回模糊记忆中的论文，理解关键方法，<br />让每条结论都能回到原文核对。</p>
              <div className="example-grid">{examples.map(example => <button key={example.paper_id}
                onClick={() => { setMode("answer"); if (!demoMode) setScope([example.paper_id]); setQuestion(example.query); input.current?.focus(); }}>
                <span>{example.label}<ArrowRight size={14} /></span><strong>{example.query}</strong>
              </button>)}</div>
              {demoMode && <button className="refusal-example" onClick={() => { setMode("answer"); setScope([]); setQuestion("DemoMesh的学习率是多少？"); input.current?.focus(); }}>
                <ShieldCheck size={14} />也可以试试资料未记载的问题
              </button>}
            </div> : <div className="question-result">
              <div className="question-heading"><span>你的问题</span><span className="neutral-badge">{labelForScope(submittedScope)}</span></div>
              <h1>{submittedQuery}</h1>
              {busy && waitingForCurrent && <p className="loading-line" role="status"><span className="loading-dot" />{demoMode ? "正在载入界面示例…" :
                jobState === "queued" ? "任务已接纳，正在排队等待…" : jobState === "running" ?
                  pending.current?.kind === "search" ? "正在检索并判断论文相关性…" : "正在检索、生成并核验答案…" : "正在提交任务…"}</p>}
              {error && <div className="error-card" role="alert">{error}</div>}
              {result && <AnswerView result={result} onCitation={citation => openCitation(citation, true)} />}
              {searchResults && <div className="search-results">
                <div className="sources-heading">{demoMode ? "示例匹配" : "候选论文"} <span>{searchResults.length}</span></div>
                <p className="small-muted">{demoMode ? "仅匹配预设关键词，尚未接入真实语义检索。" : "来自论文级检索与相关性判断，可打开原文或基于单篇继续提问。"}</p>
                {searchResults.length === 0 && <div className="error-card">{demoMode ? "当前没有匹配的界面示例。可以试试“扩散”或“隐私”。" : "没有论文通过当前相关性判断，请补充方法、任务或数据集线索。"}</div>}
                {searchResults.map(paper => <div className="search-paper" key={paper.paper_id}><h3>{paper.title}</h3><p>{paper.summary}</p>
                  <div className="paper-actions"><button onClick={() => readPaper(paper)}>阅读原文<ArrowRight size={13} /></button><button onClick={() => askPaper(paper)}>基于这篇提问</button></div>
                </div>)}
              </div>}
            </div>}
          </div>
          <div className="composer-area">
            <div className="composer">
              <div className="composer-options"><div className="mode-switch">
                <button className={mode === "answer" ? "selected" : ""} onClick={() => setMode("answer")}><BookOpen size={14} />问知识</button>
                <button className={mode === "search" ? "selected" : ""} onClick={() => setMode("search")}><Search size={14} />找论文</button>
              </div>
              <div className="scope-selector">
                <button className="scope-button" aria-expanded={scopeOpen} onClick={() => setScopeOpen(value => !value)}><FileText size={13} />{scopeLabel}<ChevronDown size={12} /></button>
                {scopeOpen && <div className="scope-popover"><div>选择提问范围</div>
                  <button onClick={() => setScope([])}><span>{scope.length === 0 && <Check size={14} />}</span>{demoMode ? "全部示例论文" : "全部论文"}</button>
                  {papers.map(paper => <button key={paper.paper_id} disabled={!scope.includes(paper.paper_id) && scope.length >= 3}
                    onClick={() => setScope(items => items.includes(paper.paper_id) ? items.filter(id => id !== paper.paper_id) : [...items, paper.paper_id])}>
                    <span>{scope.includes(paper.paper_id) && <Check size={14} />}</span>{paper.title.split("：")[0]}
                  </button>)}
                  <small>{demoMode ? "选择后仅使用这些资料的界面示例。" : "最多限定三篇论文；问答只检索选中文献。"}</small>
                </div>}
              </div></div>
              <label className="sr-only" htmlFor="question">输入你的论文问题</label>
              <textarea ref={input} id="question" value={question} maxLength={2000}
                placeholder={mode === "answer" ? "输入一个论文问题，或选一篇论文开始…" : "描述你记得的论文方法或关键词…"}
                onChange={event => setQuestion(event.target.value)}
                onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); } }} />
              <div className="composer-bottom"><span>Enter 发送 · Shift + Enter 换行</span>
                <button className="send-button" aria-label={mode === "answer" ? "提交问题" : demoMode ? "查找示例论文" : "查找论文"} disabled={!question.trim() || busy || (!demoMode && !backend.data)} onClick={() => void submit()}><ArrowUp size={18} /></button>
              </div>
            </div>
            <p className="composer-footnote">{demoMode ? "示例资料和答案仅用于交互演示；未调用真实 RAG 服务。" : busy ?
              pending.current?.kind === "search" ? "搜索任务处理中；刷新可以继续查看，切换阅读不会取消服务端执行。" :
              "已有问答任务处理中；切换阅读不会取消服务端执行，完成结果会保存在问题记录中。" : "回答依据检索证据生成，请点击引用核对原文；问题之间独立处理。"}</p>
          </div>
        </main>
        {readerOpen && <PdfReader paper={activePaper} citation={activeCitation} pageNumber={page} onPageChange={setPage} demoMode={demoMode} />}
      </div> : view === "library" ? <main className="library-page">
        <div className="page-heading"><div><p className="eyebrow">PAPER LIBRARY</p><h1>你的论文，井然有序。</h1><p>阅读原文，或选中一篇资料开始提问。</p></div><span className="collection-count">{papers.length}<small>{demoMode ? "份合成示例" : "篇论文"}</small></span></div>
        <label className="library-search"><Search size={17} /><input aria-label="筛选论文" placeholder="搜索标题、方法或关键词" value={libraryQuery} onChange={event => setLibraryQuery(event.target.value)} /><span>{demoMode ? "示例库" : "论文库"}</span></label>
        <div className="library-list">
          {filterLibrary(libraryQuery).slice(0, 50).map((paper, index) => <article className="library-card" key={paper.paper_id}>
            <div className="paper-cover"><FileText size={29} strokeWidth={1.2} /><span>0{index + 1}</span></div>
            <div className="paper-detail"><div className="paper-meta"><span>{demoMode ? "合成资料" : paper.year || "年份未登记"}</span><span>PDF · {paper.files.length} 份文件</span></div>
              <h2>{paper.title}</h2><p className="english-title">{paper.canonical_title}</p>
              <p className="paper-authors">{paper.authors.length ? paper.authors.slice(0, 3).join("、") + (paper.authors.length > 3 ? " 等" : "") : "作者未登记"}{paper.venue && ` · ${paper.venue}`}</p>
              <p className="paper-summary">{paper.summary}</p>
              <div className="paper-tags">{paper.tags.map(tag => <span key={tag}>{tag}</span>)}</div>
            </div><div className="library-actions"><button className="primary-button" onClick={() => askPaper(paper)}>基于这篇提问<ArrowRight size={14} /></button>
              <button onClick={() => readPaper(paper)}><BookOpen size={14} />阅读原文</button></div>
          </article>)}
          {!filterLibrary(libraryQuery).length && <div className="empty-library"><Search size={26} /><h3>没有匹配的资料</h3><p>请调整标题、方法或关键词。</p></div>}
          {filterLibrary(libraryQuery).length > 50 && <p className="small-muted">当前显示前 50 篇，请使用筛选缩小范围。</p>}
        </div>
        <p className="library-note"><ShieldCheck size={14} />{demoMode ? "这三份文档为原创合成示例，不代表真实研究结果或正式论文库。" : "资料库由服务端统一提供。当前页面只读，不会触发论文入库或向量重建。"}</p>
      </main> : <main className="import-page">
        <div className="page-heading"><div><p className="eyebrow">GROW YOUR LIBRARY</p><h1>新增论文，从确认来源开始。</h1><p>这一页展示入库流程，实际上传将在后端接入后开放。</p></div></div>
        <div className="upload-placeholder"><span><Upload size={29} strokeWidth={1.5} /></span><h2>上传 PDF，检查信息，再加入论文库</h2><p>文件检查、信息提取与索引状态会在这里统一展示。</p><button disabled className="primary-button">上传功能尚未开放</button></div>
        <div className="import-steps">{[
          ["01", "检查文件", "验证 PDF 格式与重复内容"],
          ["02", "确认论文档案", "检查标题、作者和方法线索"],
          ["03", "构建检索索引", "完成两级索引后才显示可检索"],
        ].map(([number, title, description]) => <div key={number}><span>{number}</span><h3>{title}</h3><p>{description}</p></div>)}</div>
        <div className="import-boundary"><ShieldCheck size={19} /><p>上传文件、登记档案和索引完成是不同步骤。正式接入后，每一步都会有独立状态与失败提示。</p></div>
      </main>}
    </div>
    <dialog ref={help} className="help-dialog"><div><span className="brand-tile"><BookOpen size={21} /></span><button className="icon-button" aria-label="关闭使用说明" onClick={() => help.current?.close()}><X size={18} /></button></div>
      <h2>{demoMode ? "界面演示说明" : "提问、回答与原文核对"}</h2><p>{demoMode ? "当前使用预设示例，没有调用外部模型或 Claim 评审。" :
        "当前页面连接真实论文 RAG 服务，复用两级检索、生成与 Claim 核验。问题独立处理，选择论文可以限定证据范围。"}</p>
      <p>点击引用可定位到 PDF 文件页并查看证据片段；不提供段落精确高亮。问题记录保存在当前浏览器，最多保留 20 条。</p>
      <p>{demoMode ? "通过 ?demo=1 明确进入演示模式。" : "任务使用有界队列。刷新会恢复尚未结束的任务；服务重启会提示中断，不自动重新调用模型。"}上传入口目前不可用。</p>
      <button className="primary-button" onClick={() => help.current?.close()}>知道了<Check size={15} /></button>
    </dialog>
  </div>;
}
