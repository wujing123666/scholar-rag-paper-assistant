import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Download, FileText, Minus, Plus } from "lucide-react";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { PDFDocumentLoadingTask, PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import type { Citation, Paper } from "../api/contracts";

export default function PdfReader({ paper, citation, pageNumber, onPageChange, demoMode = true }: {
  paper: Paper | null;
  citation: Citation | null;
  pageNumber: number;
  onPageChange: (page: number) => void;
  demoMode?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [width, setWidth] = useState(420);
  const [zoom, setZoom] = useState(1);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const file = citation ? paper?.files.find(item => item.file_id === citation.file_id) : paper?.files[0];

  useEffect(() => {
    if (!host.current) return;
    const observer = new ResizeObserver(entries => setWidth(entries[0].contentRect.width));
    observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    setDocument(null); setError(""); setZoom(1);
    if (!file) { setLoading(false); if (paper) setError("引用对应的 PDF 文件不存在。"); return; }
    let disposed = false;
    let task: PDFDocumentLoadingTask | undefined;
    setLoading(true);
    // Load the PDF engine only when a source is opened, not on the welcome page.
    import("pdfjs-dist").then(pdfjs => {
      if (disposed) return;
      pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
      task = pdfjs.getDocument({ url: file.url, useSystemFonts: true });
      return task.promise;
    }).then(doc => {
      if (disposed || !doc) return;
      setDocument(doc); setLoading(false);
    }).catch((reason: unknown) => {
      if (!disposed) { setError(reason instanceof Error ? reason.message : "无法读取这个 PDF"); setLoading(false); }
    });
    return () => {
      disposed = true;
      void task?.destroy().catch(() => {});
    };
  }, [file?.url]);
  useEffect(() => {
    if (!document || !canvas.current) return;
    setError("");
    let disposed = false;
    let renderTask: RenderTask | undefined;
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > document.numPages) {
      setError("引用页码超出 PDF 范围，请下载原文核对。"); return;
    }
    const number = pageNumber;
    document.getPage(number).then(page => {
      if (disposed || !canvas.current) return;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const base = page.getViewport({ scale: 1 });
      const fit = Math.max(200, width - 36) / base.width;
      const viewport = page.getViewport({ scale: fit * zoom * pixelRatio });
      const element = canvas.current;
      element.width = Math.ceil(viewport.width);
      element.height = Math.ceil(viewport.height);
      element.style.width = `${viewport.width / pixelRatio}px`;
      element.style.height = `${viewport.height / pixelRatio}px`;
      const context = element.getContext("2d");
      if (!context) throw new Error("浏览器无法创建 PDF 画布");
      renderTask = page.render({ canvas: element, canvasContext: context, viewport });
      return renderTask.promise;
    }).catch((reason: unknown) => {
      if (!disposed && !(reason instanceof Error && reason.name === "RenderingCancelledException"))
        setError("当前页面无法显示，请下载原文查看。");
    });
    return () => { disposed = true; renderTask?.cancel(); };
  }, [document, pageNumber, width, zoom]);

  return <aside className="reader-panel" aria-label="原文阅读区">
    <div className="reader-header"><div><span className="eyebrow">SOURCE READER</span><h2>原文核对</h2></div>
      <span className="neutral-badge">{demoMode ? "公开示例 PDF" : "论文 PDF"}</span>
    </div>
    {paper && file ? <>
      <div className="reader-file"><span className="file-icon"><FileText size={19} /></span>
        <div><strong>{paper.title}</strong><small>{file.name}</small></div>
        <a className="icon-button" href={file.url} download={file.name} aria-label="下载原文 PDF"><Download size={17} /></a>
      </div>
      <div className="pdf-toolbar">
        <div><button className="icon-button" aria-label="上一页" disabled={!document || pageNumber <= 1} onClick={() => onPageChange(pageNumber - 1)}><ChevronLeft size={16} /></button>
          <span>第 <strong>{pageNumber}</strong> / {document?.numPages ?? "…"} 页</span>
          <button className="icon-button" aria-label="下一页" disabled={!document || pageNumber >= document.numPages} onClick={() => onPageChange(pageNumber + 1)}><ChevronRight size={16} /></button>
        </div>
        <div><button className="icon-button" aria-label="缩小 PDF" disabled={zoom <= 0.8} onClick={() => setZoom(value => Math.max(0.8, value - 0.2))}><Minus size={14} /></button>
          <button className="zoom-label" onClick={() => setZoom(1)} aria-label="恢复适合宽度">{Math.round(zoom * 100)}%</button>
          <button className="icon-button" aria-label="放大 PDF" disabled={zoom >= 1.8} onClick={() => setZoom(value => Math.min(1.8, value + 0.2))}><Plus size={14} /></button>
        </div>
      </div>
    </> : null}
    <div className={`pdf-stage ${paper ? "" : "empty"}`} ref={host}>
      {!paper ? <div className="reader-empty"><span><FileText size={32} strokeWidth={1.3} /></span><h3>让每条回答，都有出处</h3><p>点击答案中的引用，或从论文库打开一篇资料。<br />这里会显示原文和对应页码。</p></div> :
       <>{error ? <div className="reader-error" role="alert"><FileText size={25} /><p>{error}</p>{file && <a href={file.url} download>下载原文</a>}</div> :
         loading && <p className="pdf-loading" role="status">正在读取 PDF…</p>}
         <canvas ref={canvas} className={loading || error ? "hidden" : ""} aria-label={`PDF 第 ${pageNumber} 页`} />
       </>}
    </div>
    {citation ? <section className="evidence-section">
      <div><span className="source-number">{citation.citation_id.replace("C", "")}</span><strong>引用证据</strong><span className="small-muted">PDF 第 {citation.page_number} 页</span></div>
      {citation.section && <p className="evidence-section-title">{citation.section}</p>}
      <p className="evidence-excerpt">{citation.excerpt}</p>
      <small>当前为页级定位，未对 PDF 段落做精确高亮。</small>
    </section> : <div className="reader-hint">PDF 页码按文件顺序计算，与论文印刷页码可能不同。</div>}
  </aside>;
}
