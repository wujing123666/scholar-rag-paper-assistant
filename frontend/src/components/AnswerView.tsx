import { BookOpen, ShieldCheck } from "lucide-react";
import type { AnswerResult, Citation } from "../api/contracts";

export default function AnswerView({ result, onCitation }: {
  result: AnswerResult;
  onCitation: (citation: Citation) => void;
}) {
  const refusalReasons: Record<string, string> = {
    no_candidate_papers: "没有检索到足够相关的候选论文，请补充方法、任务或数据集线索。",
    model_reported_insufficient_evidence: "模型检查了当前证据，但资料没有提供足够信息来可靠回答这个问题。",
    not_enough_retrieved_chunks: "检索到的原文证据不足，请调整问题或限定论文范围。",
    claim_verification_removed_all: "生成的结论未通过证据核验，因此本次不发布回答。",
  };
  if (result.status === "refused") return <div className="refusal-card" role="status">
    <div className="refusal-icon"><ShieldCheck size={21} /></div>
    <h3>这份资料暂时无法支持回答</h3>
    <p>{result.source === "demo" ? result.reason : refusalReasons[result.reason || ""] || "当前原文证据不足，暂时无法可靠回答。"}</p>
    <div className="small-muted">你可以补充论文线索，或查看另一份资料。</div>
  </div>;
  return <article className="answer-view">
    {result.status === "evidence_only" && <div className="error-card">本次未完成语义核验，仅提供原文证据供核对。</div>}
    <div className="answer-label"><span className="brand-tile small"><BookOpen size={15} /></span>ScholarRAG
      <span className="neutral-badge">{result.source === "demo" ? "演示答案" : "证据回答"}</span>
    </div>
    <div className="answer-body">
      {result.claims.map((claim, index) => <p key={index}>{claim.text}{" "}
        {claim.citations.map(id => {
          const citation = result.citations.find(item => item.citation_id === id);
          return citation ? <button className="inline-citation" key={id}
            aria-label={`查看引用 ${id}，第${citation.page_number}页`}
            onClick={() => onCitation(citation)}>{id.replace("C", "")}</button> : null;
        })}
      </p>)}
    </div>
    <div className="sources-heading">引用来源 <span>{result.citations.length}</span></div>
    {result.citations.map(citation => <button className="source-card" key={citation.citation_id}
      onClick={() => onCitation(citation)}>
      <span className="source-number">{citation.citation_id.replace("C", "")}</span>
      <span><strong>{citation.paper_title}</strong><small>PDF 第 {citation.page_number} 页{citation.section && ` · ${citation.section}`}</small></span>
      <BookOpen size={16} />
    </button>)}
    <p className="verification-note"><ShieldCheck size={13} />
      {result.verification.status === "demo" ? "演示标注 · 未调用生成模型或 Claim 评审" :
       result.verification.status === "verified" ? "已完成本次 Claim 核验；仍可对照原文确认" : "本次没有完成语义核验，请核对原文"}
    </p>
  </article>;
}
