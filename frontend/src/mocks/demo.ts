import meshUrl from "../../../examples/paper_demo/pdfs/demo_mesh.pdf?url";
import recruitUrl from "../../../examples/paper_demo/pdfs/demo_recruit.pdf?url";
import privacyUrl from "../../../examples/paper_demo/pdfs/demo_privacy.pdf?url";
import source from "../../../examples/paper_demo/papers.json";
import type { AnswerRequest, AnswerResult, Paper } from "../api/contracts";

export const DATASET_ID = "synthetic-demo-v1";
const urls = [meshUrl, recruitUrl, privacyUrl];
export const papers: Paper[] = source.map((paper, index) => ({
  paper_id: paper.paper_id,
  title: paper.title_zh.replace("（合成示例）", ""),
  canonical_title: paper.canonical_title,
  authors: ["原创合成示例"],
  year: null,
  venue: "教学演示资料",
  tags: paper.tags.split(";").filter(tag => /[\u3400-\u9fff]/.test(tag)).slice(0, 3),
  summary: paper.method_summary,
  files: [{ file_id: paper.paper_id, name: paper.pdf_file, url: urls[index] }],
}));
export const exampleQueries = [
  { query: "DemoMesh怎样补全缺失读数？", label: "理解论文方法", paper_id: "demo_mesh" },
  { query: "DemoRecruit怎样选择参与者？", label: "查找招募规则", paper_id: "demo_recruit" },
  { query: "DemoShield服务端接收什么？", label: "核对隐私机制", paper_id: "demo_privacy" },
];
function matchingPaper(query: string): Paper | undefined {
  const q = query.toLocaleLowerCase();
  if (/demomesh|扩散|残差|缺失|插值|学习率/.test(q)) return papers[0];
  if (/demorecruit|招募|参与者|覆盖|预算/.test(q)) return papers[1];
  if (/demoshield|隐私|加噪|扰动/.test(q)) return papers[2];
  return papers.find(p => p.title.toLocaleLowerCase().includes(q));
}
export function searchDemoPapers(query: string): Paper[] {
  if (!query.trim()) return papers;
  const match = matchingPaper(query);
  return match ? [match] : papers.filter(p =>
    [p.title, p.canonical_title, p.summary, ...p.tags].join(" ").toLocaleLowerCase().includes(query.toLocaleLowerCase()));
}
/** Static UI fixture; this deliberately does not simulate model verification. */
export function answerDemo(request: AnswerRequest): AnswerResult {
  const normalized = request.query.trim().replace(/[？?。\s]/g, "");
  const supported = [...exampleQueries.map(item => item.query), "DemoMesh的学习率是多少？"];
  if (!supported.some(query => query.replace(/[？?。\s]/g, "") === normalized))
    throw new Error("这个问题尚未配置界面示例。请选择示例问题；真实检索将在后端接入后开放。");
  const paper = matchingPaper(request.query);
  if (!paper) throw new Error("这个问题尚未配置界面示例。请选择下方示例问题；真实检索将在后端接入后开放。");
  if (request.paper_ids.length && !request.paper_ids.includes(paper.paper_id))
    throw new Error("该示例问题不在已选论文范围内。请切换论文范围，或选择对应的示例问题。");
  if (/学习率|优化器/.test(request.query)) return {
    status: "refused", claims: [], citations: [],
    reason: "示例资料没有提供学习率或优化器。这是预设拒答展示，未调用模型。",
    verification: { status: "demo" }, source: "demo",
  };
  const texts: Record<string, string[]> = {
    demo_mesh: [
      "DemoMesh 先根据已观测的时间戳进行线性插值，得到缺失读数的初步估计。",
      "随后，残差扩散模型利用邻近传感器学习修正量，最终读数由初步估计与预测残差相加得到。",
    ],
    demo_recruit: [
      "DemoRecruit 按“新增覆盖的网格数 ÷ 付款金额”计算候选参与者的选择优先级，每轮选择预算内比值最高者。",
      "每次选择后更新已覆盖网格；当没有负担得起且能增加覆盖的参与者时停止。示例没有声称全局最优。",
    ],
    demo_privacy: [
      "DemoShield 在设备端先裁剪传感读数，再加入本地拉普拉斯噪声。",
      "服务器只接收扰动后的值并计算均值，不接收原始个体读数。示例没有给出经过证明的组合隐私保证。",
    ],
  };
  const original = source.find(p => p.paper_id === paper.paper_id)!;
  return {
    status: "answered",
    claims: texts[paper.paper_id].map(text => ({ text, citations: ["C1"] })),
    citations: [{
      citation_id: "C1", paper_id: paper.paper_id, file_id: paper.files[0].file_id,
      paper_title: paper.title, page_number: 1, section: original.pages[0].heading,
      chunk_id: "demo-fixture:" + paper.paper_id + ":p1",
      excerpt: original.pages[0].paragraphs.slice(1).join("\n\n"),
    }],
    reason: null, verification: { status: "demo" }, source: "demo",
  };
}
