# 可运行的公开论文演示

这套资料用于验证“下载仓库后能跑通什么”，不属于正式论文库或独立盲测。DemoMesh、DemoRecruit、DemoShield 和各自数据集均为虚构；PDF 每页都标注 SYNTHETIC DEMO。原文和标签由同一份示例资料构造，不能拿演示准确率写成线上能力指标。

包含 3 份两页英文 PDF、中英混合论文档案、6 条已知论文查询、2 条库外查询及 6 条页级证据查询。PDF 文本源在 [papers.json](papers.json)，可阅读、修改和重新生成，没有复制第三方论文全文。

从仓库根目录执行下列命令。先按[根 README](../../README.md#快速开始)安装依赖；以下 `python` 指已安装本项目的 Python 环境。

## 1. 不需要 Key 的离线检查

```powershell
python -m scripts.check_paper_demo
```

检查实际 PDF 解析、稳定 Chunk ID、页码标签与关键词论文排序。结果写到 `tmp/paper_demo/offline_report.json`；不创建 Chroma，不下载模型，不调用 API，也不生成答案。BM25 门槛显式为 0，仅检查关键词找回和“无匹配词”负例，不能验证正式拒答门槛或语义质量。

## 2. 使用自己 API 的完整问答

```powershell
$env:DEEPSEEK_API_KEY = "自己的生成模型密钥"
$env:DASHSCOPE_API_KEY = "自己的Qwen密钥"
$env:SCHOLARRAG_SETTINGS = "examples/paper_demo/settings.yaml"
python -m streamlit run examples/paper_demo/app.py
```

打开终端显示的网址，选择“论文问答”。第一次问答会为这三份示例创建 Qwen 向量，调用生成模型，并做单模型 Claim 核验；需要 API 额度。实际流程复用 [PaperAssistantService](../../src/paper_assistant/service.py) 和正式页面，没有另写模拟 RAG。

为减少首次安装的模型下载，此演示的论文与正文重排为 none，Claim 为 single；常规页面仍保留 fastembed / consensus 默认值。检索使用 Hybrid 与现有 RRF 门槛，正文取证和回答均为实时调用，不是预存答案回放。不同模型/服务版本的表述和拒答可能变化。

建议依次试：

1. **DemoMesh怎样补全缺失读数？** 预期找到 DemoMesh，主要引用 PDF 第 1 页 I. METHOD。展开 C1 可读原文，并下载 PDF 核对页码。
2. **DemoMesh的学习率是多少？** PDF 未提供学习率，预期拒答。2026-10-15 实测返回 `model_reported_insufficient_evidence`。
3. **DemoRecruit怎样选择参与者？** 预期定位新增覆盖/付款的选择规则，不应编造最优性证明。
4. **DemoShield服务端接收什么？** 预期回答接收扰动值，不应声称具有已证明的组合隐私保证。

真实运行截图：[回答与原文](../../docs/assets/scholarrag-grounded-answer.jpg)、[证据不足拒答](../../docs/assets/scholarrag-refusal.jpg)。这些仅证明对应示例当次运行，不是准确率评测。

## 3. 实时检索评测（消耗 Qwen 额度）

```powershell
python -m scripts.evaluate_paper_demo --settings examples/paper_demo/settings.yaml
```

复用正式论文检索门控、正文检索器及页级评测计算。生成 `tmp/paper_demo/live_retrieval.json`，包含逐题候选、返回页、Page Recall 和证据词匹配；该命令不生成回答、不评审 Claim。没有使用 oracle 指定正确论文，候选由论文检索器选择。

[一次真实 Qwen 运行记录](../../docs/evaluation/data/public-demo-retrieval.json)已公开。样例中 Page Recall@1 为 4/6，@3 为 6/6，说明排第一的片段不一定是所需页面；这些数字仅作流程示例。

## 隔离与重建

演示启动器固定使用本目录的 CSV/PDF，向量库、Claim 缓存和请求日志位于 `tmp/paper_demo/`，不会替换正式目录或索引。已有私有 YAML 可通过 `SCHOLARRAG_SETTINGS` 指定，不要将真实密钥写进公开示例。

重新生成 PDF 与目录仅需额外安装 reportlab：

```powershell
python -m pip install reportlab==4.4.9
python examples/paper_demo/generate.py
python -m scripts.check_paper_demo
```

修改资料后应同时审查问题、页码与证据词标签。生成器固定元信息以保持同一 ReportLab 版本的输出稳定。它不负责自动生成可信评测金标。

本目录原创合成文字、标签及 PDF 可按 MIT 条款使用；不代表对项目继承内容的来源或授权作出额外声明。
