# ScholarRAG 种子论文库

本目录用于开发“根据模糊描述找回已读论文”的论文级检索功能。当前包含 10 个 PDF 文件，对应 9 篇独立论文；`TCDI (4).pdf` 与 `TCDI中文版_v22_学生五步噪声预测明确版.pdf` 是 TCDI 的两个修订版本。

## 目录结构

- `inbox/`：个人论文 PDF。目录内容被 Git 忽略，仅保留 `.gitkeep`。
- `paper_catalog.csv`：论文级档案，作为 Paper Profile 的人工可读输入。
- `eval_queries.jsonl`：模糊找论文的种子评测问题。

## 当前语料概况

当前论文集中在以下主题：

- 稀疏移动群智感知（Sparse Mobile Crowdsensing）
- 时空数据与多变量时间序列缺失值插补
- 条件扩散模型、残差扩散和扩散采样加速
- 在线用户招募、任务分配和预算控制
- 图拓扑、局部到全局相关性及概率不确定性

这批论文主题接近，适合测试系统区分相似论文，而不只是依靠标题关键词命中。

## `paper_catalog.csv` 字段

| 字段 | 含义 |
|---|---|
| `paper_id` | 同一论文跨版本共享的稳定标识 |
| `pdf_file` | `inbox/` 下的 PDF 文件名 |
| `canonical_title` / `title_zh` | 英文规范标题与中文标题 |
| `authors` / `year` / `venue` | 基本书目信息 |
| `tags` | 适合 BM25 和筛选的主题标签，分号分隔 |
| `method_summary` | 论文核心方法的短摘要 |
| `datasets` | 主要实验数据集 |
| `memory_cues` | 根据“读后可能记住什么”编写的检索线索 |
| `duplicate_group` | 同一论文不同文件版本的分组标识 |
| `read_date` / `rating` | 留给本人填写的阅读日期和评分 |
| `notes_source` | 说明当前笔记来源，防止把自动摘要误当成人工笔记 |

`memory_cues` 已根据 PDF 内容生成初稿。建议后续加入你自己的印象，例如“导师推荐”“组会讲过”“和毕业论文第二章相关”。个人线索往往比摘要更适合找回读过的论文。

## `eval_queries.jsonl` 字段

每行是一条独立 JSON：

```json
{"id":"rdpi_2025_02","description":"我记得它先用确定性模型粗补，再用扩散模型学习残差。","expected_paper_id":"rdpi_2025","acceptable_pdfs":["第四篇（cc让下）.pdf"],"difficulty":"medium","cue_types":["method","residual"],"split":"dev","source":"generated_from_pdf","reviewed_by_user":false}
```

关键字段：

- `description`：不能直接复制论文标题，应模拟真实模糊记忆。
- `expected_paper_id`：正确论文身份。
- `acceptable_pdfs`：允许命中的文件列表。同一论文存在多个版本时，任一版本都算命中。
- `difficulty`：`easy`、`medium` 或 `hard`。
- `cue_types`：描述使用的方法、数据集、年份、结果等线索。
- `split`：当前两条用于开发调参，一条为候选测试题。
- `reviewed_by_user`：本人核对后改为 `true`。

当前共 27 条种子问题，每篇独立论文 3 条。它们由论文内容生成，可用于开发检索流程，但不能直接作为最终简历指标。BM25 在这批同源问题上得到 100% Recall@1，只能说明链路和数据映射正确。最终测试集应补充本人从记忆写出的独立问题，并避免根据系统检索结果反向修改测试题。

## 版本重复的处理

两份 TCDI 文件内容相似但文件哈希不同，属于同一论文的不同修订版本。索引时应：

1. 为两个文件保留各自的 `file_id` 和文件哈希；
2. 共享 `paper_id=tcdi`；
3. 论文级检索只展示一个论文结果；
4. 在详情页列出可用版本；
5. 评测命中任一 TCDI 文件都视为找到正确论文。

这能避免同一论文的多个版本占据 Top-K。

## 运行论文级检索

根据模糊描述返回前三篇论文：

```powershell
python -m src.paper_assistant search "我记得它先粗补，再用扩散模型学习残差" --top-k 3
```

Dense 与 Hybrid 模式使用本地中文向量模型。首次运行前安装可选依赖：

```powershell
pip install -e ".[local]"
```

首次运行会下载约 90 MB 的 `BAAI/bge-small-zh-v1.5` ONNX 模型，后续从本地缓存加载：

```powershell
python -m src.paper_assistant --retriever dense --model-cache data/models/fastembed search "使用扩散模型做插补的论文" --top-k 3
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed search "使用强化学习招募用户的论文" --top-k 3
```

运行全部种子问题评测：

```powershell
python -m src.paper_assistant evaluate
```

只评估开发集或候选测试集：

```powershell
python -m src.paper_assistant evaluate --split dev
python -m src.paper_assistant evaluate --split test_candidate
```

比较三种检索器时，只改变 `--retriever`：

```powershell
python -m src.paper_assistant --retriever bm25 evaluate
python -m src.paper_assistant --retriever dense --model-cache data/models/fastembed evaluate
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed evaluate
```

当前 27 条同源种子问题的结果如下。这些问题来自同一批论文内容，只适合验证流程和做开发期消融，不能作为最终准确率：

| 检索器 | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| 加权 BM25 | 100.0% | 100.0% | 1.000 |
| BGE Dense | 74.1% | 88.9% | 0.831 |
| BM25 + Dense + RRF | 85.2% | 100.0% | 0.920 |

这组小数据上 BM25 的精确术语匹配最强，RRF 没有超过 BM25 的 Top-1。这个结果会被保留，而不是为了得到更好看的数字在同源题上反复调参。后续扩大论文库并增加独立盲测问题后，再判断混合检索是否带来稳定收益。

检索以 `paper_id` 为单位。TCDI 的两个 PDF 版本会合并成一个候选结果，但结果中仍会列出两个可用文件。

## 通过 MCP 查找论文

MCP Server 会自动注册 `find_paper` 工具。客户端只需传入模糊描述：

```json
{
  "query": "帮我找使用了扩散模型加切比雪夫的论文",
  "top_k": 3,
  "retriever": "bm25"
}
```

工具返回人类可读文本和结构化结果，包括 `paper_id`、中英文标题、作者、年份、期刊或会议、匹配词、方法摘要和本地 PDF 文件名。同一论文的多个 PDF 版本会出现在一个论文结果下。

`retriever` 可选 `bm25`、`dense` 或 `hybrid`。默认使用 `bm25`，启动快且不需要加载向量模型；只有明确选择 `dense` 或 `hybrid` 时才加载本地 Embedding。

## 盘点本地论文目录

新增论文前先运行只读盘点：

```powershell
python -m src.paper_assistant inventory
```

需要保留完整报告时：

```powershell
python -m src.paper_assistant inventory --output data/papers/inventory.json
```

盘点会递归扫描 `inbox/` 下的 PDF，逐文件流式计算 SHA-256，并检查：

- PDF 是否已经登记到 `paper_catalog.csv`；
- CSV 引用的 PDF 是否真实存在；
- 不同文件是否具有完全相同的内容哈希；
- 同一 `paper_id` 是否包含多个 PDF 版本；
- 同一路径是否被错误分配给多个 `paper_id`。

命令只读取 PDF 和 CSV，不会修改目录或论文档案。默认只在终端输出不含绝对路径和文件哈希的检查摘要。状态为 `ok` 时退出码为 0；发现需要人工处理的问题时状态为 `attention_required`，退出码为 1。使用 `--output` 保存的完整报告包含本地绝对路径、文件名、大小和哈希，应继续保存在本地；`data/papers/inventory.json` 已由现有忽略规则排除在 Git 之外。

## 推荐评测方法

第一版至少报告：

- Recall@1：第一名是否为目标论文；
- Recall@3：前三名是否包含目标论文；
- MRR：正确论文的平均倒数排名；
- Rejection Accuracy：描述不属于论文库时能否拒绝错误匹配；
- 平均查询延迟与 P95 查询延迟。

应分别比较 BM25、Dense、BM25 + Dense + RRF、RRF + Reranker 四种方案。

## 数据与版权

PDF 仅用于个人科研和本地实验，不应随公开仓库分发。公开 GitHub 仓库时只提交目录结构、合法公开的元数据和自行编写的评测描述，并检查论文标题、作者和年份是否需要进一步人工校正。

## 下一步实现

当前已经实现论文级 `PaperProfile`、加权 BM25、本地 BGE Dense、RRF 融合、版本去重及 Recall@1、Recall@3、MRR 评测。下一阶段将实现：

1. 未知论文拒答和相似度阈值；
2. 扩展到约 100 篇本人读过的论文；
3. 增量导入、文件哈希与档案更新；
4. 增加由本人独立编写并冻结的盲测问题；
5. 加入查询延迟、P95 延迟和拒答准确率评测。
