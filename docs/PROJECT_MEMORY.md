# ScholarRAG 项目改造记忆

这份文件记录项目从原始开源仓库到个人论文检索助手的完整改造过程。它既是开发日志，也是面试复习材料。后续每完成一个有意义的功能、修复、评测或 PR，都在“工作记录”末尾追加一条，不覆盖以前的内容。

## 项目定位

目标是把通用的 Modular RAG MCP Server 改造成个人论文助手 ScholarRAG。用户预计维护约 100 篇本人读过的论文。当用户只记得论文的大概内容、方法、实验数据或个人阅读印象时，系统应返回正确论文，而不是只回答论文中的某个知识点。

核心检索对象从原项目的 `Chunk` 调整为 `PaperProfile`：一篇论文可以包含多个 Chunk，也可以存在多个 PDF 修订版本，但论文检索结果中只占一个位置。

## 当前状态

- GitHub 仓库：<https://github.com/wujing123666/scholar-rag-paper-assistant>
- 上游仓库：<https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER>
- 已完成 PR：[#1 Windows 运行修复与种子数据](https://github.com/wujing123666/scholar-rag-paper-assistant/pull/1)、[#2 论文级 BM25 基线](https://github.com/wujing123666/scholar-rag-paper-assistant/pull/2)
- 本地语料：10 个 PDF 文件，对应 9 篇独立论文；TCDI 有两个修订版本
- 公开数据：10 条论文档案记录、27 条生成式种子查询；PDF 不进入 Git
- 已实现：Windows 兼容修复、论文档案、论文级版本去重、加权 BM25、Recall@1/Recall@3/MRR 评测、命令行检索
- 盲测采集模板：`docs/templates/blind-query-template.md`
- 最近验证：1208 个单元测试通过，1 个跳过；新模块 Ruff 检查通过
- 当前可信结论：BM25 检索闭环已经跑通
- 当前不能声称：真实用户查询准确率达到 100%。现有 27 条问题与档案来自同一批论文内容，只是开发基线

## Git 与仓库关系

```text
upstream = 原作者仓库 jerry-ai-dev/MODULAR-RAG-MCP-SERVER
origin   = 个人仓库 wujing123666/scholar-rag-paper-assistant
```

个人仓库保留 Fork 关系，明确说明基于开源项目二次开发。自己的贡献通过独立分支、语义化提交和 Pull Request 展示，便于面试官查看修改范围。

通常采用以下工作流：

```bash
git switch main
git pull origin main
git switch -c feature/<feature-name>
git add <related-files>
git commit -m "feat: ..."
git push -u origin feature/<feature-name>
# 创建 PR，检查变更和测试结果后合并 main
```

本项目第一次上传时，本机 Git HTTPS 到 GitHub 的连接被重置，但 GitHub CLI API 正常。因此实际操作是通过 GitHub GraphQL `createCommitOnBranch` 将与本地提交相同的文件快照写入远程分支，再创建和合并 PR。远程提交 SHA 与本地 SHA 因提交元数据不同而不一致，文件内容一致。面试时如果只问常规开发流程，可以讲上面的分支和 PR 流程；如果明确追问这次实际上传方式，应如实说明网络故障和 API 备用方案。

## 数据与隐私规则

- 论文 PDF 只在 `data/papers/inbox/` 本地保存。
- `.gitignore` 忽略 `data/papers/inbox/*`，仅提交 `.gitkeep`。
- 可以公开自行整理的论文元数据、检索问题、代码和评测方法。
- 提交前使用 `git status`、`git check-ignore` 和远程 Git tree 检查，确认 PDF 未进入 GitHub。
- 自动生成的问题不能冒充本人独立编写的盲测数据。

## 工作记录

### 2026-09-25：建立个人仓库和可审查的 Git 工作流

**目标**

把下载到本地的原项目变成用户自己的长期改造仓库，让面试官能看到贡献过程，同时保留原作者署名与历史。

**起始状态**

- 本地仓库的 `origin` 指向原作者地址。
- 还没有用户自己的 GitHub 仓库。
- 本地分支为 `fix/runtime-and-tests`，存在尚未提交的修复。
- GitHub CLI 尚未安装，Git 用户名和邮箱也未配置。

**实施过程**

1. 安装 GitHub CLI，并通过设备验证码登录 `wujing123666`。
2. 将原作者远程仓库改名为 `upstream`。
3. 使用 GitHub Fork API 建立个人公开仓库 `scholar-rag-paper-assistant`，保留来源关系。
4. 将个人仓库配置为 `origin`。
5. 使用 `wujing123666@users.noreply.github.com` 作为仓库级 Git 邮箱，避免暴露私人邮箱。
6. 将运行修复和论文数据拆成两个提交，通过 PR #1 合并到 `main`。
7. 在仓库说明中明确写明这是 ScholarRAG 开发版，并链接原作者仓库。

**特殊情况**

普通 `git push` 多次出现 GitHub 443 连接被重置。为了不丢失分支和 PR 记录，改用 GitHub GraphQL API创建远程提交。这个备用方案保留了相同的文件变更、独立提交和 PR 审查过程。

**结果**

- 个人仓库创建完成。
- `origin` 和 `upstream` 职责清晰。
- PR #1 已合并，公开修改过程可查看。
- 临时创建的空仓库已改为私有并归档，防止与正式仓库混淆。

**面试表达**

> 我 Fork 了原项目，保留原仓库作为 upstream，把个人仓库作为 origin。开发时从 main 建功能分支，将基础修复和业务改造拆成独立提交，测试后创建 PR，再合并到 main。这样既保留开源来源，也能清楚展示自己的贡献。

**下一步**

修复原项目在当前 Windows 和依赖版本下的运行问题，建立稳定基线。

### 2026-09-25：修复 Windows 运行、资源释放和接口兼容问题

**目标**

让原项目在 Windows、Python 3.12 和当前依赖环境下可以启动，并使不依赖真实云服务的测试稳定通过。

**发现的问题**

- 根目录 `main.py` 仍是早期占位入口，没有真正启动 MCP Server。
- `pyproject.toml` 缺少运行期实际使用的包，部分依赖版本范围过宽。
- Windows 下 ChromaDB 的 SQLite 文件句柄未释放，测试结束时临时目录无法删除。
- 多个 MCP 工具并发创建同一路径的 Chroma PersistentClient，存在共享客户端竞态。
- 空集合查询仍调用远程 Embedding API，既浪费请求又会在无密钥环境报错。
- Azure/OpenAI 配置可能收到 mock 或空值，环境变量和配置文件优先级不清晰。
- 短耗时阶段使用低分辨率计时，在 Windows 上可能得到 0 ms。
- 中英文混合词如 `gpt-4`、`deep_learning` 会被错误拆分。
- LLM 精炼失败后虽然回退到规则处理，但没有稳定记录回退原因。
- Dashboard 只能读取新版 Trace 结构，旧 Trace 数据可能显示不完整。

**主要修改**

- `main.py` 与 `pyproject.toml`
  - 把入口指向 `src.mcp_server.server:main`。
  - 增加并约束 `langchain-community`、`mcp`、`pymupdf`、`ragas`、`openai` 等运行依赖。
  - 补充图像测试 marker。
- `src/libs/vector_store/chroma_store.py`
  - 增加 `_ensure_open()`、`close()`、上下文管理和析构清理。
  - 每次操作前确保客户端已打开。
  - 清空集合后主动关闭客户端，释放 SQLite/HNSW 文件句柄。
- `src/libs/vector_store/chroma_lock.py`
  - 增加进程内 `RLock`，串行化 Chroma 客户端初始化。
- MCP 工具
  - 查询工具缓存同一集合的向量库，切换集合时释放旧客户端。
  - 使用初始化锁避免并发竞态。
  - 空集合直接返回空结果，不调用 Embedding API。
  - 列表和摘要工具也统一处理 Chroma 创建与关闭。
  - 自定义 ProtocolHandler 已有工具时不重复注册默认工具。
- Embedding 适配
  - 过滤空字符串、Mock 和 sentinel 值。
  - 明确参数、环境变量和配置文件的优先级。
- 摄取与可观测性
  - Pipeline 关闭时释放图片库、完整性数据库和向量库。
  - Trace 和 EvalRunner 改用 `perf_counter_ns()`。
  - SparseEncoder 改善中英文混合词分词。
  - ChunkRefiner 在 LLM 失败回退时写入 `refine_fallback_reason`。
  - TraceService 同时兼容新旧 Trace 字段布局。
- 测试清理
  - Chroma 集成测试在临时目录删除前显式调用 `store.close()`。

**验证过程**

```powershell
python -m pytest -q tests/unit
```

结果：1205 passed，1 skipped。

随后选择无需真实账号的集成与端到端测试，第一次得到 107 passed、2 skipped、1 error。唯一错误是 Windows 上 `chroma.sqlite3` 仍被占用。为测试 fixture 增加显式 `store.close()` 后重跑：107 passed，2 skipped。

完整的 `pytest -m "not llm"` 曾得到 16 failed、1323 passed、10 skipped、5 deselected、9 errors。失败主要来自测试文件没有正确标记外部依赖，却要求 Azure/OpenAI/Ollama 配置或真实 Embedding。没有把这次结果说成“全部通过”。

**Git 记录**

- 本地提交：`51c4149 fix: make runtime and tests reliable on Windows`
- 远程对应提交位于 PR #1

**面试表达**

> 我先建立可重复运行的基线，重点解决 Windows 下 ChromaDB 文件锁、并发初始化、资源生命周期和空集合误调用外部 API。随后把测试分成纯本地测试和需要云端凭据的测试，纯本地单元测试与选择后的集成测试都通过；需要真实账号的测试单独说明环境前提，没有把它们混入通过率。

**下一步**

整理真实论文数据，先定义论文身份、版本关系和可复现的检索问题。

### 2026-09-25：建立 ScholarRAG 种子论文库与评测数据

**目标**

把 10 个本地 PDF 整理成适合“根据模糊记忆找论文”的结构化数据，同时保证论文原文不上传到公开仓库。

**数据分析**

- `data/papers/inbox/` 中有 10 个 PDF 文件。
- 实际对应 9 篇独立论文。
- `TCDI (4).pdf` 和 `TCDI中文版_v22_学生五步噪声预测明确版.pdf` 是同一论文的两个修订版本，共享 `paper_id=tcdi`。
- 论文主题相近，集中在稀疏移动群智感知、时空插补、扩散模型、任务分配和预算控制，适合检验模型区分相似论文的能力。

**新增文件**

- `data/papers/paper_catalog.csv`
  - 10 行文件记录、9 个唯一 `paper_id`。
  - 包含标题、作者、年份、期刊、标签、方法摘要、数据集、记忆线索、重复版本分组等字段。
- `data/papers/eval_queries.jsonl`
  - 27 条问题，每篇论文 3 条。
  - 分为 easy、medium、hard，各 9 条。
  - 18 条为 `dev`，9 条为 `test_candidate`。
  - 使用 `expected_paper_id` 判断论文级命中，用 `acceptable_pdfs` 容纳多个版本。
- `data/papers/README.md`
  - 说明字段、版本去重、指标、版权边界和后续路线。
- `.gitignore`
  - 允许提交元数据和说明，继续忽略所有 PDF。

**关键决定**

检索目标使用稳定的 `paper_id`，文件版本另设 `pdf_file`。如果直接把 10 个 PDF 当成 10 个独立结果，TCDI 会占据两个 Top-K 位置，导致论文级结果重复，也会让评测失真。

27 条问题由论文内容生成，只用于开发检索链路。`reviewed_by_user=false` 明确表示尚未由本人核对，不能直接作为简历最终指标。

**安全检查**

提交前检查 Git 状态和忽略规则；合并后又通过 GitHub tree API 检查远程 `data/papers/`，确认只有 README、CSV、JSONL 和 `.gitkeep`，没有任何 PDF。

**Git 记录**

- 本地提交：`a87a2de feat: seed ScholarRAG paper retrieval dataset`
- 与运行修复一起通过 PR #1 合并

**面试表达**

> 我没有直接把 PDF 分块后就开始调模型，而是先定义论文级数据模型。通过 paper_id 把同一论文的多个修订版合并，并设计了模糊描述、标准答案和允许命中文件列表。原始 PDF 保持本地，公开仓库只保存元数据和自行编写的评测描述。

**下一步**

先实现无需付费模型的 BM25 论文级基线，验证整个检索和评测闭环。

### 2026-09-25：实现论文级加权 BM25 检索基线

**目标**

实现第一个可以实际使用的闭环：读取论文档案、合并重复版本、输入模糊描述、返回 Top-K 论文，并自动计算 Recall@1、Recall@3 和 MRR。

**为什么不直接复用原 Chunk 检索**

原系统以 Chunk 为检索单位。一篇论文会产生大量 Chunk，同一论文的多个片段可能同时进入 Top-K；两个 PDF 修订版也会彼此竞争。论文助手需要先回答“是哪篇论文”，所以新增独立的论文级检索层。

**新增模块**

- `src/paper_assistant/catalog.py`
  - 定义 `PaperProfile` 和 `PaperCatalog`。
  - 按 `paper_id` 合并 CSV 行。
  - 聚合 PDF 文件、标签、方法摘要、数据集和记忆线索。
- `src/paper_assistant/retriever.py`
  - 实现混合中英文分词和加权 BM25。
  - 标题与标签权重为 3；方法、数据集和记忆线索权重为 2；作者、年份和期刊权重为 1。
  - 使用正值 Robertson/Sparck Jones IDF，避免只有 9 篇论文时常见领域词产生负分。
  - 返回分数、匹配词和对应的所有 PDF 版本，方便解释结果。
- `src/paper_assistant/evaluation.py`
  - 加载 JSONL 测试集。
  - 以 `expected_paper_id` 计算排名。
  - 输出整体指标和难度分组。
- `src/paper_assistant/__main__.py`
  - 提供 `search` 和 `evaluate` 两个命令。
- `tests/unit/test_paper_assistant.py`
  - 验证多 PDF 合并、记忆线索检索和指标计算。
- `docs/evaluation/bm25-seed-baseline.md`
  - 固化实验设计、结果和局限。

**使用方法**

```powershell
python -m src.paper_assistant search "我记得它先用确定性模型粗补，再用扩散模型学习残差" --top-k 3
python -m src.paper_assistant evaluate
python -m src.paper_assistant evaluate --split dev
python -m src.paper_assistant evaluate --split test_candidate
```

真实示例中，上述残差扩散描述将 `rdpi_2025` 排在第一位，同时给出匹配词和本地 PDF 文件名。

**评测结果**

| 数据 | 数量 | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Dev | 18 | 1.000 | 1.000 | 1.000 |
| Test candidate | 9 | 1.000 | 1.000 | 1.000 |
| 全部种子问题 | 27 | 1.000 | 1.000 | 1.000 |

这不是最终准确率。档案文本与问题都从同一批论文内容生成，共享明显关键词，结果偏乐观。正确结论是“论文级检索、版本去重和评测链路均已工作”，不能说“真实模糊查询准确率为 100%”。

**代码验证**

```powershell
python -m ruff check src/paper_assistant tests/unit/test_paper_assistant.py
python -m pytest -q tests/unit
```

结果：Ruff 通过；1208 passed，1 skipped。

**Git 记录**

- 本地提交：`3536fce feat: add paper-level BM25 retrieval baseline`
- 远程提交：`9775945`
- PR：[#2 Add paper-level BM25 retrieval baseline](https://github.com/wujing123666/scholar-rag-paper-assistant/pull/2)
- PR 状态：已合并，远程 merge commit 为 `28d78f1`

**面试表达**

> 我先做了一个论文级 BM25 基线，而不是一开始就叠加向量模型。系统把标题、主题、方法、数据集和个人记忆线索设置不同权重，并按 paper_id 合并多个 PDF 版本。这样后面引入 Dense 和 RRF 时有可解释、可复现的对照组。种子集得到满分，但因为测试问题与档案同源，我只把它当作链路验证，不把它包装成真实准确率。

**下一步**

由用户脱离论文档案，根据真实记忆独立写 9～18 条问题并冻结为盲测集。之后实现 Dense 论文检索与 RRF 融合，对比 BM25、Dense、Hybrid 三种方案。

### 2026-09-25：建立持续维护的项目改造记忆

**目标**

把此前分散在对话、Git 提交和 PR 中的信息整理为一份长期文档，帮助用户复盘完整改造过程并准备面试。

**实施过程**

- 新增本文件 `docs/PROJECT_MEMORY.md`，补录仓库建立、运行修复、种子数据和 BM25 基线四个阶段。
- 每个阶段同时记录动机、问题、实现、验证结果、限制、Git 信息和面试表达。
- 新增根目录 `AGENTS.md`，规定以后每完成一个有意义的阶段，都必须在同一次修改中追加本文件。
- 在根 README 顶部增加本文件入口，方便从 GitHub 首页查阅。

**关键决定**

项目记忆采用只追加的开发日志，而不是只维护一段会被反复覆盖的总结。历史结论即使后来被修正也保留，并在新条目中说明变化，这样能还原真实的工程决策过程。

**验证**

- 检查文档中的本地文件名、提交 SHA、PR 链接和测试数字与现有 Git 记录一致。
- 使用 `git diff --check` 检查空白符错误。

**面试表达**

> 我为项目维护了一份工程决策日志。每个阶段不仅记录改了哪些文件，还记录为什么改、如何验证、有哪些限制，以及下一步实验如何由当前证据推导出来。这样可以避免只记住结果却讲不清排查过程。

**下一步**

每完成一次实现、修复、评测或 PR，继续按照文末模板追加记录。

### 2026-09-25：定义本人盲测问题采集协议

**目标**

建立一批能够反映本人真实模糊记忆、且没有被系统输出污染的测试问题，为后续 BM25、Dense 和 RRF 对比提供可信基准。

**为什么不能继续由程序生成**

当前 `eval_queries.jsonl` 和 `paper_catalog.csv` 都来自同一批论文内容，容易共享“50 步变 5 步”“Los-loop”“双强化学习”等显著关键词。即使检索达到 100%，也不能证明系统能理解用户真实、含糊、不完整的记忆。盲测问题必须由用户脱离档案和检索结果独立书写。

**采集规则**

1. 先关闭论文档案、已有测试集和检索界面。
2. 仅凭记忆写下自然语言描述。
3. 描述写完后再确认目标 PDF 或标题，不能反向补关键词。
4. 每篇论文至少 1 条，建议 2 条；首批共 9～18 条。
5. 问题冻结后再运行系统；看到结果后不得修改原问题。
6. 如果问题确实有多个合理答案，提前在备注中说明，评测时配置多个可接受答案。

**字段设计**

- `description`：真正传给检索系统的模糊描述。
- `expected_paper_id`：标准论文身份，只供评分使用。
- `acceptable_pdfs`：允许命中的版本文件。
- `difficulty`：用户在检索前判断的难度。
- `cue_types`：任务、方法、数据集、结果、年份、期刊、图表或个人记忆。
- `split=blind_test`：与生成式开发问题隔离。
- `source=user_memory`：说明问题来自本人记忆。
- `reviewed_by_user=true`、`frozen=true`：确认已核对且不能按检索结果调题。

**填写模板**

使用 `docs/templates/blind-query-template.md`。用户只需要填写中文表单，不需要手写 JSON。填写完后再转换为 JSONL，并在首次运行前记录文件哈希和冻结时间。

**面试表达**

> 为避免用论文摘要生成问题造成数据泄漏，我把生成式问题只用于开发，把本人脱离原文写出的模糊描述单独作为 blind_test。问题先冻结，再运行检索；模型只能看到 description，标准答案字段只供评测脚本计算排名指标。

**下一步**

用户按照模板填写 9～18 条问题。收到后先校验歧义并冻结，不根据任何检索结果改写，然后分别评测 BM25、Dense 和 RRF。

## 下一步路线

1. 建立本人独立编写的盲测集，避免同源数据泄漏。
2. 选择本地或云端 Embedding，建立论文级 Dense 索引。
3. 用 Reciprocal Rank Fusion 合并 BM25 与 Dense 排名。
4. 对 BM25、Dense、RRF 做消融实验并报告分难度指标。
5. 增加 `find_paper` MCP 工具，让 Codex、Claude Desktop 等客户端直接调用。
6. 扩展到约 100 篇论文，增加增量导入、哈希去重和个人阅读笔记。
7. 在可靠盲测集上形成可写入简历的指标与案例。

## 后续记录模板

```markdown
### YYYY-MM-DD：阶段名称

**目标**

**起始状态或发现的问题**

**实施过程与主要文件**

**关键技术决定**

**验证命令与结果**

**已知限制**

**Git 记录**

**面试表达**

**下一步**
```
