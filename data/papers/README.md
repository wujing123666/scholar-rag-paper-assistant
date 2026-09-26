# ScholarRAG 种子论文库

本目录用于开发“根据模糊描述找回已读论文”的论文级检索功能。当前包含 11 个 PDF 文件，对应 10 篇独立论文；`TCDI (4).pdf` 与 `TCDI中文版_v22_学生五步噪声预测明确版.pdf` 是 TCDI 的两个修订版本。

## 目录结构

- `inbox/`：个人论文 PDF。目录内容被 Git 忽略，仅保留 `.gitkeep`。
- `paper_catalog.csv`：论文级档案，作为 Paper Profile 的人工可读输入。
- `eval_queries.jsonl`：模糊找论文的种子评测问题。
- `chunk_eval_queries.jsonl`：正文检索开发集，标注目标论文、相关 PDF 页码和证据词。
- `answer_eval_queries.jsonl`：生成式问答开发集，标注参考答案、必备事实和证据页码。

## 当前语料概况

当前论文集中在以下主题：

- 稀疏移动群智感知（Sparse Mobile Crowdsensing）
- 时空数据与多变量时间序列缺失值插补
- 条件扩散模型、残差扩散和扩散采样加速
- 在线用户招募、任务分配和预算控制
- 图拓扑、局部到全局相关性及概率不确定性
- 跨域迁移、冷启动、CORAL 特征对齐和卡尔曼融合

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

当前共有 39 条开发数据：30 条已知论文种子问题，每篇独立论文 3 条；另外有 9 条 `expected_paper_id=null` 的近领域未知论文问题，用于开发拒答阈值。已知题由论文内容生成，未知题由系统构造，都不是独立盲测，不能直接作为最终简历指标。最终测试集应补充本人从记忆写出的已知和未知问题，并避免根据系统结果反向修改测试题。

未知论文问题使用 `split=dev_rejection` 和 `source=system_authored_negative`。它们故意包含“扩散模型”“强化学习”“群智感知”等库内常见词，但描述的是库中没有收录的研究，例如扩散模型生成人脸、强化学习控制机器人。这样比完全无关的关键词更能暴露强行匹配问题。

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

搜索默认启用未知论文拒答。不同检索器使用各自的开发阈值；返回 JSON 中的 `rejected`、`top_score` 和 `min_score` 说明本次判定。可以临时覆盖阈值或查看未经过滤的原始排序：

```powershell
python -m src.paper_assistant --retriever bm25 --min-score 9.0 search "使用强化学习控制机器人的论文"
python -m src.paper_assistant --retriever bm25 --disable-rejection search "使用强化学习控制机器人的论文"
```

当前默认值为 BM25 `8.0`、Dense `0.78`、Hybrid `2/61`。它们来自当前 30 条已知开发题和 9 条未知开发题，只是小语料起点；论文库、Embedding 模型或 RRF 参数变化后必须重新校准。

如果查询明确写出档案中的 `paper_id` 简称，例如 `TCDI`、`DEMI` 或 `MapT-STC`，Hybrid 检索会把对应论文提升到首位并越过开放集阈值。这条确定性规则用于处理用户已经记得论文简称、只是继续追问方法细节的场景。

论文级 Dense 向量存储在本地 Chroma collection `paper_profiles_v1`，默认目录为 `data/db/chroma/`。每条记录对应一个 `paper_id`，并记录 Paper Profile 内容哈希、Embedding 模型名称和向量维度。首次运行会写入全部论文向量；再次启动时复用未变化的向量，只重新计算新增或修改的论文，并删除目录中已经移除的论文记录。

本地开发使用默认的 `PersistentClient`。代码也支持通过同一个 `ChromaStore` 接口连接服务器模式：

```powershell
python -m src.paper_assistant --retriever dense --chroma-mode server --chroma-host localhost --chroma-port 8000 search "使用强化学习招募用户的论文"
```

服务器模式要求 Chroma Server 已经启动。本地与服务器模式使用相同的 `paper_profiles_v1` 数据契约，Dense、Hybrid 和 MCP 检索逻辑不需要随部署方式改变。

## 检索论文正文块

论文级索引负责从整个目录中选择候选论文；正文索引使用独立的 Chroma collection `paper_chunks_v1`，负责在候选论文中定位原文证据。每个正文块保存 `paper_id`、PDF 文件名、PDF 页码、章节、块序号、PDF 哈希和正文内容哈希。

自动执行“论文级路由 → 正文块检索”：

```powershell
python -m src.paper_assistant --retriever bm25 --model-cache data/models/fastembed search-chunks "哪篇论文用CORAL生成伪历史数据，并用卡尔曼滤波融合专用模型和泛化模型，它具体怎么做" --candidate-papers 3 --top-k 5
```

已经知道目标论文时，可以限定 `paper_id`：

```powershell
python -m src.paper_assistant --model-cache data/models/fastembed search-chunks "卡尔曼滤波如何根据不确定性融合两路预测" --paper-id mapt_stc_2026 --top-k 5
```

默认按页处理双栏 PDF，以 1200 个字符为目标块大小、180 个字符重叠；块不会跨越 PDF 页，因此返回页码可以直接核验。程序过滤重复页眉、页脚、低文本质量坐标轴和 References 部分，并保留最近的章节标题。中文查询中的常见科研术语会追加透明的英文别名，然后结合正文 BM25 与 Dense 得分；自动路由时再加入论文级排名先验，降低其他论文中的泛化术语块压过目标论文证据的概率。

每篇逻辑论文当前只选择第一个实际存在的登记 PDF 建立正文索引，避免同一论文的多个修订版本重复召回。首次运行会构建全部正文向量；后续运行根据 PDF 哈希、分块参数、分块规则版本和 Embedding 模型复用未变化的块。本地实测 10 篇论文得到 506 个正文块，第二个进程全部复用。

`search-chunks` 只返回可供生成模型使用的证据上下文，包括论文标题、PDF、页码、章节、原文和检索分数。需要生成中文答案并逐条绑定引用时，使用下文的 `answer` 命令。

需要改善前几条证据的顺序时，可以显式启用本地 ONNX Cross-Encoder：

```powershell
python -m src.paper_assistant --retriever bm25 --model-cache data/models/fastembed --chunk-reranker fastembed --reranker-cache data/models/fastembed search-chunks "MapT-STC如何根据不确定性融合两路预测" --top-k 5
```

默认使用 MIT 许可的 `BAAI/bge-reranker-base`，模型约 1.04 GB，第一次运行下载到本地模型缓存。重排默认关闭；开启后先完成论文路由，再用原问题和最多三个术语子查询召回正文，使用加权 RRF 融合，并从默认 14 个候选中统一重排。候选和最终证据都会限制同一论文页的重复，同时保留第一候选论文中代表不同子问题的方法或公式片段。默认融合 35% Cross-Encoder 分数与 65% 原检索分数；运行时推理失败会返回多样化后的原排序，并在 JSON 的 `reranker_fallback` 中说明原因。

## 基于证据回答问题

`answer` 命令复用相同的“论文路由 → 多查询正文检索 → 可选重排”链路，默认选择 7 个跨页证据块和最多 16000 字上下文，再交给已有的可插拔 LLM。自动路由首先经过论文级未知问题阈值，弱匹配不会进入生成阶段。模型先拆解问题并遍历全部证据，必须返回结构化 Claim，每条 Claim 至少引用一个 `C1`、`C2` 等证据编号；程序会拒绝无引用结论、未知编号、非法 JSON 和模型主动报告的证据不足。最终引用由程序绑定到 `paper_id + pdf_file + page_number + section + chunk_id`，不会直接信任模型生成的来源信息。

先复制一份仅保存在本地的 LLM 配置：

```powershell
Copy-Item config/settings.yaml config/settings.local.yaml
```

`config/settings.local.yaml` 和 `config/settings.*.local.yaml` 已被 Git 忽略。使用 DeepSeek 时把其中 `llm.provider` 改为 `deepseek`、`llm.model` 改为实际模型；密钥可以放在这份私有配置的 `llm.api_key`，也可以通过环境变量 `DEEPSEEK_API_KEY` 提供。使用 Ollama 时改为 `ollama` 和本机已有的模型。不要把真实密钥写入公开配置。

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed --chunk-reranker fastembed answer "MapT-STC如何根据不确定性融合两路预测？" --settings config/settings.local.yaml
```

返回 JSON 包含 `answer`、逐条 `claims`、实际使用的 `citations`、模型名和 token 用量。没有候选论文、正文 Chunk 数量不足、LLM 调用失败或输出未通过引用校验时，系统不会拼凑答案，而是返回 `status=insufficient_evidence` 或明确的配置错误。

默认模式能保证每条输出Claim绑定到真实返回的Chunk。需要更保守的实时回答时，可开启双模型Claim–Evidence门控：

```powershell
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed --chunk-reranker fastembed --reranker-cache data/models/fastembed answer "MapT-STC如何根据不确定性融合两路预测？" --settings config/settings.deepseek.local.yaml --verify-claims consensus --claim-judge-model deepseek-chat --claim-judge-model deepseek-v4-pro
```

程序先检查Claim、Citation ID、Chunk和来源元数据是否真实对应，再让两个模型分别判断每条Claim能否由它声明的完整引用直接推出。只有两个模型都支持的Claim才进入最终答案；不一致或共同拒绝的Claim会在原候选论文中按Claim文本补检一次，复判仍未共同通过就删除。全部Claim被删除时返回证据不足。`single`模式只使用第一个`--claim-judge-model`，`off`关闭语义门控。输出中的`claim_verification`记录原始/保留/删除数量、补检结果、实际评审模型、错误和额外token。

MapT-STC真实烟雾测试中，候选答案有10条Claim；关于“卡尔曼融合泛化模型与专用模型输出”的Claim最初被Flash和Pro共同拒绝。系统在同一论文内补检到第6、14、15页的明确证据后，两模型复判均支持，最终保留10条Claim并追加对应引用；双模型检查额外使用7,922 token。该案例证明“检查→补检→复判→保留”的闭环可运行，不代表所有问题都能通过安全门控。

15题开发集完整运行中，14题回答、1题因证据不足拒答，共生成169条Claim；两位评审在首轮就共同支持全部169条，因此本轮没有触发删除或补检，Claim保留率为100%。论文Top-1、候选召回和59项必备事实覆盖仍分别为100%、100%和72.88%。实时双审额外使用82,148 token，使总token从同类冻结基线的106,891增至189,721，平均延迟从约7.68秒增至10.69秒。该结果验证安全链路没有破坏本轮答案完整性，但也显示把双审设为每次回答的默认流程成本较高；当前保留为可选严格模式。

## 生成式问答开发集

`answer_eval_queries.jsonl` 当前包含 15 条可回答问题，覆盖全部 10 篇逻辑论文，其中 10 条用于检查各论文的核心方法，5 条进一步检查容易混淆的方法细节。每行是一条独立 JSON：

```json
{"id":"rdpi_answer_01","question":"RDPI为什么把初始确定性插补与真实缺失值之间的残差作为扩散目标？","expected_paper_id":"rdpi_2025","relevant_pages":[1,3,4],"reference_answer":"……","required_facts":["……"],"split":"dev","source":"system_labeled_from_pdf","reviewed_by_user":false}
```

字段含义：

- `question`：交给 RAG 系统回答的自然语言问题。
- `expected_paper_id`：应当提供答案证据的论文；用于检查论文路由是否正确。
- `relevant_pages`：人工从本地 PDF 核对过的相关 PDF 页码；页码从 1 开始。
- `reference_answer`：严格依据这些页面写成的参考答案，用于人工评审和后续答案相似度评测。
- `required_facts`：合格答案至少应覆盖的原子事实，可用于计算事实覆盖率。
- `split=dev`：允许在开发时反复运行和分析，不能作为最终未见测试集报告。
- `source=system_labeled_from_pdf`：问题和答案由系统根据 PDF 构造，并非作者发布的数据集，也不是用户亲自撰写的盲测题。
- `reviewed_by_user=false`：用户尚未逐条确认；确认后才能改成 `true`。

这 15 条数据适合评估论文路由、正文召回、回答完整度和引用页码是否正确。它们不适合单独证明系统对真实用户问题的泛化能力，因为编写时已经看过论文和当前检索结构。最终简历指标应另留一批不参与调参的本人盲测问题，并补充库外问题来评估拒答能力。

使用 Hybrid 论文路由、本地 Cross-Encoder 重排和私有 DeepSeek 配置运行整套评测：

```powershell
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed --chunk-reranker fastembed --reranker-cache data/models/fastembed evaluate-answers --settings config/settings.deepseek.local.yaml --output tmp/answer_evaluation_deepseek.json
```

使用同一个 DeepSeek API 配置、但让另一个模型独立评审：

```powershell
python -m src.paper_assistant --retriever hybrid --model-cache data/models/fastembed --chunk-reranker fastembed --reranker-cache data/models/fastembed evaluate-answers --settings config/settings.deepseek.local.yaml --judge-model deepseek-v4-pro --output tmp/answer_evaluation_cross_model.json
```

此时生成仍使用私有配置中的模型（当前请求别名为`deepseek-chat`，接口实际返回`deepseek-flash`），事实覆盖和Claim支撑均由`deepseek-v4-pro`判定。DeepSeek V4模型默认开启高强度思考；程序对结构化评审调用明确传入`thinking.type=disabled`，避免长推理内容增加延迟或破坏JSON协议。API Key和Base URL继续复用同一份被Git忽略的私有配置。

评测器每完成一题就原子写入本地报告，可在相同命令末尾增加 `--resume` 从已有结果继续。报告包含论文 Top-1 正确率、候选论文召回率、回答成功率、必备事实覆盖率、标注页对齐率、Claim 语义支撑率、延迟和 token 用量。生成完成后会有两次独立判分：一次检查答案是否覆盖必备事实；另一次把每条 Claim 与它实际引用的完整 Chunk 对照，只有全部实质性内容能由引用直接推出才算支撑。页码对齐率只衡量引用是否落在人工登记页，Claim 支撑率才衡量引用内容是否支持结论。两个指标均由同一 LLM 严格判分，仍存在模型偏差，不能替代人工抽检。完整回答、逐 Claim 结果和评测报告默认写入被 Git 忽略的 `tmp/`。

当前 15 条开发题使用 Hybrid 路由、BGE 重排、7 个证据块和 16000 字上下文的结果为：15/15 成功回答，论文 Top-1 与候选召回均为 100%，59 个必备事实覆盖 43 个（72.88%），标注页召回 85.29%，平均延迟 5.43 秒。改造前同集事实覆盖为 62.71%、标注页召回为 55.88%。这是一组系统看过语料后构造的开发集结果，不能当作用户盲测准确率。

加入 Claim-Evidence 判分后的一次独立运行中，14/15 题成功回答，成功答案共有 168 条 Claim，其中 167 条被其引用原文直接支撑，Claim 支撑率为 99.40%；13/14 个成功答案的全部 Claim 均获支撑。唯一未支撑项来自 CoFILL 的 Q/K/V 关系，另有一题由生成模型主动返回证据不足。该次判分额外使用 39,252 token；高支撑率来自带强引用约束的生成协议和同一模型评审，仍需用异构模型或人工抽检验证。

改用`deepseek-v4-pro`作为不同模型评审的一次运行中，14/15题成功回答，59个必备事实覆盖43个（72.88%）；175条Claim全部被Pro判为有证据支撑，判分失败0次、拒答后跳过1次，平均总延迟7.65秒。Pro与Flash虽是不同模型，但仍属于同一DeepSeek模型家族，因此100%只能作为交叉模型开发指标，不能替代人工抽检或不同供应商评审。

为避免不同生成结果干扰评审模型对比，`evaluate-answers`现在会把每条Claim实际引用的完整Chunk写入本地报告。可以让多个模型只复判同一份冻结答案，不再执行检索或生成：

```powershell
python -m src.paper_assistant compare-judges --input tmp/frozen_answer_evaluation.json --settings config/settings.deepseek.local.yaml --judge-model deepseek-chat --judge-model deepseek-v4-pro --output tmp/frozen_judge_comparison.json --audit-output tmp/codex_claim_audit.md --agreement-sample 20 --seed 20260926
```

命令会报告逐Claim判断、一致率和实际响应模型，并把全部分歧项、全部共同拒绝项及固定随机种子抽取的共同支持项写入审查Markdown。审查标签可再通过`score-judge-audit`计算各模型相对审查结果的准确率、错误放行和错误拒绝；标签文件必须记录审查者、审查类型、限制、结论与理由，避免把AI代理复核写成人类专家金标准。

需要比较不同API服务商时，为每个评审准备一份本地忽略的设置文件，并使用不同标签：

```powershell
python -m src.paper_assistant compare-judges --input tmp/frozen_answer_evaluation.json --judge-config deepseek-pro=config/settings.deepseek-pro.local.yaml --judge-config qwen-plus=config/settings.qwen.local.yaml --output tmp/cross_vendor_judge_comparison.json --audit-output tmp/cross_vendor_claim_audit.md --agreement-sample 20 --seed 20260927
```

`--judge-config`读取各自配置中的provider、model、API Key和base URL；私有设置文件由`config/settings.*.local.yaml`忽略。一次DeepSeek V4 Pro与Qwen Plus的真实冻结复判中，170条Claim全部成功判分，169条一致、1条分歧，一致率99.41%。唯一分歧项的引用原文直接描述了“保留部分观测项作为目标、其余作为输入并训练模型重构目标”，Codex代理核验标记为有证据支持，因此这一次是Qwen误拒。该核验仅覆盖模型分歧触发的1条，不是随机样本，也不是人类金标准。

一次冻结运行得到14个回答和170条Claim。Flash与Pro对167条判断一致，对3条存在分歧，一致率98.24%；Flash支持167条，Pro支持170条。Codex随后核验全部3条分歧项和随机抽取的20条共同支持项：23条中22条有直接证据、1条引用不足。在这组有意偏向风险项的审查样本上，Flash与审查结果一致21/23，包含2次错误拒绝和0次错误放行；Pro一致22/23，包含0次错误拒绝和1次错误放行。该样本不是随机无偏的人类标注集，因此这些比例用于定位评审差异，不能宣称为模型总体准确率。

## 评测正文证据检索

正文开发集当前包含 10 条问题，每篇逻辑论文 1 条。每条记录保存目标 `paper_id`、一个或多个相关 PDF 页码、证据词和最低证据词命中数。它们由程序根据本地 PDF 定位后逐页核对，`source=system_labeled_from_pdf` 且 `reviewed_by_user=false`，因此是可迭代的开发集，不是用户盲测集。

只评测已知目标论文内部的 Chunk 排序，用于隔离正文检索问题：

```powershell
python -m src.paper_assistant --model-cache data/models/fastembed evaluate-chunks --oracle-paper
```

评测完整的“BM25 论文路由 → 候选论文正文检索”：

```powershell
python -m src.paper_assistant --retriever bm25 --model-cache data/models/fastembed evaluate-chunks
```

报告分别给出论文候选 Recall、正确页的 Recall@1/3/5 与 MRR，以及同时满足页码和证据词条件的 Evidence Recall@1/3/5 与 MRR。逐字核对 STEI 摘要后，第 1 页也被认定为能完整支撑自适应系数问题的相关证据页。修正后的首批 10 条开发题基线为：论文路由 Recall=1.00，Page/Evidence Recall@1=0.30、Recall@3=0.50、Recall@5=1.00、MRR=0.520。

启用默认 Cross-Encoder 后，Recall@1=0.30、Recall@3=0.70、Recall@5=1.00、MRR=0.553。10 条题的独立命令行运行时间在当前 CPU 环境约从 2.0 秒增至 8.0 秒；常驻服务只需加载一次模型。扩大重排候选池曾使 Recall@5 降低，因此默认只重排原 Top-5。

这些数字只反映当前 10 条系统构造开发题，问题数量较少，而且术语扩展曾根据失败分析迭代，不能作为盲测成绩或最终准确率。后续需要由本人独立编写并冻结新的正文问题，再报告独立测试结果。

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

当前 30 条同源种子问题的结果如下。这些问题来自同一批论文内容，只适合验证流程和做开发期消融，不能作为最终准确率：

| 检索器 | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| 加权 BM25 | 100.0% | 100.0% | 1.000 |
| BGE Dense | 73.3% | 90.0% | 0.824 |
| BM25 + Dense + RRF | 86.7% | 96.7% | 0.925 |

这组小数据上 BM25 的精确术语匹配最强，RRF 没有超过 BM25 的 Top-1。这个结果会被保留，而不是为了得到更好看的数字在同源题上反复调参。后续扩大论文库并增加独立盲测问题后，再判断混合检索是否带来稳定收益。

启用开发阈值后，对 30 条已知题和 9 条未知题得到：

| 检索器 | 阈值后已知 Recall@1 | Rejection Accuracy | Open-set Accuracy |
|---|---:|---:|---:|
| 加权 BM25 | 96.7% | 100.0% | 97.4% |
| BGE Dense | 70.0% | 66.7% | 69.2% |
| BM25 + Dense + RRF | 73.3% | 77.8% | 74.4% |

这里的 Open-set Accuracy 把“已知题第一名正确”和“未知题成功拒答”都计为正确。阈值和指标使用同一批开发数据，因此只能说明实现链路和当前取舍；不能当作未见数据上的泛化结果。Dense 和 Hybrid 的结果也表明，简单分数阈值仍无法可靠区分所有近领域未知问题。

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

工具返回人类可读文本和结构化结果，包括 `rejected`、`rejection_reason`、`top_score`、`min_score`，以及通过门槛后的 `paper_id`、中英文标题、作者、年份、期刊或会议、匹配词、方法摘要和本地 PDF 文件名。同一论文的多个 PDF 版本会出现在一个论文结果下。未达到门槛时 `results` 为空，并提示用户补充方法、数据集、作者、年份或期刊等线索。

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

## 生成待确认的论文档案

对 `inbox/` 中的一篇 PDF 生成候选 Paper Profile：

```powershell
python -m src.paper_assistant prepare-paper "data/papers/inbox/新论文.pdf"
```

草稿默认写入 `data/papers/drafts/新论文.json`。如需覆盖已有草稿，显式添加 `--force`；也可以用 `--output` 指定其他本地路径。

第一版不调用大模型。程序从 PDF 元数据、首页排版和前五页原文中提取标题、作者、年份、摘要、DOI 和语言，并为每个字段保存来源、页码、原文证据和置信度。低置信度候选会列入 `review_required_fields`，但不会自动填入 `proposed_profile`。例如 PDF 创建年份不一定是论文发表年份，双盲稿的作者行也可能只有 `Anonymous authors`。`tags`、`method_summary`、`datasets`、`memory_cues` 等语义字段保持为空，等待后续模型生成或本人填写。

生成草稿前还会重新运行文件盘点。已经登记的文件标记为 `already_registered`；与其他文件字节完全相同的 PDF 标记为 `duplicate_review_required`；普通新文件标记为 `needs_review`。命令不会自动修改 `paper_catalog.csv`。

如果 PDF 没有可选择的文字，草稿会提示可能需要 OCR。目前不会根据扫描图片猜测论文信息。

## 人工确认后受控写入目录

生成草稿后，先在本地 JSON 的 `proposed_profile` 中核对并补全信息。全新论文至少需要：

- `paper_id`：稳定的小写标识，只使用字母、数字、点、下划线和连字符；
- `pdf_file`、`canonical_title` 和 `language`；
- `tags`、`method_summary` 和 `memory_cues`，保证新增论文具备可检索线索。

作者、年份、期刊和数据集允许暂时留空，但预览会给出提醒。先运行不带确认参数的只读预览：

```powershell
python -m src.paper_assistant --catalog data/papers/paper_catalog.csv import-paper data/papers/drafts/新论文.json --inbox data/papers/inbox
```

只有预览返回 `status=ready` 后，才执行明确确认：

```powershell
python -m src.paper_assistant --catalog data/papers/paper_catalog.csv import-paper data/papers/drafts/新论文.json --inbox data/papers/inbox --confirm
```

确认写入会依次执行：

1. 校验草稿版本和 `needs_review` 状态；
2. 拒绝已经登记的 PDF、已有 `paper_id`、重复标题和完全重复文件；
3. 重新计算 inbox 中 PDF 的 SHA-256 和文件大小，识别草稿生成后的文件替换；
4. 在本地 `data/papers/backups/` 保存原 CSV 备份；
5. 生成临时 CSV，并使用 `PaperCatalog` 完整加载验证；
6. 再次确认正式 CSV 没有被其他进程修改，然后原子替换。

默认只支持全新论文。给已有论文增加新 PDF 版本需要单独的版本导入流程，避免把不同论文误合并。备份、草稿和 PDF 均由 Git 忽略。导入成功后，下一次初始化 Dense 检索器时会根据 Paper Profile 哈希只补算新增论文向量。

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

当前已经实现论文级 `PaperProfile`、加权 BM25、本地 BGE Dense、论文级与正文块 Chroma 持久化、两级论文路由与页级正文 Hybrid 检索、阈值拒答、RRF 融合、版本去重、候选档案生成、受控增量入库及 Recall@1、Recall@3、MRR、Rejection Accuracy 和 Open-set Accuracy 评测。下一阶段将实现：

1. 基于正文证据调用生成模型，输出逐条绑定论文、页码、章节和原文的回答；
2. 扩展并冻结本人编写的正文盲测问题，独立报告 Chunk Recall@K；
3. 扩展到约 100 篇本人读过的论文；
4. 为已有论文增加受控版本导入和档案更新；
5. 按新语料重新校准拒答阈值，并加入平均查询延迟和 P95 延迟评测。
