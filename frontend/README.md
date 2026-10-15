# ScholarRAG Web · 真实论文 RAG 工作台

React + TypeScript 前端通过 FastAPI/Pydantic 接入现有 `PaperAssistantService`，复用两级检索、生成和 Claim 核验。默认是真实模式，不会在服务失败时偷偷返回预设答案。上传入库尚未开放。

## 本地运行

使用 Python 3.12、Node.js 24 和 pnpm 11（具体依赖见项目文件）。先在仓库根目录启动一个后端进程：

```bash
python -m pip install -e ".[web,rerank]"
python -m src.web_api --port 18603
```

服务使用**已经准备好的本地索引**。默认个人论文目录为 `data/papers/paper_catalog.csv`、`data/papers/inbox` 和 `data/db/chroma`。缺文件、档案/向量模型/维度/分块版本不一致时拒绝启动，不自动嵌入、删除或重建向量。新克隆请先按主 README 和[公开样例说明](../examples/paper_demo/README.md)准备资料与索引。

LLM 配置通过 `SCHOLARRAG_SETTINGS` 指向私有 YAML；Embedding 可另设 `SCHOLARRAG_EMBEDDING_SETTINGS`，未指定时沿用该 YAML。本机存在 `config/settings.deepseek.local.yaml` 时将它作为默认，否则使用公开 `config/settings.yaml`。不会在浏览器中要求输入 Key，也不会自动加载 `.env`。

然后在另一个终端启动前端：

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev --port 18602
```

打开 `http://127.0.0.1:18602/`。Vite 把 `/api` 代理到 `127.0.0.1:18603`。两者默认只监听本机。

```bash
pnpm test
pnpm build
pnpm exec openapi-typescript openapi.json -o src/api/generated.ts
```

`pnpm build` 包含 TypeScript 检查。构建后重启后端，可直接从 `http://127.0.0.1:18603/` 使用同源页面，不再需要 Vite。TypeScript 固定为 5.9.3，兼容 OpenAPI 生成器的 Compiler API；`pnpm-lock.yaml` 固定依赖解析。

## 可以体验什么

- 问答工作台：自由提问、找论文、限定最多三篇论文；查看核验后的结论、真实拒答或服务错误。
- 引用联动：点引用打开服务端映射的 PDF 文件页；支持翻页、缩放、下载和对照原文片段。没有段落坐标，不提供精确高亮。
- 论文库：读取当前真实目录、按标题/方法/关键词筛选、阅读或限定单篇提问；一次最多显示五十项，可筛选缩小范围。
- 知识问答记录：最多保留二十条，按资料库和浏览器会话隔离；演示与真实记录分开。打开旧记录不再次调用模型。已完成的找论文结果不写入问答历史。
- 后台任务：真实 queued/running/completed/failed/queue_timeout/interrupted 状态。刷新恢复同一 job；响应丢失时沿用幂等编号，服务重启不自动重跑模型。
- 窄屏布局：导航抽屉、上下排列的问答/阅读区；点引用滚动到原文。
- 入库流程说明：上传按钮保持禁用，尚未实现文件上传与索引写入。

## 显式界面演示模式

只有 URL 加 `?demo=1` 才使用预设答案，不调用 API。这个模式仅支持以下问题，不能作为真实 RAG 效果评测：

1. DemoMesh怎样补全缺失读数？
2. DemoRecruit怎样选择参与者？
3. DemoShield服务端接收什么？
4. DemoMesh的学习率是多少？（预设拒答）

演示资料来自 [`examples/paper_demo`](../examples/paper_demo/README.md)，是原创合成材料。正常 URL 的真实模式使用服务端配置的论文库；API 配置成合成库时，仍会进行真实检索与模型调用，页面会展示该资料库名称。

## 代码与后续接线

| 入口 | 作用 |
| --- | --- |
| `src/App.tsx` | 页面导航、提交生命周期和引用选择 |
| `src/components/PdfReader.tsx` | 按需加载 PDF.js、渲染当前页、清理阅读资源 |
| `src/components/AnswerView.tsx` | 逐条结论和来源展示，纯文本渲染 |
| `src/api/contracts.ts` / `src/api/generated.ts` | 来自服务端 OpenAPI 的类型，另保留显式 demo 标识 |
| `src/api/client.ts` | 统一 HTTP 客户端、幂等提交、任务轮询与恢复检查 |
| `src/mocks/demo.ts` | 显式预设的界面样例 |
| `src/storage/history.ts` | 有上限且校验结构的浏览器历史 |

后端代码在 [`src/web_api`](../src/web_api/)。修改契约后，在仓库根目录运行 `python -m scripts.export_web_openapi`，再生成前端类型。CI 检查 OpenAPI 与生成代码未漂移。

## 运行边界

- 一个 API 进程共享问答 Gate（默认 3 个执行槽、20 个等待槽）和搜索 Gate（8/20），没有另套无限队列。`run`/`submit` 使用同一 Gate 实现。
- API 状态目录有进程锁；两个进程共用同一状态目录时，第二个启动失败。不能通过换状态目录再开多个实例来声称限流仍全局统一。迁移期 Streamlit 只作调试，不与新页面同时作为生产入口。
- 回执保存在 `SCHOLARRAG_WEB_STATE`（默认 `data/web`）下的资料库子目录，默认保留二十四小时、上限两千条，达到上限返回单独的容量提示。可用 `SCHOLARRAG_JOB_TTL_SECONDS` / `SCHOLARRAG_JOB_LIMIT` 调整。终态到期清理，活动任务不因 TTL 被删除。
- 浏览器自动恢复窗口为六小时；会话 Cookie 丢失、资料库变化或超过恢复期限时不自动重提。客户端停止轮询或切换阅读不会强制终止已开始的模型调用。
- 无账号登录；随机 HttpOnly Cookie 隔离任务归属，Origin 白名单保护提交。它适用于可信本机/实验室网络，不是面向公网的账户权限系统。
- 模型失败/输出无效/核验服务不可用与“论文证据不足”区分显示；不会拿预设成功答案补位。
- 页面切换仍使用组件状态，暂不支持地址直达或浏览器后退恢复页面。上传、草稿确认和索引写入留待下一阶段。

服务端路径可通过 `SCHOLARRAG_CATALOG`、`SCHOLARRAG_INBOX`、`SCHOLARRAG_CHROMA_PATH` 配套设置；不能把小 CSV 指向大库 Chroma。规模库可以配置使用，但不会自动把它与个人库混合。索引更新期间应停止 API，离线准备后再重启；当前接口不提供写入或热切换。

资料库身份包含目录与全部已登记 PDF 的内容指纹；同名文件更新后使用新身份，旧历史不会静默打开新版本原文。直接阅读默认打开第一份文件，回答引用则打开对应的具体文件版本。

API Key 只留在服务端，不得写入 `VITE_*` 或前端源码。开发时 Vite 文件访问只允许前端和公开示例目录；真实 PDF 由服务端根据 `paper_id/file_id` 查找，客户端不能传绝对路径。
