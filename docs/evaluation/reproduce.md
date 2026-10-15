# 公开评测与复现

提供三个不同层次的入口，不把算术回放、示例跑通和正式语料实验混为一谈。

## 一、离线复算历史指标

```powershell
python -m scripts.recalculate_benchmarks
```

输入 [historical-evidence.json](data/historical-evidence.json)；输出 `tmp/public_benchmark_replay.json`。无需 API 或论文 PDF。

公开记录保留了 30 个配对问题的方面命中排名、14 个冻结答案的评审 Token/缓存计数、140 条论文门控判定，以及独立模型的 Claim 判定总数。刻意不导出问题原文、答案、PDF 片段、文件绝对路径、阅读记录和密钥。

计算方式：

- **Coverage@7**：排名不大于 7 的可覆盖方面数 / 所有可覆盖方面数。整句为 59/72，方面检索为 65/72；差值为 **8.33 个百分点**。从已舍入的 81.94% 与 90.28% 相减会得到 8.34，应以原始计数为准。
- **全部方面覆盖**：该题所有可覆盖方面都进入 Top-7 的问题数 / 问题数。
- **Page Recall@K**：该题首个相关页命中的排名不大于 K 的问题比例，不是所有标注页的集合召回率。
- **评审 Token 降幅**：`(82148 - 72081) / 82148`，即 12.25%。冷缓存记录仍保留 1 次命中，未把其改造成严格零命中实验。
- **已知/未知门控**：分别统计库内通过与库外拒答，不能合成“回答正确率”。
- **模型完全支持比例**：130/132 = 98.48%；分母是最终答案中的 Claim，判定来自模型筛查，不是专家金标或系统无幻觉保证。

导出由 [export_benchmark_evidence.py](../../scripts/export_benchmark_evidence.py) 的字段白名单完成；源报告 SHA-256 保存在公开 JSON，用于有原件时对照版本。哈希和公开记录不能独立证明原始标注正确，也不能代替完整实验重跑。

拥有历史本地报告时可重新导出：

```powershell
python -m scripts.export_benchmark_evidence
```

此命令依赖未发布的原始报告，普通克隆用户无需执行。当前计算复现不需要它。历史配置、样本来源和失败项见[评测记录](scholarrag-benchmarks.md)。

## 二、从公开 PDF 重跑检索

使用[公开示例包](../../examples/paper_demo/README.md)。它公开文本、PDF、CSV、查询与页码标签：

```powershell
python -m scripts.check_paper_demo
python -m scripts.evaluate_paper_demo --settings examples/paper_demo/settings.yaml
```

前者离线检查数据与关键词排序；后者使用自己的 Qwen Key，重新建索引并检索。默认 Hybrid、无重排、候选论文 1 篇、正文 Top-3、1200 字符 / 180 重叠、Qwen text-embedding-v4 1024 维。结果含逐题候选、排名、回退记录和配置。

[2026-10-15 真实运行报告](data/public-demo-retrieval.json)的页级 Recall@1=4/6、Recall@3=6/6。这是很小的合成开发集；它只展示计算方法与调用闭环，不能作为历史 +8.33 个百分点提升的再次验证，也不能证明大规模质量。

## 三、自动化回归

[GitHub Actions](../../.github/workflows/offline-checks.yml)在 push / PR 时执行单元测试、公开 PDF 离线检查及历史指标复算，不配置真实模型凭证，不运行付费在线评测。单测启用 `SCHOLARRAG_TEST_OFFLINE=1`，禁止外部 socket 连接；本机回环与 Unix socket 保留给测试框架。

依赖使用 Python 3.12 和 [requirements-ci.txt](../../requirements-ci.txt) 的直接依赖约束。该文件固定已测试版本，**不是全部传递依赖锁文件**。本地已检查依赖一致性、pip 解析与对应测试命令；GitHub 上的 workflow 必须提交后实际运行，才能判断云端是否通过。

跨操作系统差异、模型升级和 API 服务变动可能改变结果。当前完整单元回归在 Windows Python 3.12 上验证，CI 配置为 Ubuntu Python 3.12；尚未把本地通过宣称为云端已通过。
