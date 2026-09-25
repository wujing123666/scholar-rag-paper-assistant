# ScholarRAG 本人盲测问题填写模板

## 填写顺序

1. 不查看 `paper_catalog.csv`、已有的 `eval_queries.jsonl`、论文目录、论文摘要和检索结果。
2. 一次性写完全部 9～18 条“模糊描述”，期间不要核对任何论文信息。
3. 保存这版描述。从此不再润色、补充或删除检索关键词。
4. 所有描述写完后，才打开本地论文目录，统一补充每条的“目标论文”。不要修改已经写好的描述。
5. 最后填写难度、线索类型和个人备注，并在“是否冻结”处写“是”。
6. 冻结后不根据系统返回结果修改问题。

目标论文可以填写你认识的 PDF 文件名、论文标题或简称，不要求你知道 `paper_id`。后续由程序维护者映射成标准答案。请先写完所有模糊描述，再统一填写所有目标论文。

## 描述要求

- 建议每条 20～100 个汉字，像平时向同学求助一样自然表达。
- 可以写研究任务、核心方法、数据集、实验现象、年份、期刊、图表印象或个人阅读记忆。
- 不要复制论文标题、摘要原句、文件名或作者全名。
- 尽量不要使用能直接唯一定位标题的方法缩写，例如已经知道目标论文叫 RDPI 时，不要直接写“找 RDPI”。
- 一条问题最好对应一篇明确的目标论文。如果你认为两篇都可能正确，在备注中说明。
- 每篇论文至少写 1 条；最好写 2 条，一条较容易，一条更模糊。

## 可直接复制的空白表单

```text
【盲测问题 BQ-01】
模糊描述（必须先写）：
目标论文（描述写完后再填，写文件名或标题均可）：
难度（easy / medium / hard）：
线索类型（可多选：task / method / dataset / result / year / venue / figure / personal_memory）：
这条记忆来自哪里（例如组会、导师推荐、写论文时看过，可留空）：
是否冻结（是 / 否）：是

【盲测问题 BQ-02】
模糊描述（必须先写）：
目标论文（描述写完后再填，写文件名或标题均可）：
难度（easy / medium / hard）：
线索类型（可多选：task / method / dataset / result / year / venue / figure / personal_memory）：
这条记忆来自哪里（可留空）：
是否冻结（是 / 否）：是
```

继续复制同一段，编号写到 BQ-09 至 BQ-18。

## 合格示例

```text
【盲测问题 BQ-01】
模糊描述（必须先写）：我记得有篇交通缺失值论文把一种传统的迭代估计方法和扩散模型放到了一起，好像还能给出不确定性。
目标论文（描述写完后再填，写文件名或标题均可）：物联网期刊稀疏群智感知3【CAJ文档翻译】.pdf
难度（easy / medium / hard）：medium
线索类型：task / method / personal_memory
这条记忆来自哪里：读论文时记得它不是普通的条件扩散结构
是否冻结（是 / 否）：是
```

这个示例没有写论文标题或方法缩写，但保留了真实记忆中的任务和方法关系。

## 不合格示例

```text
帮我找 RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation。
```

这实际上是标题检索，不能测试“凭模糊记忆找论文”。

```text
那篇使用 PEMS-BAY 的论文。
```

这条信息过少，当前库中多篇论文都使用 PEMS-BAY，没有唯一正确答案。应再加入一个真实记得的方法、结果或阅读情境线索。

## 程序最终保存格式

填写完成后，维护者会转换为 JSONL：

```json
{"id":"blind_001","description":"我记得有篇交通缺失值论文把一种传统的迭代估计方法和扩散模型放到了一起，好像还能给出不确定性。","expected_paper_id":"demi_2025","acceptable_pdfs":["物联网期刊稀疏群智感知3【CAJ文档翻译】.pdf"],"difficulty":"medium","cue_types":["task","method","personal_memory"],"split":"blind_test","source":"user_memory","reviewed_by_user":true,"frozen":true}
```

检索程序只接收 `description`，不会看到 `expected_paper_id` 和 `acceptable_pdfs`；后两个字段只在检索完成后用于评分。
