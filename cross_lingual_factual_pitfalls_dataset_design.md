# Cross-Lingual Factual Pitfalls 数据集与实验设计方案

## 1. 研究目标

本方案旨在结合两个方向：

- **Cross-Lingual Pitfalls**：自动发现多语言大模型在跨语言场景下的弱点，即“英文能答对，但目标语言答错”的样本。
- **Paths Not Taken**：分析多语言大模型在事实回忆任务中的内部路径差异，并尝试通过中间层向量干预进行修复。

最终目标不是简单复现已有论文，而是构建一个新的分析型数据集：

> **Cross-Lingual Factual Pitfalls：面向多语言大模型的跨语言事实回忆弱点数据集。**

该数据集用于研究：

1. 多语言大模型在哪些事实性问题上存在跨语言不一致；
2. 这些错误是由于英文中心事实召回失败，还是目标语言转换失败；
3. 轻量向量干预能否修复这些跨语言事实回忆弱点；
4. 不同语言、不同模型、不同错误类型之间是否存在规律。

---

## 2. 数据来源设计

### 2.1 原始来源一：Cross-Lingual Pitfalls 数据集

Cross-Lingual Pitfalls 仓库已经提供了论文生成的 6000+ bilingual pairs，按语言拆分为多个 JSON 文件。

典型文件结构如下：

```text
Cross-Lingual-Pitfalls/
├── data/
│   ├── Chinese.json
│   ├── Japanese.json
│   ├── Korean.json
│   ├── French.json
│   ├── Spanish.json
│   ├── Arabic.json
│   ├── Hindi.json
│   ├── Swahili.json
│   ├── Yoruba.json
│   ├── Zulu.json
│   ├── Chinese_results.json
│   ├── Japanese_results.json
│   └── ...
└── data/source/
    ├── ai2_arc_easy.json
    ├── commonsense_qa.json
    ├── mmlu.json
    ├── sciq.json
    └── truthful_qa.json
```

其核心价值是：

```text
英文问题：模型平均正确率高
目标语言问题：模型平均正确率低
```

也就是说，样本本身已经具备跨语言弱点属性。

### 2.2 原始来源二：Paths Not Taken 数据范式

Paths Not Taken 更关注 factual recall，即事实三元组式的回忆任务：

```text
(subject, relation, answer)
```

例如：

```text
subject: Thailand
relation: main religion
answer: Buddhism
```

对应 prompt 可以构造为：

```text
English: The main religion in Thailand is
Chinese: 泰国的主要宗教是
Japanese: タイの主な宗教は
```

它适合分析模型在中间层是否已经激活正确答案，以及最终是否能转换成目标语言输出。

---

## 3. 新数据集定位

新数据集不是重新从零构造，而是在 Cross-Lingual Pitfalls 的基础上筛选和转换出一个子集：

```text
Cross-Lingual Pitfalls 原始 QA 样本
        ↓
筛选事实回忆型问题
        ↓
转换为 factual recall probe
        ↓
补充错误归因标签与中间层分析字段
        ↓
形成 Cross-Lingual Factual Pitfalls 数据集
```

该数据集的特点：

| 特点 | 说明 |
|---|---|
| 跨语言弱点属性 | 样本满足英文表现好、目标语言表现差 |
| 事实回忆属性 | 样本可以被转化为 subject-relation-answer 形式 |
| 机制分析友好 | 可以用 logit lens、activation patching、vector intervention 分析 |
| 可扩展 | 可按语言、模型、错误类型继续扩展 |
| 适合硕士论文 | 工作量可控，不依赖从头训练大模型 |

---

## 4. 数据筛选原则

### 4.1 需要保留的样本类型

优先保留具有明确事实答案的问题：

| 类型 | 示例 | 是否保留 |
|---|---|---|
| 地理事实 | “法国的首都是哪里？” | 保留 |
| 国家属性 | “日本使用的货币是什么？” | 保留 |
| 宗教文化 | “泰国的主要宗教是什么？” | 保留 |
| 历史人物 | “谁发明了电话？” | 保留 |
| 科学事实 | “水的化学式是什么？” | 保留 |
| 百科实体属性 | “埃菲尔铁塔位于哪个城市？” | 保留 |

### 4.2 需要剔除的样本类型

不适合事实回忆路径分析的样本应剔除或单独标记：

| 类型 | 原因 |
|---|---|
| 复杂数学推理 | 不属于简单 factual recall |
| 多步逻辑推理 | 中间路径不一定是事实回忆 |
| 纯常识判断 | 很难转成明确三元组 |
| 语言歧义题 | 可能是翻译问题，不是事实回忆问题 |
| 选项依赖型问题 | 去掉选项后无法独立回答 |
| 主观判断题 | 没有稳定标准事实答案 |

---

## 5. 数据转换设计

### 5.1 从多选 QA 转为 factual recall

原始 Cross-Lingual Pitfalls 样本形式：

```json
{
  "question": "What is the main religion in Thailand?",
  "choices": ["Buddhism", "Islam", "Christianity", "Hinduism"],
  "answer": "Buddhism",
  "transquestion": "泰国的主要宗教是什么？",
  "transchoices": ["佛教", "伊斯兰教", "基督教", "印度教"],
  "transanswer": "佛教",
  "source": "mmlu",
  "rate_ori": 1.0,
  "rate_trans": 0.2
}
```

转换后的 factual recall 样本形式：

```json
{
  "id": "clf_zh_000001",
  "language": "Chinese",
  "source_dataset": "mmlu",
  "subject_en": "Thailand",
  "relation_en": "main religion",
  "answer_en": "Buddhism",
  "subject_target": "泰国",
  "relation_target": "主要宗教",
  "answer_target": "佛教",
  "prompt_en": "The main religion in Thailand is",
  "prompt_target": "泰国的主要宗教是",
  "original_question_en": "What is the main religion in Thailand?",
  "original_question_target": "泰国的主要宗教是什么？",
  "choices_en": ["Buddhism", "Islam", "Christianity", "Hinduism"],
  "choices_target": ["佛教", "伊斯兰教", "基督教", "印度教"],
  "rate_ori": 1.0,
  "rate_trans": 0.2
}
```

### 5.2 转换方式

可采用三种方式结合：

#### 方法一：规则模板转换

适合结构明显的问题：

```text
What is the capital of X?  →  The capital of X is
What currency is used in X?  →  The currency used in X is
Who invented X?  →  X was invented by
```

优点：稳定、成本低、可控。

缺点：覆盖面有限。

#### 方法二：LLM 辅助抽取三元组

给 LLM 一个结构化输出要求：

```text
请从下面的问题中抽取 subject、relation、answer，
并判断该问题是否适合转为事实回忆任务。

问题：What is the main religion in Thailand?
选项：A. Buddhism B. Islam C. Christianity D. Hinduism
答案：Buddhism

输出 JSON：
{
  "is_factual_recall": true,
  "subject": "Thailand",
  "relation": "main religion",
  "answer": "Buddhism",
  "prompt_en": "The main religion in Thailand is"
}
```

优点：覆盖面高。

缺点：需要人工抽检，防止抽取错误。

#### 方法三：人工抽检修正

建议对每种语言至少抽检 50-100 条，确认：

1. 三元组是否正确；
2. 英文 prompt 是否自然；
3. 目标语言 prompt 是否自然；
4. answer 是否唯一；
5. 是否存在翻译歧义。

---

## 6. 数据字段设计

建议最终数据集每条样本包含以下字段。

### 6.1 基础字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| id | string | 样本唯一 ID |
| language | string | 目标语言 |
| source_dataset | string | 原始来源，如 mmlu、sciq、truthful_qa |
| category | string | 原始问题类别 |
| original_question_en | string | 原始英文问题 |
| original_question_target | string | 目标语言问题 |
| choices_en | list | 英文选项 |
| choices_target | list | 目标语言选项 |
| answer_en | string | 英文答案 |
| answer_target | string | 目标语言答案 |

### 6.2 事实回忆字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| subject_en | string | 英文主语实体 |
| relation_en | string | 英文关系 |
| subject_target | string | 目标语言主语实体 |
| relation_target | string | 目标语言关系 |
| prompt_en | string | 英文 factual recall prompt |
| prompt_target | string | 目标语言 factual recall prompt |
| factual_type | string | 事实类型，如 geography、history、science |
| is_factual_recall | bool | 是否为事实回忆样本 |

### 6.3 跨语言弱点字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| rate_ori | float | 生成阶段模型组在英文问题上的平均正确率 |
| rate_trans | float | 生成阶段模型组在目标语言问题上的平均正确率 |
| pitfall_score | float | 跨语言弱点强度，可定义为 rate_ori - rate_trans |
| is_cross_lingual_pitfall | bool | 是否满足跨语言弱点标准 |

### 6.4 模型评测字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| model_name | string | 被评测模型 |
| pred_en | string | 模型英文输出 |
| pred_target | string | 模型目标语言输出 |
| correct_en | bool | 英文是否正确 |
| correct_target | bool | 目标语言是否正确 |
| exact_match_en | bool | 英文精确匹配 |
| exact_match_target | bool | 目标语言精确匹配 |
| semantic_correct_en | bool | 英文语义正确 |
| semantic_correct_target | bool | 目标语言语义正确 |

### 6.5 机制分析字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| intermediate_answer_en_rank | int | 中间层英文答案 rank |
| intermediate_answer_target_rank | int | 中间层目标语言答案 rank |
| best_layer_en | int | 英文答案激活最强层 |
| best_layer_target | int | 目标语言答案激活最强层 |
| recall_path_success | bool | 是否成功激活英文中心事实回忆路径 |
| conversion_success | bool | 是否成功从英文答案转换到目标语言答案 |
| error_type | string | 错误归因类型 |

### 6.6 干预实验字段

| 字段名 | 类型 | 说明 |
|---|---|---|
| intervention_condition | string | orig、diff、task、diff_task |
| diff_layer | int | translation difference vector 注入层 |
| task_layer | int | recall task vector 注入层 |
| pred_after_intervention | string | 干预后输出 |
| correct_after_intervention | bool | 干预后是否正确 |
| repaired | bool | 是否由错误变为正确 |

---

## 7. 错误归因标签设计

建议将错误归因为以下几类：

| 标签 | 含义 | 判断方式 |
|---|---|---|
| recall_failure | 英文中心事实召回失败 | 中间层没有激活正确英文答案 |
| conversion_failure | 英文答案已召回，但目标语言转换失败 | 中间层英文答案正确，最终目标语言输出错误 |
| translation_ambiguity | 翻译或选项存在歧义 | 多个答案可接受或翻译不等价 |
| reasoning_failure | 需要复杂推理，不是单纯事实回忆 | prompt 转换后仍需多步推理 |
| entity_mapping_failure | 实体映射错误 | 人名、地名、术语跨语言映射失败 |
| non_factual | 非事实回忆问题 | 无法构造稳定三元组 |
| unknown | 暂无法判断 | 自动方法不确定，需人工复核 |

核心研究中最重要的是前两类：

```text
recall_failure：模型没有走上正确的事实回忆路径
conversion_failure：模型走到了英文答案，但没有正确转成目标语言答案
```

---

## 8. 数据集划分设计

### 8.1 推荐语言选择

为了控制实验规模，建议第一阶段选择 5 种语言：

| 语言 | 原因 |
|---|---|
| Chinese | 中文场景重点，与你研究方向最相关 |
| Japanese | 与中文在文字系统和东亚语境上有联系 |
| Korean | 东亚语言对照 |
| French | 欧洲语言代表 |
| Spanish | 欧洲语言代表，资源相对丰富 |

后续可以扩展 Arabic、Hindi、Swahili、Yoruba、Zulu 等语言，用于低资源语言分析。

### 8.2 推荐样本规模

MVP 阶段：

```text
每种语言 100-200 条 factual pitfall 样本
总计 500-1000 条样本
```

正式实验阶段：

```text
每种语言 300-500 条 factual pitfall 样本
总计 1500-2500 条样本
```

如果计算资源允许，可以进一步覆盖全部 16 种语言。

### 8.3 Train / Dev / Test 划分

虽然本研究不是传统监督训练，但仍建议划分：

| 子集 | 比例 | 用途 |
|---|---|---|
| train | 60% | 构造 intervention vector、选择提示模板 |
| dev | 20% | 调层数、选择干预强度、调阈值 |
| test | 20% | 最终报告结果 |

注意：

1. 同一个原始英文问题的不同语言版本不能同时出现在 train 和 test 中；
2. 如果做跨语言迁移实验，应按语言划分训练和测试；
3. 如果做跨数据源泛化实验，应按 source_dataset 划分训练和测试。

---

## 9. 实验任务设计

### 9.1 任务一：跨语言弱点评测

输入：

```text
prompt_en / prompt_target
```

输出：

```text
模型生成的答案
```

评价：

```text
Accuracy_en
Accuracy_target
Gap = Accuracy_en - Accuracy_target
```

目标：验证数据集是否确实体现跨语言事实回忆弱点。

### 9.2 任务二：错误归因分析

对目标语言答错样本进行中间层分析：

1. 使用 logit lens 观察每一层对正确英文答案的 rank；
2. 判断中间层是否曾经激活英文正确答案；
3. 判断最终是否成功转换到目标语言答案；
4. 给样本打上 recall_failure 或 conversion_failure 标签。

目标：解释跨语言错误发生在哪个阶段。

### 9.3 任务三：向量干预修复

实验条件：

| 条件 | 含义 |
|---|---|
| orig | 不干预 |
| diff | 使用 translation difference vector |
| task | 使用 recall task vector |
| diff_task | 同时使用两个向量 |

评价：

```text
Repair Rate = 错误变正确的样本数 / 原始错误样本数
Accuracy Gain = 干预后准确率 - 干预前准确率
```

目标：验证 Paths Not Taken 的机制干预能否修复 Cross-Lingual Pitfalls 中的事实回忆弱点。

### 9.4 任务四：跨语言迁移分析

设计：

```text
在 Chinese 上构造 intervention vector，在 Japanese / Korean 上测试
在 French 上构造 intervention vector，在 Spanish 上测试
```

目标：分析语言相似性是否影响干预迁移效果。

---

## 10. 评价指标设计

### 10.1 基础准确率指标

| 指标 | 说明 |
|---|---|
| Accuracy_en | 英文 prompt 正确率 |
| Accuracy_target | 目标语言 prompt 正确率 |
| Accuracy_gap | 英文与目标语言正确率差值 |
| Exact Match | 精确匹配答案 |
| Semantic Match | 语义等价匹配 |

### 10.2 弱点强度指标

```text
pitfall_score = rate_ori - rate_trans
```

也可以定义模型级别的 pitfall score：

```text
model_pitfall_score = correct_en - correct_target
```

其中：

```text
correct_en = 1 且 correct_target = 0 时，model_pitfall_score = 1
```

### 10.3 错误归因指标

| 指标 | 说明 |
|---|---|
| Recall Failure Ratio | recall_failure 占全部错误的比例 |
| Conversion Failure Ratio | conversion_failure 占全部错误的比例 |
| Entity Mapping Failure Ratio | 实体映射失败比例 |
| Non-factual Ratio | 非事实回忆样本比例 |

### 10.4 干预修复指标

| 指标 | 说明 |
|---|---|
| Repair Rate | 原本错误、干预后正确的比例 |
| Regression Rate | 原本正确、干预后错误的比例 |
| Net Gain | Repair Rate - Regression Rate |
| Accuracy Gain | 干预后准确率提升 |

---

## 11. 推荐实验模型

考虑 4 卡左右的计算资源，建议优先使用中小开源模型。

| 模型 | 角色 |
|---|---|
| Qwen2.5-3B | 中文友好主模型 |
| Qwen2.5-7B | 中文能力较强的扩展模型 |
| Llama-3.2-3B | 轻量对照模型 |
| Llama-3.1-8B | 主流英文能力模型 |
| Gemma-2-9B | 额外开源对照 |

建议不要一开始使用 70B 级别模型做机制分析，因为中间层提取和干预成本较高。

---

## 12. 数据质量控制

### 12.1 自动过滤

过滤条件建议：

```text
rate_ori >= 0.8
rate_trans <= 0.5
pitfall_score >= 0.3
is_factual_recall = true
answer_en 不为空
answer_target 不为空
prompt_en 不为空
prompt_target 不为空
```

### 12.2 语义等价检查

需要检查：

1. 英文问题和目标语言问题是否语义一致；
2. 英文答案和目标语言答案是否等价；
3. 选项翻译是否引入歧义；
4. prompt 转换后是否仍然自然；
5. 答案是否唯一。

可以使用：

```text
多翻译模型一致性检查
回译检查
LLM-as-a-Judge
人工抽检
```

### 12.3 人工抽检比例

建议：

```text
每种语言至少抽检 50-100 条
总体抽检比例不少于 10%
关键测试集抽检比例不少于 20%
```

---

## 13. 推荐目录结构

```text
cross_lingual_factual_pitfalls/
├── data_raw/
│   ├── cross_lingual_pitfalls/
│   │   ├── Chinese.json
│   │   ├── Japanese.json
│   │   └── ...
│   └── paths_not_taken/
│       └── factual_recall_data.json
├── data_processed/
│   ├── clf_pitfalls_zh.jsonl
│   ├── clf_pitfalls_ja.jsonl
│   ├── clf_pitfalls_ko.jsonl
│   ├── clf_pitfalls_fr.jsonl
│   └── clf_pitfalls_es.jsonl
├── data_split/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── scripts/
│   ├── filter_factual_samples.py
│   ├── convert_to_recall_prompt.py
│   ├── validate_translation_equivalence.py
│   ├── run_baseline_eval.py
│   ├── run_logit_lens.py
│   ├── run_intervention.py
│   └── analyze_results.py
├── configs/
│   ├── languages.yaml
│   ├── models.yaml
│   └── intervention.yaml
├── results/
│   ├── baseline/
│   ├── attribution/
│   └── intervention/
└── README.md
```

---

## 14. 最小可行版本 MVP

如果先做一个可以快速跑通的版本，建议如下：

```text
语言：Chinese、Japanese、French
模型：Qwen2.5-3B、Llama-3.2-3B
数据：每种语言 100 条，共 300 条
任务：
1. 筛选 factual recall 样本
2. 转换 prompt
3. 跑英文和目标语言准确率
4. 对错误样本做 logit lens
5. 初步区分 recall_failure 和 conversion_failure
```

MVP 输出：

```text
1. 一个 jsonl 格式的数据集
2. 一张语言维度准确率表
3. 一张错误归因比例表
4. 一张中间层答案 rank 曲线
5. 一份失败样例分析
```

---

## 15. 预期研究贡献

本数据集设计可以形成以下贡献：

1. **构建 Cross-Lingual Factual Pitfalls 子集**  
   从已有跨语言弱点样本中筛选事实回忆型样本，形成更适合机制分析的数据集。

2. **提出跨语言事实回忆错误归因框架**  
   将错误分为 recall_failure、conversion_failure、entity_mapping_failure 等类型。

3. **验证中间层向量干预的泛化能力**  
   检验 Paths Not Taken 的 intervention 方法是否能修复自动挖掘出的困难样本。

4. **分析语言相似性与错误类型分布**  
   比较中文、日文、韩文、法文、西文等语言在错误机制上的差异。

5. **形成可复用评测流程**  
   支持后续扩展到更多语言、更多模型和更多任务类型。

---

## 16. 一句话总结

本方案将 Cross-Lingual Pitfalls 作为跨语言弱点样本来源，将 Paths Not Taken 作为事实回忆路径分析与修复工具，最终构建一个面向多语言大模型的“跨语言事实回忆弱点数据集”，用于完成：

```text
自动发现 → 事实化转换 → 错误归因 → 向量干预修复 → 跨语言迁移分析
```
