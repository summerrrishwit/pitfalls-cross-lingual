# 从 MMLU 等数据中抽取原子事实三元组并自动搜索跨语言 Factual Badcase：实验设计

> 版本：v1.0  
> 日期：2026-08-05  
> 研究范围：仅覆盖前半部分——从原始选择题数据构造 factual triple、生成 distractor_answer，并自动搜索跨语言 factual badcase。  
> 暂不包含 Paths Not Taken 的 Logit Lens、向量干预和机制归因实验。

---

## 1. 研究目标

本实验旨在验证：

1. 能否从 MMLU、ARC-Challenge、SciQ、CommonsenseQA、TruthfulQA 等英文选择题数据集中，自动筛选出可以表示为原子事实三元组的问题；
2. 能否将这些问题统一转化为：

```text
(subject, relation, answer, prompt_en)
```

3. 能否利用原始错误选项或同 relation 的负样本构造：

```text
distractor_answer
```

4. 能否借鉴 Cross-Lingual Pitfalls 的“错误答案驱动扰动 + 多模型模拟评分 + 搜索筛选”方法，生成：

```text
英文事实补全正确
目标语言事实补全错误
```

的跨语言 factual badcase；
5. 自动生成的 badcase 是否能够泛化到未参与搜索的模型。

---

## 2. 核心研究问题

### RQ1：哪些原始选择题可以转化为 factual triple？

需要判断原始问题是否满足：

```text
一个明确 subject
+
一个明确 relation
+
一个唯一 answer
```

而不是复杂推理、计算、多条件判断或长文本理解。

### RQ2：原始错误选项能否直接作为 distractor_answer？

需要分析原始错误选项是否：

- 与正确答案类型相同；
- 确实是错误事实；
- 不是正确答案的别名、同义词或上位概念；
- 适合用于生成定向扰动。

### RQ3：同 relation 负样本是否优于原始错误选项？

例如：

```text
subject = Japan
relation = country_currency
answer = Yen
```

可从其他 `country_currency` 样本中选择：

```text
Dollar / Euro / Won / Yuan
```

作为 distractor_answer。

需要比较：

- 原始错误选项；
- 同 relation 随机负样本；
- 同 relation hard negative；
- 随机跨类型负样本。

### RQ4：自动搜索是否能稳定得到 factual cross-lingual badcase？

目标是筛选：

```text
English original correct
English perturbed correct
Target original correct/incorrect
Target perturbed incorrect
```

并区分：

- 扰动诱发型 badcase；
- 扰动放大型 badcase。

---

## 3. 总体流程

```text
MMLU / ARC / SciQ / CommonsenseQA / TruthfulQA
                        │
                        ▼
              原始问题清洗与标准化
                        │
                        ▼
              原子事实问题识别
                        │
                        ▼
        (subject, relation, answer) 抽取
                        │
                        ▼
              relation 归一化与统计
                        │
                        ▼
         distractor_answer 候选构造
                        │
                        ▼
      英文与目标语言 factual prompt 生成
                        │
                        ▼
       原始 prompt 的英文事实召回校验
                        │
                        ▼
       错误答案驱动的 factual 扰动生成
                        │
                        ▼
       英文/目标语言语义与事实一致性检查
                        │
                        ▼
         多个 simulation models 评分
                        │
                        ▼
               Beam Search / Top-k
                        │
                        ▼
             factual cross-lingual badcases
                        │
                        ▼
        non-simulation models 泛化评测
```

---

## 4. 原始数据来源

第一阶段建议使用：

| 数据集 | 主要特点 | 预期可转换性 |
|---|---|---|
| MMLU | 多学科选择题 | 中等；部分为原子事实，部分为推理题 |
| ARC-Challenge | 科学问题 | 中等；科学事实较多，但也包含推理 |
| SciQ | 科学知识问答 | 较高；大量定义、类别、属性事实 |
| CommonsenseQA | 常识推理 | 较低到中等；部分问题难以转为单一 relation |
| TruthfulQA | 事实真实性 | 较低；答案可能较长、问题结构复杂 |

建议优先顺序：

```text
SciQ
→ MMLU 中事实密集子集
→ ARC-Challenge
→ CommonsenseQA
→ TruthfulQA
```

---

## 5. 原始数据标准化

所有数据先转成统一结构：

```json
{
  "source_id": "mmlu_anatomy_000001",
  "source_dataset": "mmlu",
  "source_subset": "anatomy",
  "question": "Which organ produces insulin?",
  "choices": [
    "Liver",
    "Pancreas",
    "Kidney",
    "Spleen"
  ],
  "answer_index": 1,
  "answer_text": "Pancreas"
}
```

### 5.1 基础清洗

需执行：

- 去除空题干；
- 去除空选项；
- 去除重复选项；
- 统一答案索引；
- 统一标点和空格；
- 去除明显格式错误；
- 去除答案不在 choices 中的样本；
- 保留原始 dataset/subset/id 以支持追溯。

### 5.2 去重

建议按以下层级去重：

1. 完全相同题干；
2. 题干规范化后相同；
3. 同一 `(subject, relation, answer)`；
4. 语义近重复问题。

---

## 6. 原子事实问题判定

### 6.1 原子事实定义

一个问题可转为 factual triple，当且仅当其核心可表示为：

```text
subject --relation--> answer
```

例如：

```text
Japan --country_currency--> Yen
```

对应 prompt：

```text
The official currency of Japan is
```

### 6.2 纳入标准

问题必须同时满足：

1. 存在单一明确 subject；
2. 存在单一明确 relation；
3. answer 唯一；
4. 不依赖多步推理；
5. 不依赖计算；
6. 不依赖选项之间的比较才能确定答案；
7. 去掉选项后仍能自然构造开放式事实补全 prompt；
8. answer 类型稳定；
9. 事实在实验时间范围内相对稳定。

### 6.3 排除标准

排除：

- 否定题，如 “Which of the following is NOT...”；
- 多个条件组合题；
- 数学计算题；
- 因果推理题；
- 情景推理题；
- 长段落阅读理解；
- 多答案问题；
- 答案是长句；
- 答案依赖当前时间、政策或动态事件；
- subject 或 relation 无法唯一识别；
- 选项仅是推理路径而非事实实体。

### 6.4 自动判定标签

建议为每条问题输出：

```json
{
  "is_atomic_fact": true,
  "atomicity_confidence": 0.94,
  "rejection_reason": null
}
```

若不可转换：

```json
{
  "is_atomic_fact": false,
  "atomicity_confidence": 0.88,
  "rejection_reason": "multi_step_reasoning"
}
```

### 6.5 建议判定方式

采用“LLM 初筛 + 规则校验 + 人工抽检”：

```text
LLM 判断是否为原子事实
        │
        ▼
规则验证 subject/relation/answer 是否完整
        │
        ▼
反向生成 prompt 并检查唯一答案
        │
        ▼
人工抽检
```

---

## 7. Factual Triple 抽取

### 7.1 标准字段

```json
{
  "subject": "Japan",
  "relation": "country_currency",
  "answer": "Yen",
  "prompt_en": "The official currency of Japan is"
}
```

### 7.2 扩展字段

建议实际保存：

```json
{
  "triple_id": "mmlu_world_000001",
  "source_dataset": "mmlu",
  "source_subset": "world_history",
  "source_question": "...",
  "source_choices": ["...", "...", "...", "..."],
  "source_answer": "...",

  "subject": "...",
  "relation_raw": "...",
  "relation_normalized": "...",
  "answer_en": "...",
  "answer_type": "...",

  "prompt_en": "...",
  "prompt_template_id": "...",

  "atomicity_confidence": 0.95,
  "triple_confidence": 0.93,
  "fact_verified": true
}
```

### 7.3 relation 抽取原则

relation 不能是任意自然语言长句，应标准化为稳定标签：

```text
country_currency
country_language
country_religion
person_university
animal_classification
object_color
inventor_of
capital_of
located_in
part_of
function_of
process_location
```

---

## 8. Relation Taxonomy 设计

### 8.1 第一阶段：封闭 relation 集

为保证实验可控，建议先采用有限 taxonomy。

优先使用 Paths Not Taken 已有的 relation：

```text
person_university
country_currency
book_language
animal_classification
country_language
country_religion
language_family
musician_country
musician_instruments
object_color
```

再根据 MMLU/SciQ 实际分布增加：

```text
capital_of
inventor_of
discoverer_of
scientific_unit
biological_function
process_location
part_of
chemical_property
disease_symptom
historical_event_date
```

### 8.2 relation 映射

原始抽取：

```text
"official money used by a country"
```

归一化：

```text
country_currency
```

需保存：

```json
{
  "relation_raw": "official money used by a country",
  "relation_normalized": "country_currency"
}
```

### 8.3 relation 分布统计

至少报告：

| 指标 | 说明 |
|---|---|
| 原始问题数量 | 各数据集总题数 |
| 原子事实数量 | 可转换问题数 |
| Triple Conversion Rate | 可转换比例 |
| Relation Count | 每个 relation 的数量 |
| Relation Coverage | relation 类别数 |
| Dataset × Relation | 各数据集在 relation 上的分布 |
| Answer Type Distribution | 国家、人物、机构、类别等分布 |

### 8.4 长尾处理

对样本过少的 relation：

```text
count < 30
```

可选择：

- 合并到上位 relation；
- 暂不进入主实验；
- 仅用于泛化测试；
- 后续扩充数据。

---

## 9. Prompt 构造

### 9.1 目标格式

从选择题：

```text
Which country uses the yen?
A. China
B. Japan
C. Korea
D. Thailand
```

转为 factual completion：

```text
The country whose official currency is the yen is
```

或：

```text
The official currency of Japan is
```

优先选择满足以下要求的 prompt：

- subject 明确；
- relation 明确；
- answer 是自然的下一个 token/短语；
- 不暴露选项；
- 不要求解释；
- 不带问号时也能形成补全任务；
- 各语言结构尽量平行。

### 9.2 Prompt 模板

```json
{
  "relation": "country_currency",
  "prompt_template_en": "The official currency of {} is",
  "prompt_template_zh": "{}的官方货币是"
}
```

### 9.3 多模板控制

每个 relation 建议保留：

- 1 个主模板；
- 1–2 个 paraphrase 模板。

主实验使用固定模板，消融实验测试模板鲁棒性。

### 9.4 Answer Alias

为每个答案建立别名：

```json
{
  "answer_en": "United States dollar",
  "aliases_en": [
    "US dollar",
    "U.S. dollar",
    "Dollar"
  ]
}
```

目标语言同样建立 alias，以防准确率判定过于严格。

---

## 10. Distractor Answer 构造

### 10.1 结构

最终每条 triple 扩展为：

```text
(subject, relation, answer, distractor_answer)
```

### 10.2 来源一：原始错误选项

原始选择题中的错误选项优先保留。

示例：

```text
answer = Pancreas
wrong options = Liver, Kidney, Spleen
```

可构造：

```text
distractor_answer ∈ {Liver, Kidney, Spleen}
```

### 10.3 来源二：同 relation 负采样

从同 relation 的其他样本 answer 池中采样：

```text
country_currency:
Yen / Dollar / Euro / Won / Yuan
```

优势：

- 类型一致；
- 规模大；
- 可控制难度。

### 10.4 来源三：Hard Negative

优先选择：

- 语义相近；
- 同一地区；
- 同一类别；
- 容易与 subject 混淆；
- 模型已有较高先验概率。

例如：

```text
subject = Japan
answer = Yen
hard distractor = Won
```

通常比：

```text
hard distractor = Peso
```

更难。

### 10.5 来源四：随机跨类型负样本

例如：

```text
country_currency 的 distractor = Buddhism
```

仅作为对照，不应进入正式 badcase 数据集。

### 10.6 质量过滤

每个 distractor 必须满足：

```text
Type(distractor) = Type(answer)
distractor != answer
distractor 不是 answer 的别名
(subject, relation, distractor) 为假
```

### 10.7 Distractor 难度分级

建议定义：

```text
easy
medium
hard
```

可能指标：

- 文本语义相似度；
- 同一地理区域；
- 同一知识类别；
- 原模型在原始 prompt 上对 distractor 的概率；
- distractor 与 answer 的 embedding 距离。

### 10.8 推荐字段

```json
{
  "distractor_answer_en": "Won",
  "distractor_source": "same_relation_hard_negative",
  "distractor_type": "currency",
  "distractor_difficulty": "hard",
  "distractor_verified": true
}
```

---

## 11. 原始 Prompt 基线校验

### 11.1 必须使用开放式 factual prompt

不能只依据原选择题英文正确。

需要重新验证：

```text
model(prompt_en_original) = answer
```

### 11.2 四状态记录

对每条样本保存：

| 状态 | 内容 |
|---|---|
| EN-MCQ | 原英文选择题 |
| EN-Open | 英文 factual completion |
| T-MCQ | 目标语言选择题 |
| T-Open | 目标语言 factual completion |

主实验只保留：

```text
EN-Open original correct
```

的样本。

### 11.3 为什么必须重新校验

选择题答对可能依赖：

- 排除法；
- 选项位置；
- 选项对比；
- 猜测；
- 浅层关键词。

开放式事实补全正确，才能证明模型确实具备该英文事实知识。

---

## 12. 多语言数据构造

### 12.1 目标语言

MVP 建议：

```text
English + Chinese
```

完整实验可扩展：

```text
Chinese
Japanese
Korean
French
Spanish
German
```

### 12.2 翻译对象

需翻译：

- subject；
- prompt；
- answer；
- distractor_answer；
- perturbation。

### 12.3 翻译原则

- 保持 relation 不变；
- 保持正确答案不变；
- 保持 distractor 类型不变；
- 不增加解释；
- 不改变 factual completion 形式；
- 确保答案可自然接在 prompt 后。

### 12.4 多语言字段

```json
{
  "subject_en": "Japan",
  "subject_zh": "日本",

  "answer_en": "Yen",
  "answer_zh": "日元",

  "distractor_answer_en": "Won",
  "distractor_answer_zh": "韩元",

  "prompt_en_original": "The official currency of Japan is",
  "prompt_zh_original": "日本的官方货币是"
}
```

---

## 13. 扰动生成

### 13.1 输入

扰动生成模型接收：

```text
subject
relation
correct answer
distractor answer
original prompt
```

### 13.2 生成目标

生成一段：

- 与 distractor_answer 相关；
- 上下文合理；
- 不显式陈述错误事实；
- 不改变正确答案；
- 不修改 subject-relation-answer；
- 能增加错误答案竞争性的干扰上下文。

### 13.3 推荐生成提示

```text
You are given a factual completion prompt and a distractor answer.

Generate a short contextual distraction that is semantically associated
with the distractor answer, but does not explicitly state any false fact,
does not contradict the correct answer, and does not change the factual
relation being queried.

The original factual completion must remain answerable with the same
correct answer.

Subject: {subject}
Relation: {relation}
Correct answer: {answer}
Distractor answer: {distractor_answer}
Original prompt: {prompt_en}

Output only the generated distraction.
```

### 13.4 拼接方式

对 factual completion，统一使用前置拼接：

```text
perturbation + prompt_original
```

不要将扰动放在答案预测位置之后。

示例：

```text
Japan is located near countries that use several other East Asian currencies.
The official currency of Japan is
```

### 13.5 多轮扰动

每轮在上一轮 prompt 基础上继续增加新 distraction：

```text
P0 = original prompt
P1 = δ1 + P0
P2 = δ2 + P1
...
```

需保存：

- parent_id；
- perturbation_round；
- accumulated_perturbations；
- search_depth。

---

## 14. 语义与事实一致性检查

每个候选必须通过以下检查。

### 14.1 Ground-truth Preservation

确认：

```text
扰动前答案 = answer
扰动后答案仍 = answer
```

### 14.2 No Explicit Falsehood

扰动不能直接写：

```text
Japan uses the won.
```

### 14.3 Relation Preservation

扰动后仍在查询原 relation：

```text
country_currency
```

而不是转成邻近关系。

### 14.4 Bilingual Equivalence

英文与目标语言扰动必须表达同一内容。

### 14.5 Naturalness

扰动应当是自然上下文，而不是拼接式乱码或不连贯文本。

### 14.6 推荐检查输出

```json
{
  "ground_truth_preserved": true,
  "explicit_falsehood": false,
  "relation_preserved": true,
  "bilingual_equivalent": true,
  "naturalness_score": 4.5,
  "check_passed": true
}
```

---

## 15. Simulation Models

### 15.1 角色

Simulation models 用于：

- 回答英文扰动 prompt；
- 回答目标语言扰动 prompt；
- 计算平均准确率；
- 为搜索提供排序分数。

### 15.2 模型集合

建议使用 3–5 个模型：

```text
M = {M1, M2, ..., MK}
```

要求：

- 多语言能力不同；
- 架构和训练来源有差异；
- 至少包含一个后续白盒分析模型；
- 推理成本可控。

### 15.3 单样本统计

对每个模型记录：

```json
{
  "model": "model_name",
  "pred_en": "...",
  "pred_target": "...",
  "correct_en": true,
  "correct_target": false,
  "distractor_hit_en": false,
  "distractor_hit_target": true
}
```

### 15.4 平均准确率

```text
acc_en = simulation models 的英文平均准确率
acc_target = simulation models 的目标语言平均准确率
```

---

## 16. 搜索目标函数

### 16.1 论文形式

```text
score = (acc_en ^ gamma) - acc_target
```

其中：

```text
gamma > 1
```

强调英文准确率必须高。

### 16.2 仓库实现形式

Cross-Lingual Pitfalls 当前 GitHub `run.py` 使用 reward-minus-penalty 形式，而非论文中的简单差值形式。

本实验应显式区分：

```text
score_type = paper
score_type = repo
```

### 16.3 推荐主实验

主实验使用论文形式：

```text
score = (acc_en ^ 2) - acc_target
```

并设置硬约束：

```text
acc_en = 1.0
```

### 16.4 推荐补充指标

增加 distractor attraction：

```text
distractor_hit_rate
distractor_probability_shift
distractor_rank_shift
```

防止“目标语言答错”但错误与指定 distractor 完全无关。

---

## 17. Beam Search

### 17.1 搜索状态

每个节点包括：

```text
base triple
distractor_answer
当前累计 perturbation
英文 prompt
目标语言 prompt
simulation score
```

### 17.2 搜索步骤

```text
初始 prompt
    │
    ├─ 对每个 distractor 生成多个 perturbation
    │
    ├─ 语义检查
    │
    ├─ simulation models 评分
    │
    ├─ 保留 top-k
    │
    └─ 继续下一轮扰动
```

### 17.3 推荐参数

MVP：

```text
beam_width = 6
max_depth = 3
num_distractors = 3
num_generations_per_distractor = 2
max_good_per_triple = 2
```

完整实验可增加：

```text
beam_width = 12
max_depth = 4–6
```

### 17.4 提前停止

若满足：

```text
acc_en = 1.0
acc_target <= 0.2
```

则直接加入 strong badcase 集合。

若：

```text
acc_en < 1.0
```

则停止扩展该节点。

---

## 18. Badcase 定义

### 18.1 基础定义

```text
English perturbed correct
Target perturbed incorrect
```

### 18.2 诱发型 Badcase

```text
EN original = correct
EN perturbed = correct
Target original = correct
Target perturbed = incorrect
```

这是主实验最重要的 badcase。

### 18.3 放大型 Badcase

```text
EN original = correct
EN perturbed = correct
Target original = incorrect
Target perturbed = incorrect
```

需要进一步满足：

```text
正确答案概率下降
或
distractor 概率上升
```

### 18.4 泛化型 Badcase

在 non-simulation models 上也满足：

```text
English correct
Target incorrect
```

---

## 19. 数据输出结构

推荐最终 JSONL：

```json
{
  "sample_id": "mmlu_country_currency_0001_d2_r1",

  "source_dataset": "mmlu",
  "source_subset": "world_history",
  "source_question": "...",
  "source_choices": ["Yen", "Won", "Dollar", "Yuan"],

  "subject": "Japan",
  "relation": "country_currency",

  "answer_en": "Yen",
  "answer_target": "日元",

  "distractor_answer_en": "Won",
  "distractor_answer_target": "韩元",
  "distractor_source": "original_option",
  "distractor_difficulty": "hard",

  "prompt_en_original": "The official currency of Japan is",
  "prompt_target_original": "日本的官方货币是",

  "perturbation_en": "...",
  "perturbation_target": "...",

  "prompt_en_perturbed": "... The official currency of Japan is",
  "prompt_target_perturbed": "... 日本的官方货币是",

  "search_depth": 1,

  "simulation": {
    "acc_en_original": 1.0,
    "acc_target_original": 1.0,
    "acc_en_perturbed": 1.0,
    "acc_target_perturbed": 0.2,
    "score": 0.8
  },

  "badcase_type": "induced",
  "semantic_check_passed": true,
  "fact_check_passed": true
}
```

---

## 20. 数据划分

### 20.1 按 base triple 分组划分

同一个 base triple 的：

- 不同 distractor；
- 不同 perturbation；
- 不同语言；
- 不同搜索深度；

必须进入同一 split。

### 20.2 推荐划分

```text
train: 40%
validation: 10%
test: 50%
```

### 20.3 用途

- train：搜索策略调试、扰动提示设计；
- validation：选择阈值、beam 参数、score；
- test：最终生成效果与模型泛化评测。

### 20.4 泛化划分

额外建立：

#### Across-relation

测试集中放入未在 train 中出现的 relation。

#### Across-dataset

例如：

```text
train: MMLU + SciQ
test: ARC
```

#### Across-model

simulation models 与 non-simulation models 完全分离。

---

## 21. 对照组

### G0：Original

```text
原始 factual prompt
```

### G1：Neutral Context

加入长度相近、与 answer/distractor 无关的上下文。

### G2：Targeted Distractor

使用 distractor_answer 驱动的扰动。

### G3：Random Same-Type Distractor

使用同类型随机错误答案。

### G4：Cross-Type Distractor

使用不同类型错误答案，仅用于对照。

### G5：Explicit False Statement

直接加入错误事实，作为攻击上界，不进入正式数据集。

---

## 22. 主要评测指标

### 22.1 Triple 抽取质量

```text
Triple Conversion Rate
Subject Accuracy
Relation Accuracy
Answer Accuracy
Fact Verification Pass Rate
```

### 22.2 Relation 分布

```text
Relation Count
Relation Coverage
Dataset × Relation Distribution
Long-tail Ratio
```

### 22.3 Distractor 质量

```text
Type Match Rate
False-Fact Validity Rate
Alias Conflict Rate
Distractor Difficulty Distribution
```

### 22.4 扰动质量

```text
Ground-truth Preservation Rate
Bilingual Equivalence Rate
Naturalness Score
Explicit Falsehood Rate
```

### 22.5 Badcase 搜索效果

```text
English Retention Rate
Target Accuracy Drop
Induced Failure Rate
Amplified Failure Rate
Badcase Conversion Rate
Strong Badcase Rate
Average Generation Cost
Average Search Depth
```

### 22.6 Distractor 有效性

```text
Distractor Hit Rate
Distractor Rank Improvement
Distractor Probability Increase
```

### 22.7 泛化能力

```text
Non-simulation Transfer Rate
Across-relation Transfer
Across-dataset Transfer
Across-language Transfer
```

---

## 23. 核心消融实验

1. 原始错误选项 vs 同 relation 负样本；
2. 随机负样本 vs hard negative；
3. 无 distractor 指导 vs distractor 指导；
4. 单模型 simulation vs 多模型 simulation；
5. 论文 score vs repo score；
6. 无搜索 vs beam search；
7. 单轮扰动 vs 多轮扰动；
8. neutral context vs targeted perturbation；
9. 选择题评测 vs open factual completion；
10. simulation models vs non-simulation models；
11. 只看最终答错 vs 同时要求 distractor attraction；
12. 固定模板 vs paraphrase 模板。

---

## 24. 人工评估

### 24.1 抽样

每个主要 relation 抽取：

```text
50 条 triple
50 条 distractor
50 条生成 perturbation
```

### 24.2 标注维度

1. 是否为原子事实；
2. subject 是否正确；
3. relation 是否正确；
4. answer 是否正确；
5. distractor 是否为错误事实；
6. distractor 是否与 answer 同类型；
7. 扰动是否改变原事实；
8. 扰动是否显式引入错误事实；
9. 英文与目标语言是否等价；
10. prompt 是否仍为 factual completion。

### 24.3 一致性

报告：

```text
Cohen’s kappa
Percent Agreement
```

---

## 25. 最小可行实验（MVP）

### 25.1 数据

```text
数据集：SciQ + MMLU
目标语言：Chinese
relation：3–5 类
base triples：300
```

### 25.2 Distractor

```text
每条 triple：
1 个原始错误选项
1 个同 relation 随机负样本
1 个 hard negative
```

### 25.3 生成

```text
每个 distractor 生成 2 个 perturbation
初始候选数约：
300 × 3 × 2 = 1,800
```

### 25.4 搜索

```text
beam_width = 6
max_depth = 3
```

### 25.5 模型

```text
3 个 simulation models
2 个 non-simulation models
```

### 25.6 MVP 需要回答

1. 原始数据的 triple conversion rate；
2. relation 分布；
3. 原始错误选项的可用率；
4. 同 relation hard negative 是否更有效；
5. targeted perturbation 是否显著降低目标语言准确率；
6. 英文 retention 是否保持；
7. 目标错误是否命中 distractor；
8. badcase 是否能迁移到 non-simulation models。

---

## 26. 成功标准

建议预注册：

```text
Triple Conversion Rate >= 20%
Triple Extraction Accuracy >= 90%
Ground-truth Preservation >= 90%
Explicit Falsehood Rate <= 10%
English Retention >= 95%
Targeted Perturbation 的 Induced Failure Rate 显著高于 Neutral
Non-simulation Transfer Rate > 0
```

以上阈值为实验规划标准，不是已有论文结果。

---

## 27. 风险与解决方案

### 风险 1：MMLU 中可转换问题比例过低

解决：

- 优先使用 SciQ；
- 筛选 MMLU 事实密集 subset；
- 扩展 relation taxonomy；
- 不强行转换推理题。

### 风险 2：原始错误选项质量差

解决：

- 类型过滤；
- 事实验证；
- 使用同 relation 负采样；
- 构造 hard negative。

### 风险 3：开放式 prompt 上英文不再正确

解决：

- 重新进行 EN-Open baseline；
- 不以选择题英文正确替代事实召回正确；
- 仅保留 EN-Open 正确样本。

### 风险 4：扰动把任务变成阅读理解

解决：

- 限制扰动长度；
- 保持 factual completion 结构；
- 使用 neutral context 控制；
- 记录 prompt 长度。

### 风险 5：模型答错但与 distractor 无关

解决：

- 记录逐样本预测；
- 计算 distractor hit/rank/probability；
- 将第三类错误单独标记。

### 风险 6：搜索过拟合 simulation models

解决：

- 使用 non-simulation models；
- 更换 simulation model 组合；
- across-model 泛化评测；
- 报告 simulation 与 non-simulation 差异。

### 风险 7：翻译造成事实变化

解决：

- answer alias；
- 双语 semantic check；
- 人工抽检；
- 对低置信度翻译直接丢弃。

---

## 28. 推荐代码模块

```text
src/
├── load_source_datasets.py
├── normalize_mcq.py
├── classify_atomic_fact.py
├── extract_triples.py
├── normalize_relations.py
├── build_relation_pool.py
├── generate_distractors.py
├── verify_distractors.py
├── build_prompts.py
├── translate_prompts.py
├── validate_open_recall.py
├── generate_perturbations.py
├── semantic_check.py
├── simulate_models.py
├── beam_search.py
├── normalize_predictions.py
├── evaluate_badcases.py
└── analyze_relation_distribution.py
```

---

## 29. 推荐配置文件

```yaml
datasets:
  - mmlu
  - sciq
  - arc_challenge

languages:
  - Chinese

relations:
  closed_set: true
  min_count: 30

distractors:
  original_option: true
  same_relation_random: true
  same_relation_hard: true
  max_per_triple: 3

generation:
  perturbations_per_distractor: 2
  max_tokens: 128
  temperature: 0.7

search:
  beam_width: 6
  max_depth: 3
  max_good_per_triple: 2
  score_type: paper
  gamma: 2

filter:
  require_english_original_correct: true
  require_english_perturbed_correct: true
  strong_badcase_target_accuracy: 0.2
  candidate_target_accuracy: 0.4
```

---

## 30. 最终交付物

### 数据

```text
atomic_fact_candidates.jsonl
validated_triples.jsonl
relation_statistics.json
distractor_candidates.jsonl
validated_distractors.jsonl
factual_perturbation_candidates.jsonl
factual_cross_lingual_badcases.jsonl
```

### 报告

```text
triple_extraction_report.md
relation_distribution_report.md
distractor_quality_report.md
badcase_generation_report.md
generalization_evaluation_report.md
```

### 图表

```text
Dataset → Triple Conversion Rate
Relation Distribution
Distractor Source Distribution
English vs Target Accuracy
Original vs Perturbed Accuracy
Simulation vs Non-simulation Performance
Badcase Conversion Rate by Relation
Distractor Hit Rate by Relation
```

---

## 31. 实验结论边界

本阶段实验能够回答：

- 哪些选择题能转化为 factual triple；
- relation 分布是什么；
- 原始错误选项是否适合作为 distractor；
- 同 relation 负样本是否更有效；
- 是否能自动搜索跨语言 factual badcase；
- badcase 是否泛化到其他模型。

本阶段暂时不能回答：

- badcase 在模型内部属于事实召回失败还是语言转换失败；
- 哪一层或哪一类神经元导致错误；
- 向量干预能否修复错误。

这些问题应在后续 Paths Not Taken 机制分析阶段处理。

---

## 32. 核心定义总结

```text
原选择题 answer
    → factual triple 的 ground-truth answer

原选择题 wrong option
    → distractor_answer 候选

distractor_answer
    → 扰动生成方向，不是新的正确答案

生成 perturbation
    → 应增加 distractor 竞争性，但不改变原事实

最终 factual badcase
    → 英文开放式事实补全正确，目标语言开放式事实补全错误
```
