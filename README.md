<h1 align="center">Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models</h1>

🌐 **Project page:** https://xzx34.github.io/cross-lingual-pitfalls/

📄 **ACL Anthology:** https://aclanthology.org/2025.acl-long.404/

📄 **arXiv:** https://arxiv.org/abs/2505.18673

🤗 **Dataset (Hugging Face):** https://huggingface.co/datasets/xzx34/cross-lingual-pitfalls

## Updates & News
- [05/15/2025] 🥂 **Cross-Lingual Pitfalls has been accepted by ACL 2025! See you in Vienna!**

## Introduction

We introduce a systematic framework for studying **Cross-Lingual Weakness**—a critical challenge where LLMs fail to generalize their English proficiency to other languages. Our methodology enables:  

1. **Automated Weakness Identification**  
   Beam search-based perturbation strategy leveraging high-quality English datasets to systematically uncover cross-lingual weaknesses.  

2. **Quantitative Cross-Lingual Assessment**  
   A benchmarking framework measuring performance disparities across 16 languages, analyzing linguistic similarity effects.  

3. **Mitigation Strategy Evaluation**  
   Comparative analysis of fine-tuning effectiveness across languages, revealing how linguistic proximity influences adaptation.  

<p align="center">
<img width="85%" alt="CLP Pipeline" src="images/pipeline.jpg">    
</p>

## Dataset

The full benchmark — **6,713 bilingual (English ↔ target-language) pairs across 16 languages**, built from five English QA benchmarks (MMLU, ARC, CommonsenseQA, TruthfulQA, SciQ) — is available on the Hugging Face Hub and mirrored under [`data/`](data/):

🤗 **https://huggingface.co/datasets/xzx34/cross-lingual-pitfalls**

```python
from datasets import load_dataset
ds = load_dataset("xzx34/cross-lingual-pitfalls", "Chinese")
```

## Installation

```bash
conda create -n clp python=3.9
conda activate clp
pip install -r requirements.txt
```

## Configuration

Place the .env file in the utils folder. You can selectively add API keys for the models you intend to use.

```properties
HTTP_PROXY=your_http_proxy
HTTPS_PROXY=your_https_proxy

OPENAI_BASE_URL=https://ctapi.csxdtx.com:16000/v1
OPENAI_API_KEY=your_openai_api_key

DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
DEEPINFRA_API_KEY=your_deepinfra_api_key

YI_BASE_URL=https://api.lingyiwanwu.com/v1/
YI_API_KEY=your_yi_api_key

# Alibaba Cloud Model Studio / Bailian (OpenAI-compatible)
# You only need to fill one key variable. BAILIAN_API_KEY is preferred;
# DASHSCOPE_API_KEY is also supported for compatibility with DashScope docs.
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_TIMEOUT_SECONDS=120
BAILIAN_API_KEY=your_bailian_api_key
# DASHSCOPE_API_KEY=your_dashscope_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# SenseNova Anthropic SDK base URL; the SDK appends /v1/messages
SENSENOVA_BASE_URL=https://token.sensenova.cn
SENSENOVA_API_KEY=
```

## Usage

### Step 1: Generate cross-lingual weaknesses

```bash
python run.py 
  --input-file data/source/mmlu.json 
  --output-file data/mmlu_Chinese.json 
  --language Chinese 
  --max-good 3 
  --max-queue 12 
  --batch-size 4 
  --models gpt-4o llama-3.1-8B qwen-2.5-72B gpt-4o-mini gemma-2-27B   
```

### Step 2: Evaluate performance disparities
```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_results.json \
  --models qwen3.7-plus qwen3.7-max
```

`--models` is required. Only the models passed on the command line are evaluated
and checkpointed. Checkpoints are stored separately under `data/Chinese/`, for
example `Chinese_results_qwen3.7-plus_progress.json`.

To evaluate only Bailian Qwen 3.7 models:

```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_qwen37_results.json \
  --models qwen3.7-plus qwen3.7-max \
  --max-workers 2
```

To evaluate the additional OpenAI-compatible Qwen and DeepSeek models:

```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_qwen_deepseek_results.json \
  --models qwen3.5-plus qwen3-max deepseek-v3 deepseek-v4-pro deepseek-v4-flash \
  --max-workers 2
```

### Step 3: Analyze Results
```bash
python visualization.py 
  --languages Chinese Japanese Korean French Spanish Italian Ukrainian German Bengali Hindi Arabic Hebrew Amharic Yoruba Swahili Zulu 
  --input_folder data/ 
  --output_folder visualizations/
```

### Atomic factual-triple extraction

This workflow is intentionally separate from `run.py` and `eva.py`. It starts from
the five immutable English QA datasets under `data/source/`, creates a raw-source
manifest before any model request, extracts atomic `(subject, relation_raw, answer)`
triples independently with SenseNova and Qwen, and builds per-model relation inventories. It does not generate
factual prompts, distractors, translations, or cross-lingual scores in this phase.

Create and inspect the deterministic 100-item pilot without an API call:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  sample --run-id triple-pilot-v1 --dry-run
```

After inspecting
`data_processed/factual_triples/triple-pilot-v1/raw_source_manifest.json`, run a
dual-model extraction. Each LLM receives the question and canonical answer,
but not the answer choices. `source_choices` are copied directly from the manifest
into each output record.

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  extract \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --max-workers 1 \
  --model-parallelism 2
```

Outputs are isolated by model under `models/<model-name>/triple_extractions.jsonl`.
`qwen3.7-plus` uses `OPENAI_BASE_URL=https://ctapi.csxdtx.com:16000/v1`.

Resume a checkpoint without repeating completed records:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  extract \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --resume \
  --retry-failed \
  --max-workers 2
```

Build the global relation inventory from locally valid extracted triples:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  inventory \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --model qwen3.7-plus
```

Evaluate both model labels against the frozen 100-row Codex reference. The 300-row
experiment is eligible only when every required model has at most 3% label error:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py evaluate \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --reference data_processed/factual_triples/triple-pilot-v1/calibration/codex_reference_labels_v1.jsonl \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --input-name triple_extractions_v6.jsonl \
  --threshold 0.03
```

Create a disjoint balanced 300-row manifest, then run the same two model labelers:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_300.json sample \
  --run-id triple-300-v1 \
  --output-dir data_processed/factual_triples/triple-300-v1 \
  --exclude-manifest data_processed/factual_triples/triple-pilot-v1/raw_source_manifest.json
```

Build the final agreement/disagreement bundle for Codex review:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_300.json summarize \
  --run-dir data_processed/factual_triples/triple-300-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus
```

After Codex or `qwen3.7-plus` proposes and a reviewer freezes the taxonomy and
explicit mapping, apply them deterministically:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  normalize \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --model qwen3.7-plus \
  --taxonomy configs/relation_taxonomy_v1.json \
  --mapping configs/relation_mapping_v1.json
```

Resume is rejected when the checkpoint uses another extraction prompt version or
model. `--resume --retry-failed` retries only transport or validation failures while
preserving retry history. `source_answer` remains exact provenance; the extracted
`answer` may be a concise substring grounded in a sentence-like source answer. Before
freezing triples, inspect atomicity, subject selection, relation direction, answer
grounding, fact validity, and the Codex disagreement adjudication.

## Citation

If you use this code or dataset, please cite the ACL 2025 version:

```bibtex
@inproceedings{xu-etal-2025-cross,
    title = "Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models",
    author = "Xu, Zixiang  and Wang, Yanbo  and Huang, Yue  and Chen, Xiuying  and Zhao, Jieyu  and Jiang, Meng  and Zhang, Xiangliang",
    editor = "Che, Wanxiang  and Nabende, Joyce  and Shutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul, year = "2025", address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.404/",
    doi = "10.18653/v1/2025.acl-long.404",
    pages = "8254--8284", ISBN = "979-8-89176-251-0"
}
```

The arXiv preprint can also be cited:

```bibtex
@misc{xu2025crosslingualpitfallsautomaticprobing,
      title={Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models}, 
      author={Zixiang Xu and Yanbo Wang and Yue Huang and Xiuying Chen and Jieyu Zhao and Meng Jiang and Xiangliang Zhang},
      year={2025},
      eprint={2505.18673},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18673}, 
}
```
