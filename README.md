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
ANSWER_EXTRACT_MODEL=qwen3.7-plus

ANTHROPIC_API_KEY=your_anthropic_api_key
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
