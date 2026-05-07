# VAEX-Bench-Code

This repository is prepared for the NeurIPS 2026 Evaluations and Datasets Track code submission. All dataset names, paths, metadata, and code components have been anonymized for the double-blind review process.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/VAEX-BENCH/VAEX-Bench.git
cd VAEX-Bench
pip install -r requirements.txt
```

## Dataset Loading

VAEX-Bench can be loaded using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("VAEX-BENCH/VAEX-Bench")
print(dataset)
```
## Evaluation

This code is provided to reproduce the main evaluation results reported in Table 2 of the paper. We provide two evaluation scripts depending on how the target model is accessed.

### API-based models

Use `api_infer.sh` for models accessed through external APIs, including GPT, Gemini, Claude, and OpenRouter-hosted models such as Mistral, Gemma, and GLM.

```bash
bash api_infer.sh
```

This script runs inference on VAEX-Bench QA pairs and saves the model responses for API-based models.

### Locally served video models

Use `video_infer.sh` for open-source video MLLMs that are served locally, including Qwen and InternVL models.

```bash
bash video_infer.sh
```

This script runs video-based inference on the VAEX-Bench videos and saves the model responses for locally served models.

### Outputs

The inference results are saved in the output directory specified in each script. Each output file contains the model answer for each video-question pair, along with the corresponding task type and reasoning regime (`extractive` or `abstractive`). These outputs can be used to compute the task-wise and average accuracies reported in the main results table.
