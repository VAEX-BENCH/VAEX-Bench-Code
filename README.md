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

The evaluation code reproduces the main results in Table 2 of the paper. We provide two inference scripts.

### API Inference

`api_infer.sh` is used for API-based models: GPT, Gemini, Claude, and OpenRouter models such as Mistral, Gemma, and GLM.

```bash
sh api_infer.sh
```

### Video Model Inference

`video_infer.sh` is used for locally served open-source video MLLMs, including Qwen and InternVL models. These models require GPU resources, and the required number and memory of GPUs may vary depending on the model size.

```bash
sh video_infer.sh
```

The scripts save model responses for each video-question pair. The saved outputs are used to compute extractive and abstractive task accuracies for reproducing Table 2.

### Outputs

The inference results are saved in the output directory specified in each script. Each output file contains the model answer for each video-question pair, along with the corresponding task type and reasoning regime (`extractive` or `abstractive`). These outputs can be used to compute the task-wise and average accuracies reported in the main results table.
