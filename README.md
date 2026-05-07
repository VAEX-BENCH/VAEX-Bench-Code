# VAEX-Bench-Code

This repository is prepared for the NeurIPS Evaluations and Datasets Track code submission. All dataset names, paths, metadata, and code components have been anonymized for the double-blind review process.

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
