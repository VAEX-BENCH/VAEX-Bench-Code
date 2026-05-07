# VAEX-Bench-Code

## Dataset Loading

VAEX-Bench can be loaded using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("VAEX-BENCH/VAEX-Bench")
print(dataset)
