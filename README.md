# DPO LLaMA Summarizer

Direct Preference Optimization (DPO) pipeline for training LLaMA models to generate concise abstractive summaries.

---

## Installation

Requirements: Python ≥3.10, CUDA-enabled PyTorch.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

# train
```bash
./scripts/run.sh
```
# inference
```bash
python -m src.cli.infer --model-checkpoint models/example_ckpt --input examples/sample.txt
```
