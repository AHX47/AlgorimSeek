# AlgorimSeek
Algorim languages fine-tuning  training  deepseek thinking-coder model  at Datastructures and algorithms  languages   using  VLM  or LLM  differnts type 
## Project Overview

This repository contains a complete fine-tuning pipeline for **AlgorimSeek** – a family of specialized language models for the **Algorim (algo a47)** pseudocode language. The models are trained to understand, generate, debug, and visualize algorithms and data structures, supporting both text-only and vision+text inputs.

The dataset includes:
- **1000+ code examples** covering fundamental algorithms (sorting, searching, recursion, dynamic programming, graphs, etc.)
- **200+ code‑to‑image pairs** for visual understanding
- **Execution traces, compile/debug commands, and bilingual (Arabic/English) prompts**

Total dataset size: **~5.3M tokens** (text + vision annotations).

---

# 1. LLM Chat (Text‑Only) Fine‑Tuning  
📄 [`README_LLM.md`](./README_LLM.md)

**Model:** DeepSeek Coder / Qwen2.5‑Coder (7B)  
**Task:** Text → Algorim code, code explanation, algorithmic reasoning  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AHX47/AlgorimSeek/blob/main/AlgorimSeek_FineTuning.ipynb)


...

### Key Features
- Generates Algorim actions from natural language descriptions.
- Explains time/space complexity.
- Supports `/compile`, `/debug`, `/execute` commands in a chat interface.

### Dataset
The datasets included in the project are:

- **algorim_dataset.csv** – metadata table with 1000+ entries (id, category, difficulty, topic, question, description, etc.)
- **algorim_dataset.json** – full structured dataset (2.3 MB) containing code, image paths, execution traces, thinking steps, and compiled output.
- **algorim_training.json** – instruction‑tuning dataset (1.5 MB) for text‑only code generation and explanation.
- **algorim_vision_training.json** – vision‑language dataset (1.2 MB) pairing code images with prompts and responses.
- **algorim_execution_traces.json** – step‑by‑step execution traces for debugging and /compile commands.
- **algorim_extra.json** – additional command examples (e.g., `/imagine`, `/debug`).
- **Download zip**
- [.Download Datasets.](https://github.com/AHX47/AlgorimSeek/releases/tag/1.0)
- [.Hugging Face](https://huggingface.co/Abdohak47)

Additionally, the following directories contain raw data:
- **1000_code_dataset/** – 1000 original Algorim source files.
- **200_image_dataset/** – 200 rendered PNG images of code snippets.

### Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AHX47/AlgorimSeek/blob/main/AlgorimSeek_Training.ipynb)
```bash
python train_llm.py \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --data ./datasets/algorim_training.json \
  --lora_r 64 --lora_alpha 128 \
  --batch_size 2 --grad_accum 8 \
  --output_dir ./algorim_llm
```

### Inference Example
```python
response = model.generate(
    "Write an Algorim action to reverse a singly linked list."
)
```

---

# 2. Vision + Text Model (Code Understanding from Images)  
📄 [`README_VISION.md`](./README_VISION.md)

**Model:** Qwen2‑VL‑7B / Phi‑3.5‑Vision  
**Task:** Read code from screenshots, explain algorithm, generate equivalent Algorim code  

### Key Features
- Extracts code from handwritten or printed algorithm images.
- Can answer questions like “What does this code do?” or “Translate this Python snippet to Algorim”.

### Dataset
- `algorim_vision_training.json` – 200 image‑text pairs (code images + prompts + responses).
- `200_image_dataset/` – corresponding rendered code images.

### Training (LoRA)
```python
# See AlgorimSeek_Training.ipynb
CONFIG = {
    'BASE_MODEL': 'Qwen/Qwen2-VL-7B-Instruct',
    'VISION_DATASET': './datasets/algorim_vision_training.json',
    'LORA_R': 64,
    'USE_4BIT': True,
}
```

### Inference
```python
answer = ask_algorimseek(
    "Read this code image and explain the sorting algorithm",
    image_path="demo_render.png"
)
```

---

# 3. Vision + Text + Execution (Full Agent)  
📄 [`README_VISION_EXECUTION.md`](./README_VISION_EXECUTION.md)

**Model:** AlgorimSeek‑7B (merged vision + code + execution fine‑tune)  
**Task:** Code generation, image understanding, bytecode compilation, step‑by‑step debugging  

### Key Features
- `/imagine` – generate a visualisation of an algorithm.
- `/compile` – translate Algorim code to a pseudo‑bytecode.
- `/debug` – produce an execution trace with variable states.

### Dataset
- Combines `algorim_training.json`, `algorim_vision_training.json`, and `algorim_execution_traces.json`.
- Additional command examples in `algorim_extra.json`.

### Training Pipeline
1. Pre‑train on text‑only code data.
2. Continue with vision‑language fine‑tuning (LoRA on Qwen2‑VL).
3. Final stage: instruction tuning with execution traces and special commands.

### Example Usage
```python
# /compile command
r = ask_algorimseek("""
/compile
Action binary_search(T: arr, n: int, target: int): int
var low, high, mid: int
begin
    low <--- 0; high <--- n-1
    while low <= high do
        mid <--- (low+high) div 2
        if T[mid] = target then return mid
        else if T[mid] < target then low <--- mid+1
        else high <--- mid-1
    done
    return -1
end
""")
print(r)  # Bytecode representation
```

---

## Repository Structure

```
.
├── 1000_code_dataset/          # Raw Algorim code files
├── 200_image_dataset/          # PNG renders of code snippets
├── notebooks/                  # Jupyter notebooks for exploration
├── Tools/                      # Helper scripts (data prep, evaluation)
├── algorim_training.json       # Main instruction tuning data (1.5M)
├── algorim_vision_training.json # Vision‑language pairs (1.2M)
├── algorim_dataset.csv         # Metadata for all examples
├── algorim_dataset.json        # Full structured dataset (2.3M)
├── algorim_execution_traces.json # Step‑by‑step traces
├── algorim_extra.json          # Additional command examples
├── demo_render.png             # Sample code image
├── AlgorimSeek_Training.ipynb  # Vision‑language fine‑tuning
└── AlgorimSeek_FineTuning.ipynb # Inference & evaluation
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Transformers >= 4.40
- PEFT, bitsandbytes, trl, datasets
- flash-attn (optional, for speed)

Install with:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use this dataset or models in your research, please cite:

```bibtex
@misc{algorimseek2025,
  title={AlgorimSeek: Fine‑tuning LLMs and VLMs for Pseudocode Comprehension},
  author={AlgorimSeek Team},
  year={2025},
  howpublished={\url{https://github.com/your-username/AlgorimSeek}}
}
```

---

## License

This project is released under the MIT License. The dataset is provided for academic and research purposes.

---

**Maintainer:** abdohak47  
**Last Updated:** May 31, 2026
