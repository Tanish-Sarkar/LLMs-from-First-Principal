# 📚 LLMs-from-First-Principal

![GPT Architecture Diagram](https://towardsdatascience.com/wp-content/uploads/2023/07/10N0aHoN6MzSvFloJiSS1Rg.png)

A **hands‑on, from‑scratch implementation** of the core building blocks of modern Large Language Models (LLMs) – from tokenisation to the full GPT‑style transformer.  This repository is a learning playground for anyone who wants to understand the inner workings of LLMs at a **deep, code‑level** while building a **production‑ready, well‑structured** code base.

---

## ✨ Why This Repo?
- **Educational depth** – Every component (embedding, attention, feed‑forward, optimizer) is implemented from first principles with clear, commented PyTorch code.
- **Modular design** – Plug‑and‑play modules let you experiment with shortcuts, layer‑norm variations, and custom loss functions.
- **Premium aesthetics** – The notebooks feature clean visualisations, gradient‑rich plots, and a polished README that feels like a professional technical spec.
- **Ready for research** – Baselines for pre‑training, fine‑tuning, and inference are included, making it easy to extend the model for your own experiments.

---

## 🚀 Features
- **Tokenizer**: Byte‑Pair Encoding (BPE) with customizable vocab size.
- **Embedding layer** with sinusoidal positional encodings.
- **Multi‑head self‑attention** with optional residual shortcut connections.
- **Feed‑forward network** (GeLU activation) with configurable hidden dimensions.
- **Layer‑norm** placed before/after attention as per recent best‑practice papers.
- **Training loop** with mixed‑precision support, gradient clipping, and learning‑rate schedulers.
- **Utilities** for gradient inspection, model checkpointing, and generation scripts.

---

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/Tanish-Sarkar/LLMs-from-First-Principal.git
cd LLMs-from-First-Principal

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
> **Tip**: Use a GPU‑enabled PyTorch build for faster training.

---

## 🛠️ Quick Start
```python
import torch
from model import ExampleDeepNeuralNetwork
from utils import print_gradients

# Hyper‑parameters
seq_len = 128
batch_size = 8
vocab_size = 50257
embed_dim = 768
n_layers = 12
n_heads = 12

# Initialise model
model = ExampleDeepNeuralNetwork(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_heads=n_heads,
    use_shortcut=True,
)

# Dummy input (batch of token IDs)
sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))

# Forward and inspect gradients
print_gradients(model, sample_input)
```
The script prints the mean absolute gradient of each weight tensor, confirming that back‑propagation works through the entire transformer stack.

---

## 📚 Notebook Tour
- **`GPT_ARCHITECTURE.ipynb`** – Step‑by‑step walkthrough of the model construction, training loop, and generation examples.
- **`GRADIENT_INSPECTION.ipynb`** – Visualise gradient magnitudes across layers to debug training stability.
- **`TRAINING_LOOP.ipynb`** – End‑to‑end training on a small dummy dataset with live loss curves.

---

## 🤝 Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your‑idea`).
3. Ensure code style with `ruff` and pass all tests (`pytest -q`).
4. Open a Pull Request with a clear description of your changes.

---

## 📄 License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 🙏 Acknowledgements
- The original **Transformer** paper by Vaswani *et al.* (2017).
- Open‑source implementations that inspired this educational repo (e.g., `minGPT`, `nanoGPT`).
- The PyTorch community for providing an excellent deep‑learning framework.

---

*Happy hacking and enjoy building your own LLMs from the ground up!*
