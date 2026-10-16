<div align="center">

# 🤖 Mini GPT

### Decoder-Only Transformer from Scratch

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Parameters](https://img.shields.io/badge/Parameters-~3.23M-blueviolet?style=for-the-badge)
![Colab](https://img.shields.io/badge/Trained_on-Colab_T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muneer-ahmad10/mini-gpt-pytorch/blob/main/Mini_GPT_Colab.ipynb)

</div>

---

A GPT-style character-level language model implemented from scratch in PyTorch. This project explores the core components behind autoregressive language models, including causal self-attention, Transformer blocks, next-token prediction, model training, and text generation.

The model was trained on the Tiny Shakespeare dataset to learn character-level patterns and generate Shakespeare-style text.

## 📑 Table of Contents

- [Project Highlights](#-project-highlights)
- [Architecture](#-architecture)
- [Key Implementation Details](#-key-implementation-details)
- [Dataset](#-dataset)
- [Training Results](#-training-results)
- [Tech Stack](#-tech-stack)
- [Run the Project](#-run-the-project)
- [What I Learned](#-what-i-learned)
- [Current Limitations](#-current-limitations)
- [Potential Improvements](#-potential-improvements)
- [Repository Structure](#-repository-structure)
- [Conclusion](#-conclusion)

---

## ✨ Project Highlights

| | Detail |
|:--|:--|
| 🏗️ **Architecture** | Decoder-only Transformer |
| 🔥 **Framework** | PyTorch |
| 🧮 **Parameters** | Approximately 3.23 million |
| 🧱 **Transformer blocks** | 4 |
| 🎯 **Attention heads per block** | 4 |
| 📐 **Embedding dimension** | 256 |
| 🪟 **Context window** | 128 characters |
| 🔤 **Tokenization** | Character-level |
| 📚 **Training dataset** | Tiny Shakespeare |
| ☁️ **Training environment** | Google Colab with NVIDIA T4 GPU |

---

## 🏛️ Architecture

The model follows a GPT-style autoregressive architecture, predicting the next character using the preceding context.

```text
Input Text
    ↓
Character-Level Tokenization
    ↓
Token Embeddings + Positional Embeddings
    ↓
4 × Transformer Blocks
    ├── Layer Normalization
    ├── Causal Multi-Head Self-Attention
    ├── Residual Connection
    ├── Layer Normalization
    ├── Feed-Forward Network (GELU)
    └── Residual Connection
    ↓
Final Layer Normalization
    ↓
Linear Language-Model Head
    ↓
Next-Character Probability Distribution
    ↓
Autoregressive Text Generation
```

### ⚙️ Model Configuration

| Component | Configuration |
|:--|:--|
| Vocabulary size | Approximately 65 characters |
| Token embeddings | 256 dimensions |
| Positional embeddings | Learned |
| Transformer blocks | 4 |
| Attention heads per block | 4 |
| Dimension per attention head | 64 |
| Feed-forward hidden dimension | 1,024 |
| Maximum context length | 128 characters |
| Dropout | 0.2 |
| Output | Next-character logits |
| **Total parameters** | **Approximately 3.23M** |

---

## 🔧 Key Implementation Details

### 1️⃣ Character-Level Tokenization

- Built a character-to-integer mapping from the dataset vocabulary.
- Converted text into integer sequences for model training.
- Used a context window of up to 128 characters to predict the next character.

### 2️⃣ Causal Multi-Head Self-Attention

- Implemented query, key, and value projections using PyTorch linear layers.
- Applied a lower-triangular causal mask to prevent the model from attending to future positions.
- Used scaled dot-product attention, softmax, and dropout.
- Combined multiple attention heads to learn relationships between characters.

### 3️⃣ Transformer Blocks

- Implemented four Transformer blocks with pre-normalization.
- Used residual connections around attention and feed-forward sublayers.
- Built feed-forward networks using linear layers, GELU activation, and dropout.
- Combined token and learned positional embeddings to represent character identity and sequence position.

### 4️⃣ Training Pipeline

- Trained the model using next-token prediction and cross-entropy loss.
- Used the AdamW optimizer with a learning rate of `3e-4`.
- Applied gradient clipping with a maximum norm of `1.0`.
- Split the dataset into training and validation sets.
- Monitored training and validation loss during training.

### 5️⃣ Autoregressive Text Generation

- Generated text one character at a time using the preceding context.
- Limited the context to the model's maximum block size.
- Implemented temperature scaling and top-k sampling to control generation diversity.

---

## 📚 Dataset

The project uses the **Tiny Shakespeare** text dataset.

| Property | Value |
|:--|:--|
| Dataset | Tiny Shakespeare |
| Dataset size | 1,115,394 characters |
| Training split | 90% |
| Validation split | 10% |
| Prediction task | Next-character prediction |

> The model learns statistical patterns from Shakespeare's writing rather than retrieving or memorizing answers from a knowledge base. Since it uses character-level tokenization and a relatively small architecture, its generated text can contain grammatical inconsistencies and incoherent passages.

---

## 📈 Training Results

The following results are from the training run documented in the notebook.

| Metric | Result |
|:--|--:|
| Reported training iterations | 3,000 |
| Final training loss | 1.4420 |
| Final validation loss | 1.6375 |
| Training time | Approximately 288 seconds |
| Hardware | Google Colab NVIDIA T4 GPU |

> 💡 **How to interpret these results:** The training loss measures the model's next-character prediction error on training data, while validation loss measures the same objective on held-out data. These values help assess learning and generalization; they are not accuracy percentages or direct measures of text quality.

---

## 🧰 Tech Stack

| Category | Tools |
|:--|:--|
| **Language** | Python |
| **Deep learning** | PyTorch |
| **Data processing** | Python standard library |
| **Training environment** | Google Colab |
| **Dataset** | Tiny Shakespeare |

---

## 🚀 Run the Project

The implementation is provided as a Jupyter notebook.

1. Open [`Mini_GPT_Colab.ipynb`](./Mini_GPT_Colab.ipynb).
2. Upload the notebook to [Google Colab](https://colab.research.google.com/).
3. Select a GPU runtime if available.
4. Run the notebook cells in order.
5. Review the model configuration, training and validation loss, and generated text.

The notebook contains the model implementation, training loop, evaluation steps, and text-generation code.

---

## 🎓 What I Learned

- How decoder-only Transformer language models are structured.
- How causal masking prevents information leakage from future tokens.
- How multi-head self-attention models relationships within a sequence.
- How token and positional embeddings are combined.
- How cross-entropy loss, AdamW, and gradient clipping are used during training.
- How autoregressive generation and sampling parameters affect generated text.
- How to monitor training and validation loss when evaluating a language model.

---

## ⚠️ Current Limitations

- Character-level tokenization is less efficient than subword tokenization for many language-modeling tasks.
- The model's 128-character context window limits the amount of preceding text it can use.
- The approximately 3.23M-parameter model is intended for learning and experimentation, not production-scale language understanding.
- Generated text may be repetitive, inconsistent, or incoherent.
- The project currently focuses on training and generation rather than deployment as a production application.

---

## 🔮 Potential Improvements

- [ ] Add a saved model checkpoint and a separate inference workflow.
- [ ] Save the model configuration alongside the trained weights.
- [ ] Add a dependency file for easier environment setup.
- [ ] Compare generated text under different temperature and top-k settings.
- [ ] Experiment with subword tokenization, larger datasets, and longer context windows.
- [ ] Compare model configurations using consistent training and validation procedures.

---

## 🗂️ Repository Structure

```text
mini-gpt-pytorch/
├── Mini_GPT_Colab.ipynb
└── README.md
```

---

## 🏁 Conclusion

This project demonstrates a from-scratch implementation of a small GPT-style language model using PyTorch. It brings together causal multi-head self-attention, Transformer blocks, next-character prediction, model optimization, validation, and autoregressive text generation in a single learning project.

<div align="center">

**GitHub:** [muneer-ahmad10/mini-gpt-pytorch](https://github.com/muneer-ahmad10/mini-gpt-pytorch)

</div>
