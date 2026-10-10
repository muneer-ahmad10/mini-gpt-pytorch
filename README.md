# Mini GPT: A Decoder-Only Transformer from Scratch in PyTorch

A small GPT-style language model built from scratch in PyTorch and trained on character-level text (Tiny Shakespeare). The goal of this project is to understand how GPT models work internally by implementing every component by hand, without using `nn.Transformer` or Hugging Face.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muneer-ahmad10/mini-gpt-pytorch/blob/main/Mini_GPT_Colab.ipynb)

## What is implemented

- **Token and positional embeddings:** convert token IDs to vectors and add position information, since attention has no sense of order
- **Causal multi-head self-attention:** scaled dot-product attention with a lower-triangular mask, so each token can only attend to itself and earlier tokens
- **Transformer blocks:** attention and feed-forward layers with residual connections and LayerNorm
- **Training loop:** next-token prediction with cross-entropy loss, AdamW, gradient clipping and a train/validation split
- **Autoregressive generation:** predict one token, append it, repeat, with temperature and top-k sampling

## Model configuration

| Setting | Value |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Context length | 128 tokens |
| Dropout | 0.2 |
| Parameters | ~3.2M |
| Vocabulary | character-level (~65 characters) |

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), about 1.1 million characters, split 90% for training and 10% for validation. The notebook downloads it automatically.

## Results

| Metric | Value |
|---|---|
| Final train loss | [fill in from your run] |
| Final validation loss | [fill in from your run] |
| Training time (Colab T4 GPU) | [fill in from your run] |

Loss curves are plotted in the notebook.

### Sample generated text

```
[paste a short sample from your own run here]
```

The model is small and trained for a few minutes, so the output imitates the style of Shakespeare (character names, line breaks, archaic words) but does not always make sense. This is expected at this scale.

## How to run

1. Click the **Open In Colab** badge above (or upload `Mini_GPT_Colab.ipynb` to [Google Colab](https://colab.research.google.com)).
2. Go to **Runtime → Change runtime type** and select **T4 GPU**.
3. Run all cells. Training takes roughly 5 to 10 minutes.

To run locally, install the dependencies and open the notebook in Jupyter:

```bash
pip install torch matplotlib jupyter
```

## Project structure

```
mini-gpt-pytorch/
├── Mini_GPT_Colab.ipynb   # full implementation, training and generation
└── README.md
```

## Key concepts demonstrated

- Why a **causal mask** is needed so the model cannot see future tokens during training
- Why **positional embeddings** are required in attention-based models
- How **multi-head attention** lets different heads learn different relationships
- How **residual connections** and **LayerNorm** stabilize deep networks
- How GPT generates text one token at a time

## Limitations and next steps

- Character-level tokenization is simple but inefficient. A BPE tokenizer would give better results.
- The model is far too small to produce coherent long text.
- Possible extensions: train on a larger dataset, add a BPE tokenizer, add weight tying, try different context lengths, and compare with a pretrained GPT-2 from Hugging Face.

## Acknowledgments

The Tiny Shakespeare dataset comes from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository. The overall design follows the standard GPT architecture from "Attention Is All You Need" (Vaswani et al., 2017) and the GPT-2 paper.

## Author

**Muneer Ahmad Dar**: [GitHub](https://github.com/muneer-ahmad10)
