# MyLLM

A minimal decoder-only Transformer project built with PyTorch and Lightning.

This repository includes a small from-scratch implementation for training and inference, mainly for learning and experimentation.

Most of the code in this repository is based on the StatQuest YouTube video "Coding a Decoder-Only Transformer from Scratch in PyTorch":
https://www.youtube.com/watch?v=C9QSpl5nmrY

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
python inference.py
```

## Files

- `train.py`: trains the model and saves weights to `model/model.pth`
- `inference.py`: loads the saved model and runs token generation
- `transformer.py`, `self_attention.py`, `position_encoding.py`: core model components
- `tokenizer.py`, `dataset.py`: tokenization and dataset utilities

## Model Structure

The model now follows a more standard decoder-only Transformer layout:

- token embedding + positional encoding
- repeated `DecoderBlock` layers
- each block contains:
  - masked multi-head self-attention
  - residual connection
  - layer normalization
  - feed-forward network
- final layer normalization + output projection

You can control the model structure from `DecoderOnlyTransformer`:

- `num_layers`: number of decoder blocks
- `num_heads`: number of attention heads
- `d_ff`: hidden size of the feed-forward layer
- `d_model`: embedding / hidden size
