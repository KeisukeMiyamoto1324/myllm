# MyLLM

A minimal decoder-only Transformer project built with PyTorch and Lightning.

This repository includes a small from-scratch implementation for training and inference, mainly for learning and experimentation.

Most of the code in this repository is based on the StatQuest YouTube video "Coding a Decoder-Only Transformer from Scratch in PyTorch":
https://www.youtube.com/watch?v=C9QSpl5nmrY

## Files

- `train.py`: trains the model and saves weights to `model/model.pth`
- `inference.py`: loads the saved model and runs token generation
- `transformer.py`, `self_attention.py`, `position_encoding.py`: core model components
- `tokenizer.py`, `dataset.py`: tokenization and dataset utilities
