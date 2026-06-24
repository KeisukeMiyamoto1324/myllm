# lambda

A small decoder-only Transformer project built with PyTorch and Lightning.

It includes tokenizer training, pretraining, midtraining, posttraining, and inference code.

## Setup

```bash
pip3 install -r requirements.txt
```

## Usage

```bash
python3 src/tokenizer/train.py
python3 src/pretraining/train.py
python3 src/midtraining/train.py --model-path "model file path"
python3 src/inference_base/inference.py --prompt "Hello"
```

## Test

```bash
python3 -m pytest
```
