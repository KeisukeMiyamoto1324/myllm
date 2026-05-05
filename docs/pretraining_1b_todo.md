# src/pretraining 1B Training TODO

1B 規模の事前学習に向けて、リスクが高・中の改善項目を管理する TODO。
既存のデフォルトパラメータ値は対象外とする。

## 高リスク

- [ ] 分散学習とメモリ削減戦略を追加する
  - 対象: `src/pretraining/train.py`, `src/pretraining/transformer.py`
  - 現状: `devices=1` 固定で、通常の `AdamW` が全パラメータと optimizer state を保持する。
  - TODO: Lightning の FSDP または DeepSpeed ZeRO、gradient accumulation、必要に応じて optimizer state sharding を導入する。
  - 完了条件: 1B 付近の構成で OOM を避けて学習開始でき、実効 batch size を CLI から制御できる。

- [ ] streaming dataset のシャッフルを追加する
  - 対象: `src/pretraining/dataset.py`
  - 現状: Hugging Face streaming dataset を順番に読み込んでおり、データ順序の偏りが学習に影響しやすい。
  - TODO: buffer shuffle、または tokenized shard 化による shard/sample シャッフルを導入する。
  - 完了条件: train stream が固定順序ではなく、seed 管理された再現可能なシャッフルで供給される。

- [x] train/validation split を worker 数に依存しない安定分割にする
  - 対象: `src/pretraining/dataset.py`
  - 現状: DataLoader worker で shard した後に `enumerate()` して modulo split しているため、worker ごとの local index によって validation 文書が train 側に混ざる可能性がある。
  - TODO: 文書 ID、安定 hash、または shard 前の global index に基づく split に変更する。
  - 完了条件: `num_workers` を変えても train/validation の文書集合が変わらず、互いに disjoint になる。

## 中リスク

- [ ] LR scheduler、warmup、gradient clipping を追加する
  - 対象: `src/pretraining/train.py`, `src/pretraining/transformer.py`
  - 現状: 固定 learning rate の AdamW で、warmup と decay と gradient clipping がない。
  - TODO: `configure_optimizers()` で optimizer と scheduler を返し、Trainer に `gradient_clip_val` を追加する。
  - 完了条件: warmup steps、total steps、minimum LR、clip 値を設定でき、ログに LR が出る。

- [ ] activation checkpointing を追加する
  - 対象: `src/pretraining/transformer.py`
  - 現状: 全 decoder block の activation を通常どおり保持するため、1B 規模では activation memory が大きくなる。
  - TODO: 各 `DecoderBlock` を `torch.utils.checkpoint.checkpoint` で再計算可能にする。
  - 完了条件: CLI から有効化でき、有効時に同じ batch/context で peak memory が下がる。

- [ ] 位置エンコーディングを RoPE に変更する
  - 対象: `src/pretraining/position_encoding.py`, `src/pretraining/self_attention.py`, `src/pretraining/transformer.py`
  - 現状: token embedding に固定 sinusoidal position を加算している。
  - TODO: Q/K に Rotary Position Embedding を適用する実装へ変更する。
  - 完了条件: 通常 forward と cache inference の両方で position offset が正しく反映される。

- [ ] FFN を SwiGLU または GEGLU に変更する
  - 対象: `src/pretraining/transformer.py`
  - 現状: `Linear -> GELU -> Linear` の標準 MLP。
  - TODO: gated MLP に変更し、パラメータ数が大きく変わりすぎないよう hidden size を調整可能にする。
  - 完了条件: FFN 種別と hidden size を CLI から制御できる。

- [ ] AdamW の weight decay 対象を分離する
  - 対象: `src/pretraining/transformer.py`
  - 現状: `self.parameters()` 全体に同じ AdamW 設定を適用している。
  - TODO: 行列重みは decay 対象、bias、RMSNorm weight、embedding は decay 対象外に分ける。
  - 完了条件: optimizer param group が decay/no_decay に分かれ、対象パラメータ名を検証できる。
