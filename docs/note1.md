## はじめに

PyTorch と PyTorch Lightning を用いて、非常に小規模な Decoder-only の Transformer を構築しました。本プロジェクトのコードは私の GitHub レポジトリ（[https://github.com/KeisukeMiyamoto1324/myllm](https://www.google.com/search?q=https://github.com/KeisukeMiyamoto1324/myllm)）にて公開しています。また、本実装は主に YouTube チャンネル「StatQuest with Josh Starmer」の動画「Coding a Decoder-Only Transformer from Scratch in PyTorch」を大いに参考にして作成しました。

## Token と Position Encoding の基礎

自然言語処理において、コンピュータは人間が書いたテキストをそのまま理解することはできません。そのため、まずはテキストを Token と呼ばれる最小単位に分割し、それぞれに一意の数値を割り当てる必要があります。さらに、それらの数値を多次元のベクトルに変換する Embedding という処理を行うことで、コンピュータが計算できる形式にします。これに加えて Transformer では、文章の中での単語の並び順をモデルに伝えるために Position Encoding という技術を使用します。`position_encoding.py` ファイルの `PositionEncoding` クラスでは、サイン関数とコサイン関数を用いて各 Token の位置情報を計算する処理を実装しています。

```python
class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
```

このコードでは、偶数番目のインデックスにはサイン関数を、奇数番目のインデックスにはコサイン関数を適用することで、波の性質を利用した位置表現を作成しています。計算された位置情報は学習によって更新される Parameter ではないため、PyTorch の register\_buffer を用いて Buffer としてモデルに登録し、Forward 処理の際に元の Embedding ベクトルに足し合わせる仕組みになっています。

## Self-Attention の仕組み

Transformer の最大の特徴は Self-Attention というメカニズムです。これは、文章中のある単語が、同じ文章の中の他のどの単語と強い関連を持っているかを計算するための仕組みです。`self_attention.py` ファイルの `Attention` クラスに、この処理の心臓部が記述されています。

```python
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
```

ここでは Linear 層を用いて、入力データから Query、Key、Value という3つの行列を生成しています。検索エンジンに例えると、Query は「検索キーワード」、Key は「記事のタイトル」、Value は「記事の中身」のような役割を果たします。順伝播の過程で Query と Key の内積を計算して単語間の類似度 Score を求めます。さらに Decoder モデル特有の処理として、文章を生成する際に未来の Token をカンニングしてしまわないための Mask 処理が重要になります。`self_attention.py` の Forward メソッド内では、Mask された箇所に極端に小さな値を代入し、Softmax 関数を通すことで未来の情報への影響がゼロになるように設計されています。

## Decoder-only Transformer の全体像

これまで解説した基礎的なコンポーネントを組み合わせて、最終的なモデルの全体像を構築します。今回作成したのは Decoder-only というアーキテクチャであり、入力された Token 列から次に続く Token を予測することに特化しています。`transformer.py` ファイルの `DecoderOnlyTransformer` クラスの Forward メソッドに、データが流れる一連の経路が記述されています。

```python
    def forward(self, token_ids: torch.Tensor):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        
        mask = torch.tril(
            torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=token_ids.device)
        )
        mask = mask == 0
        
        self_attention_value = self.self_attention(
            position_encoded,
            position_encoded,
            position_encoded,
            mask
        )
        
        residual_connection_values = position_encoded + self_attention_value
        fc_layer_output = self.fc_layer(residual_connection_values)
        
        return fc_layer_output
```

Forward 処理では、まず入力 Token を Embedding ベクトルに変換し、Position Encoding を加算します。その後、未来の情報を隠すための下三角行列の Mask を動的に生成し、Self-Attention 層へと渡します。アテンションの出力に対して、元の入力をそのまま足し合わせる Residual Connection という処理を行い、勾配消失を防ぎます。最後に Linear 層を通して、次にどの Token が来るべきかの確率の指標となる Logit を出力して処理が完了します。

## PyTorch Lightning を用いた Training の簡素化

モデルに賢さを身につけさせるための Training の工程では、PyTorch Lightning を活用することで、煩雑になりがちな学習ループのコード記述を大幅に削減しています。`transformer.py` ファイルの `DecoderOnlyTransformer` クラスの後半部分に、最適化アルゴリズムの設定と1 Step ごとの Loss 計算が定義されています。

```python
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])
        
        return loss
```

通常であれば Epoch を回すループを書き、Gradient を初期化し、Loss を計算して Backpropagation を行うという一連の処理をすべて手作業で記述する必要があります。しかし、PyTorch Lightning を利用することでこれらの定型的な処理をフレームワーク側に任せることができます。実際のモデル学習の実行は `train_model.py` ファイルにて Trainer の fit メソッドを呼び出すだけで完結し、Tokenizer の学習は `train_tokenizer.py` に分離されているため、モデルのアーキテクチャ自体の理解や実験に集中することができます。
