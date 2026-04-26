# Byte-Level BPE tokenizer とは何か

この記事では テキストをトークン列に変換する手法である Byte-Level BPE (Byte-Level Byte Pair Encoding) という tokenizer について解説します。

この記事を書くにあたって、まず Hugging Face の LLM Course にある以下の記事を参考にさせていただきました。

[Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) 


## BPE とは何か

まず、 Byte-Level BPE の前身である BPE (Byte Pair Encoding) について解説します。BPE の考え方を一言でいうと、「よく一緒に出てくる文字のペアを、どんどん一つの token としてまとめていく」というものです。

仕組みを説明するために以下の用語を定義します。

- Token: tokenizer が実際に扱う単位。最初は文字単位から始まり、学習によって subword や単語のような大きな単位に成長していきます。
- Vocabulary (語彙): tokenizer が持つ token の集合。

BPE を学習するには、大まかに以下のステップを踏みます。

1. Vocabulary を初期化する。
   1. データセットとして与えられた文章を文字の最小単位に分割する（例えば dog が与えられたら d, o, g のように分割する)。
   2. 分割された文字を全て Vocabulary に追加する。
2. Vocabulary 内の token の merge を繰り返す。
   1. 現在の Vocabulary を使用してデータセットとして与えられた文章を tokenize する。
   2. tokenize され文章の中で、隣り合っている token のペアを作る。同じペアの出現回数をデータセット全体にわたってカウントする。
   3. 最も出現回数の多かった token ペアをマージし、新しい token を作成する。
   4. 3で作った token を Vocabulary に追加する。
3. 2 を Vocabulary が指定の数に達するまで繰り返す。

