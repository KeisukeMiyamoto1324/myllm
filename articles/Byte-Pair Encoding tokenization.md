# Byte-Level BPE tokenizer とは何か

この記事では テキストをトークン列に変換する手法である Byte Pair Encoding という tokenizer について解説します。

この記事を書くにあたって、まず Hugging Face の LLM Course にある以下の記事を参考にさせていただきました。

[Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) 


## BPE とは何か

BPE の考え方を一言でいうと、「よく一緒に出てくる文字のペアを、どんどん一つの token としてまとめていく」というものです。

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


## BPE の学習例

ここでは `my cat ate a carrot` という 1 文だけをデータセットとして、BPE がどのように Vocabulary を学習するのかを具体的に見ていきます。

説明をわかりやすくするために、空白は `▁` という文字で表します。例えば ` cat` は `▁cat` と表します。また、実際の tokenizer では事前に文章を単語単位に分けてから BPE を学習することが多いため、ここでも以下の単位に分けて考えます。

```text
my
▁cat
▁ate
▁a
▁carrot
```

### 1. Vocabulary を初期化する

まず、それぞれの単位を文字に分割します。

```text
my       -> m, y
▁cat     -> ▁, c, a, t
▁ate     -> ▁, a, t, e
▁a       -> ▁, a
▁carrot  -> ▁, c, a, r, r, o, t
```

この時点で Vocabulary は、文章に登場した文字の集合になります。

```text
Vocabulary = { m, y, ▁, c, a, t, e, r, o }
```

最初の token 列は以下のようになります。

```text
my       -> m y
▁cat     -> ▁ c a t
▁ate     -> ▁ a t e
▁a       -> ▁ a
▁carrot  -> ▁ c a r r o t
```

### 2. 隣り合う token ペアを数える

次に、現在の token 列から隣り合う token のペアを作り、データセット全体で何回出現するかを数えます。

```text
(m, y) = 1
(▁, c) = 2
(c, a) = 2
(a, t) = 2
(▁, a) = 2
(t, e) = 1
(a, r) = 1
(r, r) = 1
(r, o) = 1
(o, t) = 1
```

最も出現回数が多いペアは複数あります。ここでは、同じ回数のペアが複数ある場合は、先に見つかったペアを選ぶことにします。したがって、最初は `(▁, c)` を merge します。

```text
新しい token = ▁c

Vocabulary = { m, y, ▁, c, a, t, e, r, o, ▁c }
```

token 列は以下のように更新されます。

```text
my       -> m y
▁cat     -> ▁c a t
▁ate     -> ▁ a t e
▁a       -> ▁ a
▁carrot  -> ▁c a r r o t
```

ここで `▁cat` と `▁carrot` に着目すると、学習する前はそれぞれ 4 トークン、7 トークン使用していたのに対し、1 step 学習後の現在は `_c` が Vocabulary に追加されたことで、3 トークン、6 トークンと少ないトークンで表現できていることがわかります。 

### 3. merge を繰り返す

更新後の token 列に対して、もう一度 token ペアを数えます。

```text
(m, y) = 1
(▁c, a) = 2
(a, t) = 2
(▁, a) = 2
(t, e) = 1
(a, r) = 1
(r, r) = 1
(r, o) = 1
(o, t) = 1
```

ここでは `(▁c, a)` を merge して、`▁ca` という新しい token を作ります。

```text
Vocabulary = { m, y, ▁, c, a, t, e, r, o, ▁c, ▁ca }
```

token 列は以下のようになります。

```text
my       -> m y
▁cat     -> ▁ca t
▁ate     -> ▁ a t e
▁a       -> ▁ a
▁carrot  -> ▁ca r r o t
```

さらに token ペアを数えると、`(▁, a)` が 2 回出現します。

```text
(▁, a) = 2
```

そこで `(▁, a)` を merge して、`▁a` という新しい token を作ります。

```text
Vocabulary = { m, y, ▁, c, a, t, e, r, o, ▁c, ▁ca, ▁a }
```

token 列は以下のようになります。

```text
my       -> m y
▁cat     -> ▁ca t
▁ate     -> ▁a t e
▁a       -> ▁a
▁carrot  -> ▁ca r r o t
```

ここまでで、よく出現する `▁c` や `▁a` のようなまとまりが token として Vocabulary に追加されました。これによって、各単語が最初より少ないトークンで表現できていることがわかります。

### 4. Vocabulary が指定の数に達するまで続ける

BPE の学習では、この merge を Vocabulary が指定されたサイズになるまで繰り返します。例えば現状は

```text
Vocabulary = { m, y, ▁, c, a, t, e, r, o, ▁c, ▁ca, ▁a }
```

なので、token が12個登録されていますが、最終的に大きさが 16 の Vocabulary を作りたいなら、merge を後4回繰り返します。

## BPE の弱点