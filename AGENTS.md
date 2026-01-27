## 初期のメモ(後でより適切な形に変更して良い)
指示書：Tiny Recursion Model(TRM)を自然言語生成モデル（Generative TRM）として実装せよ
0. あなたの役割

あなたは PyTorch を用いてニューラルネットを実装できるエンジニア兼研究者 です。
この指示書に基づき、論文 Tiny Recursive Model (TRM) の仕組みを参考にしながら、
日本語の自然言語生成を行う「Generative TRM」 を Python で実装してください。

1. 目標と前提
1.1 目標

TRM のアイデア（小さなネットワークを再帰的に呼び出すことで深い推論を実現）を用いて、

日本語の 言語モデル（次トークン予測） を実装する。

ただし HRM のような複雑な固定点定理・Q-learning は使わず、
TRM 流の「単一ネット + 再帰 + Deep Supervision + ACT(BCE)」の構造を採用する。

モデルは テキスト生成（サンプリング） もできるようにする。

1.2 データセット

事前学習には以下を想定する：

コーパス：llm-jp-corpus-v3 / ja / ja_wiki

使用ファイル：

train_10.jsonl.gz

validation_0.jsonl.gz

各 .jsonl.gz は、1行1サンプルの JSON とし、
"text" フィールドに日本語テキストが入っていると仮定してよい。

要求：

このデータを読み込み、

トークナイズ

バッチ化

次トークン予測用の (input_ids, target_ids) を作るデータローダ
を実装せよ。

2. TRM 本来の仕組み（元論文の要約）
2.1 役者：x, y, z と net

TRM では、以下の3つのベクトル（もしくはテンソル）が重要：

x：入力（問題）

y：現在の解答（solution embedding）

z：潜在的推論状態（latent reasoning feature）

更新則は以下のようなイメージ：

# 潜在推論の更新
z = net(x, y, z)

# 解答の更新
y = net(y, z)


ここで net は 同じ小さなネットワーク（パラメータ共有） であり、
入力を concat → 線形層 → FFN で処理する。

2.2 latent_recursion（n ステップの再帰）

latent_recursion(x, y, z, n) は、

上の (z = net(x,y,z), y = net(y,z)) を n回 繰り返し、

「思考 z と解答 y を段階的に磨く」モジュール。

2.3 deep_recursion（T 回のブロック + Deep Supervision）

deep_recursion(x, y, z, n, T) は、

最初の T − 1回 を torch.no_grad() で実行し、

最後の 1回だけ勾配を流す。

その後、(y, z) を detach() して次の supervision ステップへ渡す。

これにより、

実効的にはかなり深い再帰（n×T）を行いながら、

メモリと計算量を抑えつつ学習できる。

2.4 Deep Supervision と ACT(BCE)

同じ入力サンプルに対して、N_supervision ステップ分、
(y, z) を持ち回りで更新し続ける（Deep Supervision）。

各ステップで：

メイン損失：CrossEntropy(output_head(y), y_true)

停止判定：q_hat = Q_head(y) を出力させ、

「現時点の y_hat が正解かどうか」をラベルにして

Binary Cross Entropy で学習
→ これが TRM版 ACT。追加の forward を使わない 1-pass ACT。

3. Generative TRM への拡張方針

ここからが重要な本題です。
本来 TRM は「入力 x → 出力 y（単一正解）」の教師ありマッピングでしたが、
ここでは 自然言語生成（次トークン予測） に拡張します。

3.1 モデルの入出力定義

入力：トークン列

(
𝑡
1
,
𝑡
2
,
.
.
.
,
𝑡
𝐿
)
(t
1
	​

,t
2
	​

,...,t
L
	​

)

モデルが学習すべきは：

𝑝
(
𝑡
𝑖
+
1
∣
𝑡
1
,
.
.
.
,
𝑡
𝑖
)
p(t
i+1
	​

∣t
1
	​

,...,t
i
	​

)

のような 次トークン予測。

設計案：

x：入力文の埋め込み（全ステップで固定）

y：「現在の予測状態」の表現（初期値は x かゼロ埋め）

z：推論メモリ（初期はゼロ）

最終的な logits は output_head(y) から得て、
target_ids に対する クロスエントロピーで学習する。

3.2 アーキテクチャ
3.2.1 net の中身

自然言語はコンテキスト長 L が長くなり得るため、
Sudokuのような「MLPだけ」の構造ではなく、
Self-Attention + FFN のような Transformer ミニブロックを net として用いることを推奨。

例（イメージ）：

class TinyBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, h, attn_mask=None):
        # h: [B, L, D]
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        h = self.ln1(h + attn_out)
        ff_out = self.ffn(h)
        h = self.ln2(h + ff_out)
        return h


これを net() の内部で利用しつつ、
入力 (x,y,z) あるいは (y,z) を concat → 線形層で h に変換 → TinyBlock で処理
という構造にする。

3.2.2 入力の結合方法

net(x, y, z) のとき：

[x, y, z] を特徴次元方向（最後の次元）で concat

Linear(3*D, D) で圧縮 → TinyBlock

net(y, z) のとき：

[y, z] を concat → Linear(2*D, D) → TinyBlock

3.3 再帰構造（Training 時）

latent_recursion(x, y, z, n)：

n回ループし、毎回

z = net(x, y, z)

y = net(y, z)

deep_recursion(x, y, z, n, T)：

T−1回、with torch.no_grad(): latent_recursion(...)

最後に 1回だけ latent_recursion を勾配ありで実行

(y,z) を detach() して返却

学習ループ：

各バッチに対して N_supervision 回、

deep_recursionを呼んで y_hat を更新

output_head(y_hat) から logits → next-token cross entropy

Q_head(y_hat) から q_hat → ACTのBCE損失

両方を足して backward, step

q_hat がしきい値より高ければそのバッチは早期終了

3.4 推論時（生成）

生成モードでは、以下の流れを実装せよ：

プロンプト prompt_ids を与える。

その埋め込みを x として用い、y, z を初期化。

deep_recursion(x, y, z) を数回呼び出して 内部状態を安定化。

output_head(y) から logits を取り出し、

greedy, top-k, top-p などのサンプリングで次トークン t_{L+1} を選ぶ。

新たなトークンを末尾に追加した新シーケンスを x として再度再帰。

これを指定トークン数 or EOS が出るまで繰り返す。

4. データ読み込み・前処理
4.1 トークナイザ

BPE/SentencePiece いずれでもよいが、
日本語用のサブワードトークナイザ を利用せよ。

指示：

トークナイザクラスを抽象化し、encode(text) -> List[int], decode(ids) -> str を持たせる。

既存のトークナイザ（例：SentencePieceモデル）を読み込むコードも書けるとなお良い。

4.2 JSONL.GZ ローダー

gzip + json を用いて train_10.jsonl.gz と validation_0.jsonl.gz をストリーム読み込み。

各行について：

obj = json.loads(line) とし、

text = obj["text"] を取得。

必要なら長さ制限（最大シーケンス長）でカット。

4.3 next-token 用のバッチ化

1サンプルのテキストから：

input_ids = tokens[:-1]

target_ids = tokens[1:]

ミニバッチ化：

パディング（pad_id）＋ attention_mask を作る。

形：[B, L]

5. 実装に必要なコンポーネント一覧

以下のクラス・関数を定義せよ（名前は多少変えて良いが、役割は保つこと）：

TokenizerWrapper

encode(text: str) -> List[int]

decode(ids: List[int]) -> str

WikiJsonlDataset

.jsonl.gz を読み込んでトークナイズし、__getitem__ で (input_ids, target_ids) を返す。

DataLoader

PyTorch の DataLoader を利用。

TinyNet / TinyBlock

TRM の核となる再帰用ネットワーク net の本体。

TRMModel

forward(x_ids, ...) で

埋め込み → 再帰 (deep_recursion) → logits

さらに generate(prompt_ids, max_new_tokens, ...) を実装。

Training Loop

Optimizer（AdamWなど）

LR scheduler（任意）

EMA 更新（Exponential Moving Average）のオプション

Config

d_model, n_heads, vocab_size, n, T, N_supervision, max_seq_len などの設定管理。

6. 実装上のポイント・注意

再帰回数：

最初は小さめ（例：n=2, T=2, N_supervision=2）で動作を確認し、
その後段階的に増やしていく設計にせよ。

ACTの停止判定：

q_hat は sigmoid されたスカラー（あるいは [B,1]）。

学習時のラベルは 1（正解時） or 0（不正解時）。

推論時の早期停止は q_hat > threshold（例：0.7）などで判定。

EMA：

学習時に ema_model を更新し、評価時は EMA 版パラメータを使う実装が望ましい。

メモリ対策：

with torch.no_grad() ブロック内では勾配・グラフを一切持たないこと。

detach() を忘れないこと。

7. 最終アウトプットの期待される形

あなたが出力すべきものは：

model.py

TRM本体（TinyNet / TRMModel）の実装

dataset.py

トークナイザラッパ・ja_wiki JSONL.GZ ローダー

train_trm_lm.py

学習スクリプト（引数で学習ステップ数・バッチサイズなど指定可能）

generate.py

学習済みモデルを使った日本語テキスト生成スクリプト（プロンプトを与えて続きを生成）

なお、最初のバージョンでは「とりあえず学習が回る・生成が動く」ことを優先し、

モデルサイズは小さめ（例：d_model=256, n_heads=4程度）

学習ステップも少なめで良い。