"""
Code Deep Dive: model.py の入出力処理を実装レベルで理解する

このファイルは、実際の model.py のコードを引用しながら、
入出力と再帰的処理の実装詳細を説明しています。
"""

# ============================================================================
# 1. forward() メソッド - 訓練時の全体フロー
# ============================================================================

"""
【ファイル】 model.py, lines ~220-310

def forward(
    self,
    input_ids: torch.Tensor,                    # [B, L]
    attention_mask: Optional[torch.Tensor] = None,
    target_ids: Optional[torch.Tensor] = None,  # [B, L]
    supervision_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    
    # ────────────────────────────────────────────────────────────────
    # ステップ1: トークンをembedding + 位置エンコーディング
    # ────────────────────────────────────────────────────────────────
    attention_mask = attention_mask if attention_mask is not None else (input_ids != self.config.pad_token_id).long()
    
    token_embeddings = self.token_embed(input_ids)  # [B, L, D]
    x = self._add_positional_encoding(token_embeddings)  # [B, L, D]
    
    # ここで x は「入力シーケンスの固定された埋め込み表現」
    # 以後の再帰ステップで x は一切変わらない（重要！）
    
    # ────────────────────────────────────────────────────────────────
    # ステップ2: y（推論状態）と z（潜在推論）を初期化
    # ────────────────────────────────────────────────────────────────
    y, z = self._init_states(x)  # y = x.clone(), z = zeros_like(x)
                                  # 両方とも [B, L, D]
    
    # ────────────────────────────────────────────────────────────────
    # ステップ3: Deep Supervision ループ
    # ────────────────────────────────────────────────────────────────
    sup_steps = supervision_steps or self.config.supervision_steps
    logits_per_step: List[torch.Tensor] = []
    q_hats: List[torch.Tensor] = []
    ce_losses: List[torch.Tensor] = []
    act_losses: List[torch.Tensor] = []
    
    for sup_idx in range(sup_steps):  # 例: 2回
        
        # ────────────────────────────────────────────────────────────
        # ステップ3a: Deep Recursion（複数の no_grad + 最後のみ grad）
        # ────────────────────────────────────────────────────────────
        y_grad, _, y, z = self._deep_recursion_step(x, y, z, attention_mask)
        
        # 返り値の意味：
        # - y_grad: 勾配を記録した版（損失計算用）
        # - _: 使わない
        # - y: 勾配を切った版（次のsupervision_stepの入力）
        # - z: 勾配を切った版
        
        # ────────────────────────────────────────────────────────────
        # ステップ3b: 出力層で確率分布を計算
        # ────────────────────────────────────────────────────────────
        logits = self.output_head(y_grad)  # [B, L, vocab_size]
        
        # logits[b, i, :] = 「位置iでの推論から予測される
        #                     次トークンの確率分布」
        
        # ────────────────────────────────────────────────────────────
        # ステップ3c: ACT用の「確信度」を計算
        # ────────────────────────────────────────────────────────────
        pooled = self._pool_for_q(y_grad, attention_mask)  # [B, D]
        q_hat = torch.sigmoid(self.q_head(pooled))  # [B, 1]
        
        # q_hat = 「現在の推論状態y_grad から見て、
        #          このバッチは正確に解けているか」の確信度
        
        q_hats.append(q_hat)
        logits_per_step.append(logits)
        
        # ────────────────────────────────────────────────────────────
        # ステップ3d: 損失を計算
        # ────────────────────────────────────────────────────────────
        if target_ids is not None:
            
            # Cross Entropy: 各位置での予測と正解を比較
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*L, vocab]
                target_ids.view(-1),               # [B*L]
                ignore_index=self.config.pad_token_id,
            )
            ce_losses.append(ce)
            
            # ACT Label: 実際の正確さ
            labels = self._compute_act_labels(
                logits,
                target_ids,
                attention_mask,
                threshold=self.config.act_threshold,
            )
            # labels[b, 0] = 1 if sequence b が十分正確, 0 otherwise
            
            # Binary Cross Entropy: q_hatが正確さを予測しているか
            bce = F.binary_cross_entropy(q_hat, labels)
            act_losses.append(bce)
    
    # ────────────────────────────────────────────────────────────
    # ステップ4: 複数ステップの損失を集約
    # ────────────────────────────────────────────────────────────
    loss = None
    ce_loss = torch.stack(ce_losses).mean() if ce_losses else None
    act_loss = torch.stack(act_losses).mean() if act_losses else None
    
    if ce_loss is not None and act_loss is not None:
        loss = ce_loss + self.config.act_weight * act_loss
    elif ce_loss is not None:
        loss = ce_loss
    
    # ────────────────────────────────────────────────────────────
    # ステップ5: 返却
    # ────────────────────────────────────────────────────────────
    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "act_loss": act_loss,
        "logits": logits_per_step[-1] if logits_per_step else None,
        "all_logits": logits_per_step,
        "q_hats": q_hats,
    }


【重要なポイント】
①「x」は最初に1回だけ計算され、以後変わらない
②「y」と「z」は毎回改善される
③「logits」は毎回（supervision_stepごとに）出力される
④「loss」は複数ステップの平均
"""

# ============================================================================
# 2. _deep_recursion_step() - 再帰的更新
# ============================================================================

"""
【ファイル】 model.py, lines ~180-210

def _deep_recursion_step(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    y_state, z_state = y, z
    tracked_y = None
    tracked_z = None
    
    # Deep Steps をループ（例: 2回）
    for depth in range(self.config.deep_steps):  # deep_steps=2
        
        use_grad = depth == self.config.deep_steps - 1  # 最後だけTrue
        
        if use_grad:
            # ★ 最後のステップだけ勾配を記録
            y_state, z_state = self.latent_recursion(x, y_state, z_state, attention_mask)
            tracked_y, tracked_z = y_state, z_state
            # これらは backward() で勾配が流れる
        else:
            # 最初の deep_steps-1 ステップは勾配を記録しない
            with torch.no_grad():
                y_state, z_state = self.latent_recursion(x, y_state, z_state, attention_mask)
    
    assert tracked_y is not None and tracked_z is not None
    
    # ★ 重要：勾配ありの版と勾配を切った版の両方を返す
    return tracked_y, tracked_z, tracked_y.detach(), tracked_z.detach()
    #      ↑ 損失計算用     ↑使わない   ↑次のステップ用


【何をしているのか】
──────────────────────────────────────────

例: deep_steps=2

初回呼び出し時（sup_idx=0）:
  ├─ depth=0:
  │  └─ with torch.no_grad():
  │     └─ y, z = latent_recursion(x, y, z)
  │        [B, L, D] が改善される
  │
  └─ depth=1:  ← use_grad = True
     └─ y, z = latent_recursion(x, y, z)  ← 勾配あり
        tracked_y, tracked_z を保存

返却:
  ├─ tracked_y: 勾配グラフを保持（逆伝播可能）
  ├─ tracked_z: 勾配グラフを保持
  ├─ tracked_y.detach(): 勾配グラフを破棄
  └─ tracked_z.detach(): 勾配グラフを破棄


【メリット】
──────────────────────────────────────────
- 実質的に deep_steps × latent_steps の深さで処理
  （この例では 2 × 2 = 4層分）
- メモリ効率：最後のステップだけ履歴を保持
- 計算効率：no_grad ブロックはPythonの実行速度が上がる
"""

# ============================================================================
# 3. latent_recursion() - 再帰的な y, z の更新
# ============================================================================

"""
【ファイル】 model.py, lines ~155-165

def latent_recursion(
    self,
    x: torch.Tensor,        # [B, L, D] - 入力（固定）
    y: torch.Tensor,        # [B, L, D] - 推論状態（改善される）
    z: torch.Tensor,        # [B, L, D] - 潜在推論（改善される）
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # latent_steps 回ループ（例: 2回）
    for _ in range(self.config.latent_steps):
        
        # 更新式1: z を改善（x と y の新しい情報を取り込む）
        z = self.tiny_net.forward_xyz(x, y, z, attention_mask)  # [B, L, D]
        
        # 更新式2: y を改善（改善された z を使う）
        y = self.tiny_net.forward_yz(y, z, attention_mask)  # [B, L, D]
    
    return y, z


【何をしているのか】
──────────────────────────────────────────

Iteration 1:
  ├─ z = net(concat[x, y_old, z_old])  ← x は常に同じ
  └─ y = net(concat[y_new, z_new])

Iteration 2:  ← 同じことをもう一度
  ├─ z = net(concat[x, y, z])  ← 改善されたyを使用
  └─ y = net(concat[y, z])     ← さらに改善


【イメージ】
──────────────────────────────────────────
迷路の例で考えると：
  初期状態: y = [未確定, 未確定, 未確定, ...]
  
  Iteration 1:
    「このマスから周りを見ると、このマスは通路っぽい」
    y = [0.3, 0.7, 0.2, ...] ← まだ不確実
  
  Iteration 2:
    「周りが通路だと分かったから、このマスも通路に違いない」
    y = [0.8, 0.9, 0.7, ...] ← より確実に
"""

# ============================================================================
# 4. TinyNet - 再帰の核となるネットワーク
# ============================================================================

"""
【ファイル】 model.py, lines ~70-90

class TinyNet(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.xyz_proj = nn.Linear(d_model * 3, d_model)  ← 3つ入力を圧縮
        self.yz_proj = nn.Linear(d_model * 2, d_model)   ← 2つ入力を圧縮
        self.block = TinyBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
    
    def forward_xyz(
        self,
        x: torch.Tensor,  # [B, L, D]
        y: torch.Tensor,  # [B, L, D]
        z: torch.Tensor,  # [B, L, D]
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 3つのベクトルを特徴次元で連結
        h = torch.cat([x, y, z], dim=-1)  # [B, L, D*3]
        
        # Linear層で圧縮
        h = self.xyz_proj(h)  # [B, L, D]
        
        # Transformer ブロック（Self-Attention + FFN）
        return self.block(h, attn_mask=attn_mask)  # [B, L, D]
    
    def forward_yz(
        self,
        y: torch.Tensor,  # [B, L, D]
        z: torch.Tensor,  # [B, L, D]
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 2つのベクトルを特徴次元で連結
        h = torch.cat([y, z], dim=-1)  # [B, L, D*2]
        
        # Linear層で圧縮
        h = self.yz_proj(h)  # [B, L, D]
        
        # Transformer ブロック
        return self.block(h, attn_mask=attn_mask)  # [B, L, D]


【アーキテクチャ】
──────────────────────────────────────────

forward_xyz(x, y, z):
  
  [x: D] ┐
  [y: D] ├─ concat ─→ [D*3] ─→ Linear ─→ [D] ─→ TinyBlock ─→ [D]
  [z: D] ┘                     (xyz_proj)


forward_yz(y, z):
  
  [y: D] ┐
  [z: D] ├─ concat ─→ [D*2] ─→ Linear ─→ [D] ─→ TinyBlock ─→ [D]
         └            (yz_proj)
"""

# ============================================================================
# 5. generate() - 自動回帰的なテキスト生成
# ============================================================================

"""
【ファイル】 model.py, lines ~330-380

@torch.no_grad()
def generate(
    self,
    prompt_ids: Sequence[int],        # [L_prompt] トークンID列
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    supervision_steps: Optional[int] = None,
    q_threshold: Optional[float] = 0.7,
    device: Optional[torch.device] = None,
) -> List[int]:
    
    device = device or next(self.parameters()).device
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # shape: [1, L_prompt]
    
    for step in range(max_new_tokens):
        # ────────────────────────────────────────────────────────────
        # ステップ1: 現在の input_ids（1トークン増えている）を処理
        # ────────────────────────────────────────────────────────────
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        token_embeddings = self.token_embed(input_ids)  # [1, L, D]
        x = self._add_positional_encoding(token_embeddings)  # [1, L, D]
        
        y, z = self._init_states(x)  # y = x, z = 0
        
        # ────────────────────────────────────────────────────────────
        # ステップ2: 推論サイクル（no_grad内なので勾配なし）
        # ────────────────────────────────────────────────────────────
        y, z, _ = self._inference_cycle(
            x, y, z, attention_mask, supervision_steps, q_threshold
        )
        
        # ────────────────────────────────────────────────────────────
        # ステップ3: 出力層で確率分布を計算
        # ────────────────────────────────────────────────────────────
        logits = self.output_head(y)  # [1, L, vocab_size]
        
        # ────────────────────────────────────────────────────────────
        # ★ ステップ4: 最後の位置だけ取得（重要！）
        # ────────────────────────────────────────────────────────────
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        # [1, vocab_size]
        # logits[:, -1, :] = 「最後のトークン位置での推論」
        
        # ────────────────────────────────────────────────────────────
        # ステップ5: サンプリング
        # ────────────────────────────────────────────────────────────
        next_token = _sample_logits(next_token_logits, top_k=top_k, top_p=top_p)
        # [1]
        
        # ────────────────────────────────────────────────────────────
        # ステップ6: シーケンスに追加
        # ────────────────────────────────────────────────────────────
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        # [1, L+1] に増えた
        
        # ────────────────────────────────────────────────────────────
        # ステップ7: EOS (End of Sequence) チェック
        # ────────────────────────────────────────────────────────────
        if self.config.pad_token_id in next_token.tolist():
            break
    
    return input_ids.squeeze(0).tolist()  # [L] → List[int]


【シーケンス長の推移】
──────────────────────────────────────────

初期: input_ids = [235, 67, 432, 189]  L=4

Step 1:
  L=4 → process → logits[1, 4, vocab]
  next_token_logits = logits[:, -1, :]  ← logits[0, 3, :]
  sample → token=500
  input_ids = [235, 67, 432, 189, 500]  L=5

Step 2:
  L=5 → process → logits[1, 5, vocab]
  next_token_logits = logits[:, -1, :]  ← logits[0, 4, :]
  sample → token=234
  input_ids = [235, 67, 432, 189, 500, 234]  L=6

...

Step 64:
  L=67 → process → logits[1, 67, vocab]
  next_token_logits = logits[:, -1, :]  ← logits[0, 66, :]
  sample → token=EOS
  break


【鍵となる工夫】
──────────────────────────────────────────
「logits[:, -1, :] だけを使う」
→ 「次の位置の予測」を1個だけ取り出す
→ Causal な自動回帰生成を実現
"""

# ============================================================================
# 6. 入出力の数学的な見方
# ============================================================================

"""
【表記】
x:  入力埋め込み [B, L, D]
y:  推論状態     [B, L, D]
z:  潜在推論     [B, L, D]

【再帰方程式】
──────────────────────────────────────────

z_{t+1} = TinyNet_xyz(x, y_t, z_t)
y_{t+1} = TinyNet_yz(y_t, z_{t+1})

これをn回繰り返して（latent_recursion）:

(y_n, z_n) = latent_recursion(x, y_0, z_0, n)

さらにT回の「Deep Supervisionブロック」で（deep_recursion):

for i in range(T):
    (y_n^i, z_n^i) = latent_recursion(x, y_n^{i-1}, z_n^{i-1}, n)
    logits_i = OutputHead(y_n^i)
    loss_i = CE(logits_i, targets)


最終損失:

L = (1/T) Σ_i loss_i + w * (1/T) Σ_i BCE(q_hat_i, correctness_i)


【目的】
──────────────────────────────────────────
x を固定しながら、y と z を段階的に改善することで、
最終的に「正確な次トークン予測」を実現する

（迷路では「正確な解」、言語では「正確な確率分布」）
"""

