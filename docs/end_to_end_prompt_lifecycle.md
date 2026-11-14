# The End-to-End LLM Prompt Lifecycle & Transformer Architecture

> **Purpose:** Comprehensive deep-dive walkthrough of what physically and mathematically happens from the millisecond a user presses **Enter** in a chatbot UI to the moment tokens stream back to the screen.

---

## 1. High-Level Lifecycle Architecture

```
User types: "What is 2+2?"
        │
        ▼
[ STEP 1: TOKENIZATION ] ──► String converted to Token IDs: [2065, 374, 220, 17, 28, 30]
        │
        ▼
[ STEP 2: EMBEDDING LOOKUP ] ──► IDs converted to vectors: [1, 6, d_model]
        │
        ▼
[ STEP 3: PREFILL PHASE (The Prompt) ] ──► Compute-Bound parallel forward pass
        │  ├── All 6 tokens processed simultaneously across all 32 layers
        │  ├── K and V vectors computed and saved to KV Cache (or Paged Blocks)
        │  └── Logits computed for the last token position
        │
        ▼
[ STEP 4: SAMPLING FIRST TOKEN ] ──► Argmax / Top-p selects: "4" (Token ID 19)
        │
        ▼
┌───► [ STEP 5: DECODE LOOP (Autoregressive) ] ──► Memory-Bound sequential pass
│       ├── Input is ONLY the 1 new token ("4") [1, 1, d_model]
│       ├── RoPE rotated at offset = current_cache_len
│       ├── Compute Q, K, V for the 1 new token
│       ├── Append K, V to KV Cache
│       ├── Q attends to ALL past cached keys [1 x kv_len]
│       ├── Predict next token: "." (Token ID 13)
│       └── Loop until <eos> or max_tokens reached
│
└───────┘
        │
        ▼
[ STEP 6: DETOKENIZATION & STREAMING ] ──► Token IDs mapped back to text and pushed over WebSocket/SSE
```

---

## 2. Step-by-Step Deep Dive

### Step 1: Tokenization (Text $\to$ Integers)
Computers and neural networks cannot read characters or words directly. 
* **The Tokenizer (e.g. Byte-Pair Encoding / BPE):** Chops text into sub-word tokens based on a fixed vocabulary (e.g., $128{,}256$ tokens for Llama-3).
* **Input String:** `"Explain AI"`
* **Token IDs:** `[32541, 9552]` (A 1D array of integers of shape `[B, S]`, where $B=1$ and $S=2$).

---

### Step 2: Embedding Table Lookup ($[B, S] \to [B, S, d_{\text{model}}]$)
* The model contains a weight matrix: `W_embed` of shape `[vocab_size, d_model]`.  
  *(For Llama-3-8B: $128{,}256 \times 4096$).*
* For each token ID, the model retrieves that integer's corresponding row from `W_embed`.
* **Important:** This is a **gather** operation (indexing), not a matrix multiplication!
* **Output Shape:** Tensor `x` of shape `[1, S, 4096]`.

---

### Step 3: Inside a Decoder Block (Repeated $L$ Times)

Every transformer layer takes an input tensor of shape `[B, S, d_{\text{model}}]` and outputs an identical shape `[B, S, d_{\text{model}}]`.

```
          x  [B, S, d_model]
          │
          ├───► RMSNorm ──► Attention Sublayer ───┐
          │                                        │
          └────────────────────────────────────────┴──► (+) Residual [B, S, d_model]
          │
          ├───► RMSNorm ──► SwiGLU FFN Sublayer ──┐
          │                                        │
          └────────────────────────────────────────┴──► (+) Residual [B, S, d_model]
```

#### 3.1. Pre-Normalization (RMSNorm)
Instead of standard LayerNorm, modern LLMs use **Root Mean Square Normalization (RMSNorm)** because it avoids calculating the mean, saving memory bandwidth:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

#### 3.2. Q, K, V Projections
The normalized vector is multiplied by three weight matrices to create Queries, Keys, and Values:
* $Q = x \cdot W_q \quad [B, S, n_{\text{heads}} \times \text{head\_dim}]$
* $K = x \cdot W_k \quad [B, S, n_{\text{kv\_heads}} \times \text{head\_dim}]$
* $V = x \cdot W_v \quad [B, S, n_{\text{kv\_heads}} \times \text{head\_dim}]$

*In Llama-3-8B:*
* $Q$ has $32$ heads $\times 128 = 4096$ width.
* $K$ and $V$ have only **$8$ heads** $\times 128 = \mathbf{1024}$ width (**GQA 4:1 ratio**).

#### 3.3. RoPE (Rotary Position Embedding)
Attention does not inherently know word order. RoPE rotates each 2D pair in the $Q$ and $K$ vectors by an angle proportional to the token's position $m$:
$$\text{RoPE}(x, m) = x \cos(m\theta) + \text{rotate\_half}(x) \sin(m\theta)$$
* *Crucial rule:* RoPE is applied **only to Q and K**, never to V!

#### 3.4. Causal Grouped-Query Attention (GQA)
1. **Head Expansion (`repeat_kv`):** The 8 KV heads are expanded $4\times$ to match the 32 Query heads.
2. **Attention Scores:** Compute dot products between Queries and all Keys:
   $$\text{Scores} = \frac{Q \cdot K^T}{\sqrt{\text{head\_dim}}} \quad [B, n_{\text{heads}}, S, S]$$
3. **Causal Masking:** Future tokens are masked out with $-\infty$ so words cannot peek ahead:
   $$\text{Masked Scores}_{i, j} = -\infty \quad \text{for } j > i$$
4. **Softmax:** Turns scores into attention probabilities:
   $$P = \text{Softmax}(\text{Masked Scores}) \quad [B, n_{\text{heads}}, S, S]$$
5. **Weighted Value Sum:**
   $$\text{Attention Out} = P \cdot V \quad [B, n_{\text{heads}}, S, \text{head\_dim}]$$
6. **Out Projection ($W_o$):** Re-merges heads back into a single vector of width $d_{\text{model}}$ ($32 \times 128 = 4096$).
7. **First Residual Connection:** $x = x + \text{Attention Out}$.

#### 3.5. SwiGLU Feed-Forward Network (FFN)
The FFN expands the vector from $d_{\text{model}} (4096) \to d_{\text{ff}} (14336)$, does non-linear feature transformation, and compresses back to $4096$:
$$\text{FFN}(x) = \left(\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})\right) \cdot W_{\text{down}}$$
* **Second Residual Connection:** $x = x + \text{FFN Out}$.

---

### Step 4: Final Norm & LM Head (Picking the Next Word)

After passing through all 32 layers:
1. **Final RMSNorm:** Normalizes the final hidden state.
2. **The LM Head:** A linear layer multiplying the hidden vector by `W_embed^T` (tied embedding matrix) to project from $d_{\text{model}} (4096) \to \text{vocab\_size} (128256)$.
   $$\text{Logits} = x_{[-1]} \cdot W_{\text{embed}}^T \quad [1, \text{vocab\_size}]$$
   * *Note:* We only compute the LM head for the **last position** $[-1]$!
3. **Sampling:**
   * **Greedy:** $\text{argmax}(\text{logits})$ $\implies$ pick the highest probability word.
   * **Temperature / Top-p:** Softens the distribution and randomly samples for creativity.

---

### Step 5: The Autoregressive Decode Loop (Where KV Cache Shines)

Now that Word 1 is emitted, how do we generate Word 2?

```
Naive Way (No Cache):
Feed [Prompt + Word 1] (S + 1 tokens) ──► Recomputes all layers ──► O(N^2) waste!

Cached Way (nanoinfer):
Feed ONLY [Word 1] (1 token!) ──► Reads past K,V from Cache ──► O(1) step!
```

#### Inside a Cached Decode Step:
1. Input is **only 1 token**: shape `[1, 1, 4096]`.
2. **RoPE Offset:** We look at the cache bookmark: `offset = cache.get_seq_len()`. The new token is rotated with angle for `position = offset` (not 0!).
3. **Project Q, K, V:** Computes $Q, K, V$ for just that 1 token.
4. **Append to Cache:** Writes the new $K, V$ to the cache. The cache now holds `kv_len = offset + 1` tokens.
5. **Attention:**
   $$Q [1, 32, 1, 128] \times K_{\text{cached}}^T [1, 32, 128, \text{kv\_len}] \implies \text{Scores } [1, 32, \mathbf{1}, \text{kv\_len}]$$
   * *Notice the shape:* It is **not** a square matrix! It is just **one single row of probabilities**!
6. **Emit next token:** Next token is chosen and the loop repeats until the model outputs the `<|end_of_text|>` token.

---

### Step 6: Detokenization
The engine takes the sequence of emitted token IDs, passes them through the tokenizer's decoder table, transforms the numbers back into human-readable Unicode text, and streams the characters over an HTTP Server-Sent Events (SSE) or WebSocket stream to the user's browser.

---

## 3. Real Worked Tensor Shapes: Llama-3-8B

| Stage | Input Tensor | Layer / Operation | Output Tensor | Key Note |
|---|---|---|---|---|
| **Tokenize** | `"Explain AI"` | BPE Tokenizer | `[1, 2]` | Two integer token IDs |
| **Embed** | `[1, 2]` | `nn.Embedding(128256, 4096)` | `[1, 2, 4096]` | $S=2, d_{\text{model}}=4096$ |
| **QKV Proj** | `[1, 2, 4096]` | Linear $W_q, W_k, W_v$ | $Q: [1, 32, 2, 128]$<br>$K,V: [1, 8, 2, 128]$ | GQA: 8 KV heads vs 32 Q heads |
| **Attention** | $Q, K, V$ | $Q \cdot K^T / \sqrt{d}$ | `[1, 32, 2, 2]` | Small $2 \times 2$ causal score matrix |
| **SwiGLU** | `[1, 2, 4096]` | Gate, Up, Down | `[1, 2, 4096]` | Expands to $d_{\text{ff}}=14336$ and compresses |
| **LM Head** | `[1, 1, 4096]` (last pos) | Linear $(4096 \to 128256)$ | `[1, 128256]` | Probabilities over entire vocabulary |
| **Decode Step** | `[1, 1, 4096]` (1 new token)| Attention against cache | `[1, 32, 1, kv_len]` | Single row attention score! |

---

## 4. Hardware States Throughout the Lifecycle

| Phase | Bound By | GPU Activity | Bottleneck / Solution |
|---|---|---|---|
| **Prefill (Prompt)** | **Compute-Bound** | Tensor Cores running near 100% | High arithmetic intensity ($\approx S\text{ FLOP/byte}$). FlashAttention prevents $S \times S$ memory blowup. |
| **Decode (Generation)**| **Memory-Bound** | Memory bus saturated; cores 99% idle | Low arithmetic intensity ($1.0\text{ FLOP/byte}$). Batching and Quantization attack this. |
| **KV Storage** | **Capacity-Bound** | VRAM capacity holding past tokens | Grows with sequence length and concurrency. PagedAttention eliminates the 92% contiguous waste. |
