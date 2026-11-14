# LLM Inference Engineering — Comprehensive Interview Revision Guide

> **Purpose:** Rapid-revision reference for inference engineering interviews.  
> **Key Philosophy:** "Say exact numbers, explain the physical bottleneck first, and state the targeted optimization."

---

## 1. Hardware Fundamentals & The Roofline Model (Milestone 1)

### Mental Model: The Chef & The Conveyor Belt
* **The Compute Engine (The Chef):** Measured in **TFLOP/s** (Trillion Floating-Point Operations per Second).  
  * *NVIDIA H100 SXM:* **989 TFLOP/s** (dense fp16).
* **The Memory Bus (The Conveyor Belt):** Measured in **GB/s or TB/s** (Memory Bandwidth).  
  * *NVIDIA H100 SXM:* **3,350 GB/s (3.35 TB/s)**.
* **The Golden Rule:** A compute core cannot calculate with a weight until that weight travels across the conveyor belt from High-Bandwidth Memory (HBM) into the compute register.

---

### The Ridge Point: Hardware's Natural Balance
$$\text{Ridge Point} = \frac{\text{Peak Compute (FLOP/s)}}{\text{Memory Bandwidth (Bytes/s)}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx \mathbf{295\text{ FLOP/byte}}$$

* **Arithmetic Intensity ($\text{FLOP/byte}$):** Operations performed per byte of data loaded.
* **If Intensity $< 295$:** **Memory-Bound** (The Chef is starving, waiting on the belt).
* **If Intensity $\ge 295$:** **Compute-Bound** (The Chef is sweating, cores are 100% saturated).

---

### The 1.0 FLOP/Byte Truth (Why Single-Token Decode is Slow)
To generate **1 token** at **batch size 1**:
1. **Math (FLOPs):** Every parameter performs 1 multiply and 1 add:
   $$\text{FLOPs} = 2 \times P$$
2. **Memory (Bytes):** Every parameter is stored in 16-bit precision (2 bytes):
   $$\text{Bytes} = 2 \times P$$
3. **Arithmetic Intensity:**
   $$\text{Intensity} = \frac{2P\text{ FLOPs}}{2P\text{ Bytes}} = \mathbf{1.0\text{ FLOP/byte}}$$

#### The Interview Punchline:
> *"At batch size 1, decode operates at an arithmetic intensity of exactly 1.0 FLOP/byte. Because modern accelerators have ridge points between 150 and 300 FLOP/byte, the tensor cores run at ~0.3% utilization—sitting idle over 99% of the time waiting for weights to stream across the memory bus."*

---

### Prefill vs. Decode Comparison

| Property | Prefill Phase (Prompt) | Decode Phase (Token Generation) |
|---|---|---|
| **What it does** | Ingests the entire prompt ($S$ tokens) | Generates 1 token at a time autoregressively |
| **Token Dimensions** | $q\_len = S,\ kv\_len = S$ | $q\_len = 1,\ kv\_len = cur + 1$ |
| **Weight Reuse** | Weights loaded once, multiplied by $S$ tokens | Weights loaded once, multiplied by 1 token |
| **Arithmetic Intensity** | $\approx S\text{ FLOP/byte}$ (e.g. 512 FLOP/byte at $S=512$) | $\approx 1.0\text{ FLOP/byte}$ (at batch 1) |
| **Limiting Regime** | **Compute-Bound** (above prompt ridge threshold) | **Memory-Bound** (bandwidth-bound) |
| **Hardware State** | Tensor cores fully saturated | Memory bus saturated; compute cores idle |

---

### Why Batching is the Primary Lever for Decode
* Since future tokens cannot be computed in parallel (autoregressive dependency), we batch **different users** together ($B$ users).
* Weights ($2P$ bytes) are loaded **once from HBM** and multiplied against $B$ independent token vectors simultaneously:
  $$\text{FLOPs} = 2 \times P \times B \implies \text{Intensity} = \frac{2PB}{2P} = \mathbf{B\text{ FLOP/byte}}$$
* Batching scales intensity linearly from $1.0 \to B\text{ FLOP/byte}$, moving decode closer to the hardware ridge point.

---

## 2. Transformer Anatomy & The KV Cache (Milestone 2)

### Single-Layer Forward Pass Data Flow
```
x [B, q_len, d_model]
  │
  ├──► RMSNorm ──► Linear W_q, W_k, W_v ──► RoPE (Q & K only) ──► GQA Expansion
  │                                                                     │
  │                                   Attention: Softmax(Q @ K^T / √d) @ V
  │                                                                     │
  ├──► (+) Residual <── Linear W_o <───────────────────────────────────┘
  │
  ├──► RMSNorm ──► SwiGLU FFN: (SiLU(W_gate @ x) * (W_up @ x)) @ W_down
  │
  └──► (+) Residual ──► Output x [B, q_len, d_model]
```

---

### The KV Cache Motivation: $O(N^2) \to O(N)$
* **Naive Decode:** To generate token $N$, the model recalculates Keys and Values for all past $N-1$ tokens from scratch.
  $$\text{Total Work} = \sum_{i=1}^N i \approx \mathbf{\frac{N^2}{2} \text{ forward steps (Quadratic)}}$$
* **Cached Decode:** Deterministic past Keys and Values are saved in VRAM. Each step only computes $Q, K, V$ for the **1 new token** and appends to the cache.
  $$\text{Per-step Work} = O(1) \implies \mathbf{\text{Total Work} = O(N) \text{ (Linear)}}$$

---

### Sizing the KV Cache: The Exact Formula
$$\mathbf{\text{Bytes per Token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times \text{head\_dim} \times \text{dtype\_bytes}}$$

* Factor of **$2$**: Stores both **Key** and **Value** vectors.
* **$n_{\text{layers}}$**: Each layer maintains its own independent KV state.
* **$n_{\text{kv\_heads}}$**: Number of KV heads (reflects GQA).
* **$\text{head\_dim}$**: Vector dimension per head ($d_{\text{model}} / n_{\text{heads}}$).
* **$\text{dtype\_bytes}$**: 2 bytes for fp16/bf16, 1 byte for int8, 0.5 byte for int4.

#### Worked Example: Llama-3-8B (fp16)
* $n_{\text{layers}} = 32$, $n_{\text{kv\_heads}} = 8$, $\text{head\_dim} = 128$, $\text{dtype\_bytes} = 2$
$$\text{Memory/token} = 2 \times 32 \times 8 \times 128 \times 2 = \mathbf{131{,}072\text{ bytes}} = \mathbf{128\text{ KiB / token}}$$

#### Context Scale:
* **1 User at 8K Context:** $8{,}192 \times 128\text{ KiB} = \mathbf{1\text{ GiB}}$
* **64 Users at 8K Context:** $64 \times 1\text{ GiB} = \mathbf{64\text{ GiB}}$  
  *(The KV cache rapidly outgrows the 16 GB model weights!).*

---

### Grouped-Query Attention (GQA)
* **Standard MHA (Multi-Head Attention):** $n_{\text{kv\_heads}} = n_{\text{heads}} = 32$.
* **GQA (Llama-3):** $n_{\text{kv\_heads}} = 8$, $n_{\text{heads}} = 32$ ($4:1$ ratio).
* **Why it matters:** 4 Query heads share 1 KV head. Slashes KV cache memory by **$4\times$** with negligible loss in model accuracy.

---

### The 3 Classic KV Cache Implementation Traps

| Gotcha | The Bug | Failure Mode | The Solution |
|---|---|---|---|
| **1. Cursor Scope** | Using a single shared integer for all layers | Layer 1 writes into slots $[S:2S]$ instead of $[0:S]$, silently corrupting representations | Maintain **per-layer cursors** (`cur_len = [0] * n_layers`) |
| **2. Buffer Slicing** | Returning the full `max_seq_len` buffer to attention | Attention calculates softmax over uninitialized memory slots; output is silent garbage | Slice sequence axis strictly to live cursor: `k[layer, :, :, :cur, :]` |
| **3. RoPE Position** | Applying RoPE starting at position 0 during decode | New token loses relative distance to past tokens | Offset RoPE by `start = cache.get_seq_len()` computed **once before the layer loop** |

---

## 3. FlashAttention: Eliminating the Quadratic Memory Wall

### The Problem: The Temporary $[B, H, S, S]$ Scratchpad
Standard attention calculates:
1. $S = \frac{Q K^T}{\sqrt{d_k}} \quad [B, H, S, S]$ $\implies$ **Written to slow HBM**
2. $P = \text{Softmax}(S) \quad [B, H, S, S]$ $\implies$ **Read from HBM & written back to HBM**
3. $O = P V \quad [B, H, S, d_k]$ $\implies$ **Read from HBM**

* At $S = 2048$ for Llama-3-8B: Materializing this matrix costs **1 GB in fp32 per layer**!
* At $S = 8192$: Memory explodes to **16 GB per layer**!
* Writing and reading this giant matrix 3 times across the memory bus chokes the GPU.

---

### The GPU Memory Hierarchy
* **HBM (VRAM):** Large ($80\text{ GB}$), relatively slow bandwidth ($3.35\text{ TB/s}$).
* **SRAM (On-Chip Cache):** Tiny ($~228\text{ MB}$), ultra-fast bandwidth ($~33\text{ TB/s}$, $10\times$ faster).

---

### The FlashAttention Solution: Tiling + Online Softmax
1. **Never materialize the $[S \times S]$ matrix in HBM:** The matrix is computed in small tiles ($64 \times 64$) inside ultra-fast **SRAM**.
2. **Online Softmax Rescaling:** Softmax requires the full row sum $\sum e^{x_i}$. FlashAttention maintains running maximums and running normalizers, rescaling intermediate outputs dynamically as new tiles are processed:
   $$\text{Output}_{\text{new}} = \text{Output}_{\text{old}} \times \left(\frac{\text{Sum}_{\text{old}}}{\text{Sum}_{\text{new}}}\right) + \text{Tile}_{\text{new}}$$
3. **Result:** HBM memory footprint drops from **$O(S^2) \to O(S)$**, and wall-clock execution accelerates by **$2\text{–}4\times$** purely by eliminating HBM memory roundtrips.

---

## 4. PagedAttention & Block Tables (Milestone 3)

### Why Contiguous Memory Fails (The 92% Waste Problem)
* **The Unknown Future:** When a request arrives, you cannot know if the user will produce 10 tokens or 2,000 tokens.
* **Contiguous Allocator Penalty:** Must preallocate `max_context` (e.g., 2,048 tokens = $268\text{ MB}$ per sequence in Llama-3-8B).
* **The Reality:** The average response is ~150–250 tokens. Over **90% of allocated VRAM sits as blank reservation waste**.

---

### The PagedAttention Architecture
* **Physical Block Pool:** GPU VRAM is pre-partitioned into fixed-size blocks (typically **16 tokens** per block).
* **Decoupled Tensor Shape:**
  $$\text{Old (Contiguous): } [n_{\text{layers}}, \mathbf{batch}, n_{\text{kv\_heads}}, \mathbf{max\_seq\_len}, \text{head\_dim}]$$
  $$\text{New (Paged): } [\mathbf{num\_blocks}, n_{\text{layers}}, n_{\text{kv\_heads}}, \mathbf{block\_size}, \text{head\_dim}]$$
* **Block Table (Address Translation):** Each sequence has a routing list mapping logical tokens to physical blocks:
  $$\text{Logical Block Index} = \text{pos} // \text{block\_size}$$
  $$\text{Block Slot Offset} = \text{pos} \% \text{block\_size}$$

---

### Key Capabilities of PagedAttention

1. **Near-Zero Fragmentation:** Internal fragmentation is bounded to the final unfilled block ($< 16$ tokens). Total cache waste drops from **$92.3\% \to 4.5\%$** (a **$12.5\times$ footprint reduction**).
2. **12x Concurrency Boost:** On an 80 GB GPU, concurrent active users scale from **298 users** (contiguous) to **3,699 users** (paged).
3. **Copy-on-Write (Prefix Caching):**
   * Multiple requests with identical system prompts share the exact same physical block IDs.
   * `PhysicalBlock.ref_count` increments ($1 \to 2$).
   * Zero memory duplication. When sequences diverge during generation, a private block is allocated on demand.

---

### The Real-World 3-Tier Serving Lifecycle

```
┌────────────────────────────────────────────────────────┐
│ Tier 1: GPU VRAM (Hot)   ──► Active decoding (HBM)     │
├────────────────────────────────────────────────────────┤
│ Tier 2: Host CPU RAM (Warm)──► Swapped inactive blocks │
├────────────────────────────────────────────────────────┤
│ Tier 3: Database (Cold)  ──► Raw conversation text     │
└────────────────────────────────────────────────────────┘
```
1. **Active Chat:** Tokens live in GPU VRAM blocks.
2. **Pause / Memory Pressure:** LRU blocks are swapped over PCIe to Host CPU RAM.
3. **Disconnect / Log Off:** `BlockManager` decrements `ref_count` to 0 and reclaims blocks to the free pool. Text is stored in the database.
4. **Next Session:** Database text is fetched; fast compute-bound **prefill** regenerates the KV cache in $< 50\text{ ms}$.

---

## 5. Key Metrics to State Cold from Memory

| Metric | Exact Value | Meaning & Context |
|---|---|---|
| **Decode Intensity ($B=1$)** | **$1.0\text{ FLOP/byte}$** | $\frac{2P\text{ FLOPs}}{2P\text{ Bytes}}$ in fp16. Compute cores are 99% idle. |
| **H100 Ridge Point** | **$295\text{ FLOP/byte}$** | Intensity threshold where compute overtakes memory bandwidth. |
| **Llama-3-8B KV Cache per token** | **$128\text{ KiB / token}$** | $2 \times 32\text{ layers} \times 8\text{ heads} \times 128\text{ dim} \times 2\text{ bytes}$. |
| **Llama-3-8B Cache at 8K** | **$1\text{ GiB / sequence}$** | $8{,}192\text{ tokens} \times 128\text{ KiB}$. |
| **GQA Memory Savings** | **$4\times$ reduction** | 8 KV heads vs 32 Query heads. |
| **Attention FLOPs Crossover** | **$\approx 27\text{K}$ tokens** | Attention FLOPs only exceed weight FLOPs far beyond normal context windows. |
| **Contiguous Cache Waste** | **$92.3\%$** | Due to preallocating `max_context` for short requests. |
| **PagedAttention Waste** | **$< 5\%$** | Bounded to the last block ($< 16$ tokens). |
| **Paged Concurrency Gain** | **$12.4\times$** | Scales concurrent users from 298 to 3,699 on an 80 GB GPU. |

---

## 6. Top 8 Interview Questions — Ready-to-Speak 60-Second Answers

### Q1: Why is single-token generation memory-bound?
> *"In autoregressive decoding at batch size 1, generating a single token requires streaming every weight of the model from HBM into the registers to perform two operations per parameter: a multiply and an add ($2P$ FLOPs). Because weights are stored in 16-bit precision, that requires reading $2P$ bytes. The arithmetic intensity is exactly 1.0 FLOP per byte moved. Since accelerators like the H100 have a ridge point near 300 FLOP/byte, the compute cores spend over 99% of their time idle waiting for memory transfer."*

### Q2: Why does a KV cache help, and what is its computational complexity trade-off?
> *"Without a KV cache, autoregressive generation requires re-running the entire prefix through all layers at every step, creating an $O(N^2)$ computational explosion. Because transformer layers are deterministic, earlier Keys and Values never change. Caching them reduces each decode step to $O(1)$ work, bringing total generation down to $O(N)$. The trade-off is memory: the cache grows linearly with context and concurrency, rapidly becoming the primary memory consumer on the GPU."*

### Q3: How do you calculate the memory size of a KV cache?
> *"The closed-form formula is $2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times \text{head\_dim} \times \text{dtype\_bytes}$ per token. For Llama-3-8B in fp16, that is $2 \times 32 \times 8 \times 128 \times 2 = 128\text{ KiB per token}$. For a single user at 8K context, the KV cache requires exactly 1 GiB of VRAM. At batch 64, it requires 64 GiB, which far exceeds the 16 GB required to store the model weights."*

### Q4: What is Grouped-Query Attention (GQA) and why is it used?
> *"In standard Multi-Head Attention, every query head has its own dedicated key and value head. GQA groups multiple query heads—for example, 4 query heads to 1 KV head in Llama-3-8B. Because the attention map expands back to the query dimension during matrix multiplication, the model maintains high expressive capacity while reducing the KV cache memory footprint and memory bandwidth traffic by exactly 4x."*

### Q5: What does FlashAttention actually avoid?
> *"Standard attention materializes the intermediate $[S \times S]$ attention score matrix in HBM, reads it back to compute softmax, and writes it back again to multiply by V. At long contexts, this matrix consumes gigabytes of memory per layer and saturates HBM bandwidth. FlashAttention tiles $Q, K, V$ into on-chip SRAM cache and uses Online Softmax rescaling to compute attention without ever writing the $S \times S$ matrix to HBM. It drops memory from $O(S^2)$ to $O(S)$ and delivers a 2–4x wall-clock speedup."*

### Q6: What is PagedAttention and what problem does it solve?
> *"Naive KV caches allocate contiguous VRAM up to `max_context` because request length cannot be predicted in advance. In practice, this results in over 90% memory waste from internal and external fragmentation and over-reservation. PagedAttention borrows virtual memory paging from operating systems: it partitions KV cache storage into small, fixed-size physical blocks (e.g. 16 tokens) allocated dynamically on demand via a block table. It slashes memory waste down to under 5%, enabling a 2x to 4x increase in effective serving concurrency."*

### Q7: What is Copy-on-Write in LLM serving?
> *"Copy-on-Write is a mechanism enabled by PagedAttention where multiple requests that share an identical prefix—such as a system prompt or common few-shot examples—point to the exact same physical block IDs in VRAM using a reference counter. The prompt is computed once and stored once. Only when individual user sequences generate differing output tokens does the system allocate private physical blocks, eliminating duplicated memory across users."*

### Q8: What is TTFT vs. TPOT and why is there a trade-off?
> *"TTFT is Time to First Token, dominated by the compute-bound prefill phase. TPOT is Time per Output Token, dominated by memory-bound autoregressive decoding. Increasing batch size improves GPU arithmetic intensity and total system throughput, but it degrades both TTFT and TPOT for individual users due to queuing delays and increased memory traffic per step. Production schedulers balance this trade-off using continuous batching and chunked prefill."*
