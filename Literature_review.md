# Literature Review: Semantic ID Recommendation Systems

---

## TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations

### **Core Contribution**

- Converts collaborative-filtering (GNN) user/item embeddings into discrete token sequences via masked vector quantization, so an LLM can generate/retrieve recommendations in token space.

**Solution Approach:**
```
User/Item Interactions → GNN (CF) → Representations
                                          ↓
                          Masked Vector Quantization (MQ-Tokenizer)
                                          ↓
                            Discrete Tokens (compatible with LLMs)
                                          ↓
                            LLM Generative Retrieval
```

### **Strengths** ✅

• Brings CF signal into an LLM-compatible representation (tokens instead of raw IDs).
• Supports serving-friendly retrieval using a precomputed item token/pool rather than full-catalog scoring.
• Can accommodate new items by updating the item pool (less end-to-end retraining than many generative setups).
• Masking improves robustness/generalization of tokenization.

### **Limitations** ⚠️

• Codebook management risks (collapse/under-utilization) and quantization loss vs dense embeddings.
• Still depends on CF/GNN training; cold-start is not fully addressed.
• Additional complexity from multi-codebook token design and maintenance.

---

## TIGER: Transformer Index for Generative Recommenders

### **Core Contribution**

- Predicts top-K Semantic ID prefixes in parallel (no autoregressive decoding) and retrieves candidates via a prefix → items index lookup, enabling fast candidate generation at scale.

**Solution Approach:**
```
Item Metadata/Signals → RQ-VAE → Semantic ID codes [c0,c1,c2,...]
                                    ↓
User History (Semantic IDs) → Transformer Encoder → Top-K Prefixes [c0..cP]
                                    ↓
                         Prefix Index Lookup (prefix → item bucket)
                                    ↓
                              Candidate Items → (Downstream Ranker)
```

### **Strengths** ✅

• Serving-friendly: parallel prefix prediction avoids slow autoregressive generation.
• Scales cleanly: prefix→bucket lookup is index-driven and predictable at large catalogs.
• Controllable candidate generation: tune K prefixes and prefix length P to manage recall vs cost.

### **Limitations** ⚠️

• Prefix granularity tradeoff: short prefixes create huge buckets; long prefixes make routing brittle.
• Router dependence: wrong prefix predictions can drop relevant items; often needs fallback retrieval (e.g., ANN union).
• Buckets are unranked: requires a downstream ranker/reranker to order candidates.
• Index/storage overhead: maintaining prefix tables and refresh cadence adds operational complexity.

---

## LIGER: Unifying Generative and Dense Retrieval

### **Core Contribution**

- Combines dense retrieval (embedding-based) and generative retrieval (semantic ID-based) into a unified model with shared encoder and dual prediction heads, enabling flexible trade-offs between accuracy and efficiency through multi-task learning.

**Solution Approach:**
```
Input: User Sequence
  ↓
Bidirectional Transformer Encoder
  ↓
Two Parallel Outputs:
  ├─→ Dense: Next Item Embedding (similarity-based)
  └─→ Generative: Next Item Semantic ID (token generation)
  
Training: Multi-task loss
L_total = L_dense + λ × L_generative
```

### **Strengths** ✅

• Best of both worlds: combines generative + dense to mitigate performance differences and enhance cold-start.
• Improved cold-start: text representations help by integrating content into sequential model.
• Flexible inference: can use either channel or both based on use case (speed vs accuracy).
• Better coverage: union of both retrieval methods demonstrated in experiments.
• Theoretical foundation: formally analyzes trade-offs between paradigms.

### **Limitations** ⚠️

• Increased complexity: two parallel models increase training cost.
• Hyperparameter tuning: need to balance λ between objectives, difficult to optimize.
• Storage overhead: requires maintaining both embedding indices and semantic ID mappings.
• Training time: multi-task learning leads to longer convergence.

---

## DPO: Direct Preference Optimization

### **Core Contribution**

- Simplifies preference learning by directly optimizing policy to satisfy preferences without explicit reward model, eliminating instability and complexity of RLHF while achieving comparable or better results.

**Solution Approach:**
```
Preference Data: (chosen, rejected) pairs
                    ↓
        DPO Loss: -log(σ(β × (score_chosen - score_rejected)))
                    ↓
        Direct Policy Update (no reward model)
                    ↓
        Policy satisfies preferences implicitly
```

### **Strengths** ✅

• Simpler than RLHF (no reward model, no RL optimization loop).
• More stable training (avoids reward model collapse and RL instabilities).
• Memory efficient (single model instead of policy + reward + reference).
• Works with any differentiable scoring function.

### **Limitations** ⚠️

• Unbounded corrections can destroy base model rankings if not carefully designed.
• Quality depends heavily on preference data distribution.
• Less flexible than RLHF for complex reward shaping.
• Requires careful hyperparameter tuning (β, learning rate, regularization).

---

## ActionPiece: Context-Aware Tokenization

### **Core Contribution**

- Treats each action as an unordered set of features (rather than single token) and learns context-aware vocabulary through feature co-occurrence merging with Set Permutation Regularization (SPR), making the same action have different representations in different contexts.

**Solution Approach:**
```
Action → Unordered Feature Set {item_id, category, brand, price}
                    ↓
        Within-set & Adjacent-set Co-occurrence Analysis
                    ↓
        Merge Frequent Patterns → New Compound Tokens
                    ↓
        Set Permutation Regularization (SPR)
                    ↓
        Context-Aware Tokenization
```

### **Strengths** ✅

• Context-aware: same action gets different tokens in different contexts, enabling better understanding.
• Order-invariant: SPR handles permutations, creating robust representations regardless of feature ordering.
• Flexible vocabulary: merges based on co-occurrence patterns, adaptive to data distribution.
• Hierarchical: multi-level feature grouping produces rich, compositional representations.

### **Limitations** ⚠️

• Computational cost: K permutations per sequence multiply training time by K.
• Merge strategy: heuristic-based approach may miss important feature interaction patterns.
• Vocabulary explosion: many possible feature combinations lead to large vocab size.
• Complexity: harder to implement and maintain, increasing engineering overhead.

---

## RQ-VAE: Residual Quantization

### **Core Contribution**

- Multi-stage residual quantization that progressively refines discrete codes at each level, preventing codebook collapse and creating natural hierarchical structure from coarse to fine-grained representations.

**Solution Approach:**
```
Input Embedding → Encoder → z₀
                              ↓
                    Quantize Level 1 → q₀ (coarse)
                              ↓
                    residual₁ = z₀ - q₀
                              ↓
                    Quantize Level 2 → q₁ (medium)
                              ↓
                    residual₂ = residual₁ - q₁
                              ↓
                    Quantize Level m → qₘ (fine)
                              ↓
                    Reconstruction: q₀ + q₁ + ... + qₘ
```

### **Strengths** ✅

• Prevents codebook collapse through residual structure (each level must contribute).
• Natural hierarchical organization (coarse-to-fine) without explicit supervision.
• Compact representation with controllable granularity (add/remove levels).
• Enables prefix matching at any hierarchy level.

### **Limitations** ⚠️

• Reconstruction quality depends on number of levels (more levels = more complexity).
• Each level adds computational cost during encoding/decoding.
• Codebook initialization still matters (random init can be suboptimal).
• Hyperparameter tuning for number of levels and codebook sizes.

---

## LightGCN: Simplified Graph Collaborative Filtering

### **Core Contribution**

- Simplifies graph neural networks for collaborative filtering by removing feature transformation and nonlinear activation, keeping only neighborhood aggregation, achieving better performance with less complexity.

**Solution Approach:**
```
User-Item Interaction Graph
            ↓
    Neighborhood Aggregation (no transformation)
            ↓
    Layer-wise Propagation:
    e⁽ᵏ⁺¹⁾ᵤ = Σ (1/√(|Nᵤ||Nᵢ|)) × e⁽ᵏ⁾ᵢ
            ↓
    Weighted Sum Across Layers:
    eᵤ = Σ αₖ × e⁽ᵏ⁾ᵤ
            ↓
    Final User/Item Embeddings
```

### **Strengths** ✅

• Simpler architecture with fewer parameters (faster training).
• Better performance than complex GNN variants on recommendation tasks.
• Captures multi-hop collaborative signals through layer stacking.
• Easy to implement and understand.

### **Limitations** ⚠️

• Still requires interaction data (cold-start problem remains).
• Scalability challenges for very large graphs.
• Fixed aggregation scheme (no learned attention).
• Cannot incorporate rich item/user features easily.

---

## Comparison

| Paper | Paradigm | Main Contribution | Key Limitation |
|-------|----------|-------------------|----------------|
| **TokenRec** | Generative | CF → LLM tokens | Codebook collapse |
| **TIGER** | Generative | Prefix routing | Coverage gaps (15%) |
| **LIGER** | Hybrid | Dense + generative | Training complexity |
| **DPO** | Preference | Direct optimization | Unbounded corrections |
| **ActionPiece** | Tokenization | Context-aware vocab | Computational cost |
| **RQ-VAE** | Representation | Residual quantization | Multi-level overhead |
| **LightGCN** | Collaborative | Simplified GNN | Cold-start remains |

---

## Our Integration

| What We Adopted | What We Innovated | Result |
|----------------|-------------------|--------|
| RQ-VAE quantization | K-means initialization | 92.4% vs 30% utilization |
| TIGER prefix routing | Hybrid retrieval (prefix ∪ ANN) | 96.5% vs 76%/88% coverage |
| DPO framework | Bounded corrections (α=0.05) | +31% vs -59% unbounded |
| ActionPiece SPR | Simplified K=2 | Efficient training |
| LightGCN CF | Multi-modal (CLIP+T5+LightGCN) | Cold-start support |

All metrics from `run_summary.json` and `dpo_history.json`.
