# /plan — example responses

## Example 1: Understand — layer 4

**User:** `/plan explain SAE on layer 4`

**Agent (truncated):**

```markdown
## Concept: SAE on Pythia-70M

### What we're interpreting
We train a sparse autoencoder on the **residual stream after transformer block 4** of
Pythia-70M. Each token position produces a 512-dimensional vector (d_model). Those
vectors are what the LM "knows" at that depth; the SAE rewrites each vector as a sparse
combination of interpretable features.

Hook: `blocks.4.hook_resid_post` (TransformerLens) or `hidden_states[5]` (HuggingFace).

### Architecture (d_in, d_sae, sparsity)
- d_in = 512 (fixed for pythia-70m)
- d_sae = 4096 typical (8× expansion)
- Encoder: Linear(512 → 4096) + ReLU; Decoder: Linear(4096 → 512)
- Loss: MSE(reconstruction) + λ · L1(sparse activations)

### Training data flow
1. Sample text → tokenize with Pythia tokenizer
2. Forward pass → save residual vectors at layer 4 (activation cache)
3. Train SAE only on cached vectors (fast, repeatable)
4. No need to backprop through the full LM during SAE training

### How to read a trained feature
Pick feature index i. Find token contexts where encoder activation f[i] is largest
(max-activating examples). Patterns (punctuation, names, code-like spans) suggest what
the feature detects.

### Pitfalls
- HF hidden_states index 5 = after layer 4, not index 4
- Use the same Pythia checkpoint step (e.g. step143000) for cache and eval
- Start with a small text shard before scaling cache size
```

---

## Example 2: Build — SAELens path

**User:** `/plan build with SAELens on layer 4`

**Agent (truncated):**

```markdown
/plan decisions:
- [x] Goal: scaffold repo
- [x] Stack: SAELens
- [x] Hook layer: 4  Position: all tokens
- [x] Pythia checkpoint step: step143000
- [x] d_sae: 4096  Sparsity: L1
- [ ] Hardware: (confirm GPU VRAM and disk for cache)

## Build plan — Path A (SAELens)

Build progress:
- [ ] Phase 0: README.md, requirements.txt, .gitignore
- [ ] Phase 1: configs/pythia70m_layer4.yaml
- [ ] Phase 2: scripts/cache_activations.py (or SAELens cache runner)
- [ ] Phase 3: scripts/train_sae.py (SAELens training entry)
- [ ] Phase 4: scripts/eval_sae.py
- [ ] Phase 5: notebooks/01_explore_features.ipynb

### Config snippet (configs/pythia70m_layer4.yaml)
model_id: EleutherAI/pythia-70m
pythia_step: step143000
layer: 4
hook: blocks.4.hook_resid_post
d_in: 512
d_sae: 4096
l1_coeff: 0.001
batch_size: 4096
train_steps: 50000
max_tokens_cache: 1000000   # smoke test; increase later

### Commands (after scaffold)
pip install -r requirements.txt
python scripts/cache_activations.py --config configs/pythia70m_layer4.yaml
python scripts/train_sae.py --config configs/pythia70m_layer4.yaml
python scripts/eval_sae.py --checkpoint runs/sae/latest

Say when to scaffold files in the repo.
```

---

## Example 3: Build — minimal path

**User:** `/plan build minimal SAE to learn how it works`

**Agent (truncated):**

```markdown
/plan decisions:
- [x] Goal: scaffold repo
- [x] Stack: minimal custom
- [x] Hook layer: 4
- [x] d_sae: 4096  Sparsity: L1

## Build plan — Path B (minimal)

Deliverables:
- src/sae.py — SparseAutoencoder + MSE/L1 loss
- scripts/cache_activations.py — HF hook on gpt_neox.layers[4]
- scripts/train_sae.py — train on .pt shards [N, 512]
- scripts/eval_sae.py — MSE, L0, dead features
- notebooks/01_explore_features.ipynb

Do not import sae-lens in Path B scripts. See reference.md for HF hidden_states[5] mapping.
```
