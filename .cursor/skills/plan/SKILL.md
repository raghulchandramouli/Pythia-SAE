---
name: plan
description: >-
  Plans, explains, and scaffolds sparse autoencoders on EleutherAI Pythia-70M
  residual activations. Use when the user invokes /plan, mentions Pythia-SAE,
  SAE training, activation caching, GPT-NeoX hooks, feature interpretability,
  or understanding vs building an SAE on pythia-70m.
disable-model-invocation: true
---

# /plan — Pythia-70M SAE

Helps you **build and understand** sparse autoencoders (SAEs) on activations from [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m).

## Modes

Detect from the user message; ask once if unclear.

| Mode | Action |
|------|--------|
| **Understand** | Explain concepts; no code unless asked |
| **Build** | Phased implementation plan; scaffold repo only after approval |
| **Both** | Short understand section, then build checklist |

## Pythia-70M essentials

| Item | Value |
|------|--------|
| Model id | `EleutherAI/pythia-70m` or `EleutherAI/pythia-70m-deduped` |
| `d_model` | **512** (= SAE `d_in`) |
| Layers | 6 (indices 0–5) |
| Default hook | Residual **after block 4** |
| TransformerLens | `blocks.4.hook_resid_post` |
| HF `hidden_states` | Index **5** (embedding + layers 0–4) — see [reference.md](reference.md) |

## Decision gate

Copy this checklist into the response and fill in choices before coding:

```
/plan decisions:
- [ ] Goal: understand only | scaffold repo | extend existing code
- [ ] Stack: SAELens | minimal custom | both (SAELens first, minimal second)
- [ ] Hook layer: ___  Position: all tokens | last token | prompt-only
- [ ] Pythia checkpoint step: ___  (e.g. step0, step143000)
- [ ] d_sae: ___ (default 4096)  Sparsity: L1 | TopK
- [ ] Hardware: GPU ___, activation cache budget ___
```

**Stack guidance when user wants both:**

- **SAELens + TransformerLens** — first runnable pipeline (cache → train → eval).
- **Minimal PyTorch + HuggingFace** — second path in `minimal/` or `src/sae.py` to learn internals.

**Do not mix SAELens and minimal logic in one training script.**

## Understand workflow

Explain in plain language; link to [reference.md](reference.md) for hook tables and hyperparameters.

Topics to cover:

1. **What we interpret** — residual stream at a chosen layer; each token position is a 512-d vector.
2. **SAE architecture** — encoder maps `x ∈ R^512` to sparse `f ∈ R^d_sae`; decoder reconstructs `x̂`; goal is `x̂ ≈ x` with few active features.
3. **Loss** — reconstruction (MSE) + sparsity (L1 or TopK); optional penalty for dead features.
4. **Training data flow** — text → tokenize → forward Pythia → cache activations → train SAE on cached vectors (not end-to-end through LM each step).
5. **Reading features** — high activation on specific tokens/contexts; max-activating examples in a notebook.
6. **Why 70M** — fast iteration, small cache, same GPT-NeoX family as larger Pythia models.
7. **Pitfalls** — layer index off-by-one; Pythia **pretrain step** vs SAE optimizer steps; normalization must match training stack.

### Understand output template

```markdown
## Concept: SAE on Pythia-70M

### What we're interpreting
### Architecture (d_in, d_sae, sparsity)
### Training data flow
### How to read a trained feature
### Pitfalls
```

## Build workflow

Always emit the phased plan **before** creating files. Wait for user approval to scaffold unless they explicitly say to implement now.

### Phased checklist

```
Build progress:
- [ ] Phase 0: README.md, requirements.txt, .gitignore
- [ ] Phase 1: configs/pythia70m_layer4.yaml
- [ ] Phase 2: scripts/cache_activations.py
- [ ] Phase 3: scripts/train_sae.py
- [ ] Phase 4: scripts/eval_sae.py
- [ ] Phase 5: notebooks/01_explore_features.ipynb
```

### Target repo layout

```
Pythia-SAE/
├── README.md
├── requirements.txt
├── configs/pythia70m_layer4.yaml
├── scripts/cache_activations.py
├── scripts/train_sae.py
├── scripts/eval_sae.py
├── src/sae.py                 # Path B (minimal) only
├── notebooks/01_explore_features.ipynb
└── .cursor/skills/plan/
```

Path A may use SAELens runners instead of custom `train_sae.py`; document commands in README.

### Path A — SAELens (ship fast)

**Dependencies:** `sae-lens`, `transformer-lens`, `torch`, `datasets`, `transformers`; optional `wandb`.

**Steps:**

1. Load Pythia in TransformerLens with the chosen `checkpoint_index` / revision for the Pythia step.
2. Set `hook_name` to `blocks.{L}.hook_resid_post` (default `L=4`).
3. Run SAELens activation caching (or its training runner’s cache step) on a **small** text shard first.
4. Train with `d_in=512`, `d_sae` per config (default 4096), L1 coefficient per [reference.md](reference.md).
5. Eval: reconstruction, L0, dead feature fraction; export checkpoints + config JSON.

See [reference.md](reference.md) for SAELens config keys and links.

### Path B — Minimal custom (learn internals)

**Dependencies:** `torch`, `transformers`, `datasets`, `einops`.

**Steps:**

1. `src/sae.py` — `SparseAutoencoder`: encoder, decoder (optional tied weights), `encode`, `decode`, MSE + L1.
2. `cache_activations.py` — load `AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", ...)`, register forward hook on layer `L`, save shards `[n_tokens, 512]` as `.pt` or `.npy`.
3. `train_sae.py` — Dataset over shards, Adam, log loss/L0, checkpoint `sae.pt` + config.
4. `eval_sae.py` — held-out shard metrics.
5. Notebook — top activating token windows per feature index.

Keep hooks and layer index consistent with [reference.md](reference.md) HF table.

### Config template (`configs/pythia70m_layer4.yaml`)

```yaml
model_id: EleutherAI/pythia-70m
pythia_step: step143000          # HF revision folder name
layer: 4
hook: blocks.4.hook_resid_post   # TransformerLens; see reference for HF
d_in: 512
d_sae: 4096
l1_coeff: 0.001
batch_size: 4096
train_steps: 50000
lr: 3.0e-4
corpus: openwebtext               # small shard for dev
max_tokens_cache: 10_000_000
```

### Evaluation (every build plan)

- Held-out **reconstruction MSE** / variance explained on activation shards
- **Mean L0** (avg active features per token)
- **Dead feature** fraction (never fires above threshold)
- Notebook: **top-10 max-activating** token snippets per feature

## Anti-patterns

- Using wrong `hidden_states` index (layer off-by-one) — verify against [reference.md](reference.md)
- Training on MLP or attention hooks while believing it is residual post
- Skipping activation normalization expected by SAELens
- Caching huge corpora on the first run — use a small shard until pipeline works
- Confusing Pythia **pretrain checkpoint step** with SAE training **optimizer steps**
- Mixing SAELens and minimal training code in one file

## Agent behavior

1. Read this skill and, if needed, [reference.md](reference.md) / [examples.md](examples.md).
2. Run the **decision gate**; do not assume GPU or corpus size.
3. **Understand** → use the output template; stay concise.
4. **Build** → emit phased checklist + Path A/B steps; scaffold only when approved.
5. When scaffolding, match existing repo files if present; otherwise create the target layout above.
6. README must document: model id, layer, `d_sae`, cache → train → eval commands.

## More detail

- [reference.md](reference.md) — dimensions, hook mapping, hyperparameters, SAELens vs minimal notes
- [examples.md](examples.md) — sample /plan responses
