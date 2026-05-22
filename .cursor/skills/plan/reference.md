# Pythia-70M SAE — reference

## Model configuration

| Field | Value |
|-------|--------|
| HuggingFace id | `EleutherAI/pythia-70m`, `EleutherAI/pythia-70m-deduped` |
| Architecture | GPT-NeoX |
| Parameters | ~70M |
| `d_model` | 512 |
| `n_layers` | 6 (layer indices **0–5**) |
| `n_heads` | 8 |
| `d_head` | 64 |
| MLP inner dim | 2048 |
| Vocab size | 50304 |
| SAE input `d_in` | **512** (residual stream at hook) |

### Pythia checkpoint steps

Pythia releases multiple training snapshots as HF revisions (e.g. `step0`, `step1000`, … `step143000`). SAEs trained on one step should be evaluated on the **same** step. Default for experiments: **`step143000`** (fully trained 70M) unless studying training dynamics.

Load example:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    revision="step143000",
)
```

TransformerLens: use `HookedTransformer.from_pretrained` with the matching `checkpoint_index` / config for the desired step.

---

## Hook sites and layer indexing

**Goal:** SAE on the **residual stream after transformer block L**.

| Layer L (0-based) | TransformerLens hook | Meaning |
|-------------------|----------------------|---------|
| 0 | `blocks.0.hook_resid_post` | After block 0 |
| 4 | `blocks.4.hook_resid_post` | After block 4 (**default**) |
| 5 | `blocks.5.hook_resid_post` | After final block |

Other useful hooks (not default for first SAE):

| Hook | Use case |
|------|----------|
| `blocks.L.hook_resid_mid` | Mid-block residual |
| `blocks.L.hook_mlp_out` | MLP sublayer output |
| `blocks.L.attn.hook_result` | Attention output |

### HuggingFace `output_hidden_states` mapping

With `output_hidden_states=True`, `hidden_states[k]` has shape `[batch, seq, 512]`:

| `k` | Content |
|-----|---------|
| 0 | Embeddings |
| 1 | After layer 0 |
| 2 | After layer 1 |
| … | … |
| `L + 1` | **After layer L** |

**Examples:**

- Residual after block **4** → `hidden_states[5]`
- Residual after block **5** → `hidden_states[6]`

Off-by-one errors are the most common bug; always log `hidden_states` length (should be 7 for 6 layers).

### Token positions

| Strategy | Cache size | Interpretability |
|----------|------------|------------------|
| All tokens | Large | Standard for SAE papers |
| Last token only | Small | Good for quick tests |
| Prompt tokens only | Medium | Context-specific features |

---

## SAE architecture

```
x  ∈ R^512   (one residual vector)
f  = encoder(x)  ∈ R^d_sae   (sparse after ReLU + L1 or TopK)
x̂ = decoder(f)  ∈ R^512
```

| Symbol | Typical value |
|--------|----------------|
| `d_in` | 512 |
| `d_sae` | 2048 (4×), 4096 (8×), 8192 (16×) |
| Sparsity | L1 on `f`, or TopK (e.g. k=32) |

**Loss (minimal):**

```
L = MSE(x, x̂) + λ * ||f||_1
```

Optional: auxiliary loss to reduce **dead features** (never activate).

**Decoder:** Many implementations unit-normalize decoder columns during training.

---

## Hyperparameter starter grid

| Parameter | Dev default | Notes |
|-----------|---------------|-------|
| `d_sae` | 4096 | 8× expansion |
| `l1_coeff` | 1e-3 | Sweep: 1e-4, 1e-3, 1e-2 |
| `batch_size` | 4096 tokens | Activations, not sequences |
| `lr` | 3e-4 | Adam |
| `train_steps` | 30k–100k | Until recon plateaus |
| `max_tokens_cache` | 1e7–1e8 | Start with 1e6 for smoke test |

Sweep order: fix layer 4 → tune `l1_coeff` → then try `d_sae`.

---

## Path A — SAELens + TransformerLens

**Repos:**

- [SAELens](https://github.com/jbloomAus/SAELens) — training, caching, eval harness
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — `HookedTransformer`, named hooks

**Install (example):**

```bash
pip install sae-lens transformer-lens torch datasets transformers
```

**Conceptual config fields** (names may vary by SAELens version; check repo docs):

| Key | Example |
|-----|---------|
| `model_name` | `pythia-70m` or TL alias |
| `hook_name` | `blocks.4.hook_resid_post` |
| `d_in` | 512 |
| `d_sae` | 4096 |
| `l1_coefficient` | 0.001 |
| `dataset` | small OpenWebText / custom JSONL |
| `training_tokens` | 10_000_000 |

Workflow: configure run → cache activations (if separate) → train → run eval metrics packaged with SAELens.

**Normalization:** SAELens may expect centered or scaled activations; follow the version’s README for `normalize_activations` / `dtype` — match at cache and train time.

---

## Path B — Minimal custom SAE

### `src/sae.py` sketch

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in=512, d_sae=4096):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

def loss(x, x_hat, f, l1_coeff=1e-3):
    return F.mse_loss(x_hat, x) + l1_coeff * f.abs().mean()
```

### Activation caching (HF hook)

```python
acts = []

def hook(module, input, output):
    # output[0] or output depending on GPT-NeoX block API
    acts.append(output[0].detach().cpu())

handle = model.gpt_neox.layers[L].register_forward_hook(hook)
```

Prefer verifying tensor shape `[batch, seq, 512]` on a single batch before full cache run.

Shard format: `torch.save({"acts": tensor}, "shards/shard_000.pt")` with shape `[N, 512]`.

---

## Corpora (development)

| Corpus | Notes |
|--------|-------|
| OpenWebText (small slice) | Common in interpretability; easy via `datasets` |
| SlimPajama subset | Larger; use after pipeline works |
| Custom JSONL | Tiny sentences for debugging |

Tokenize with the **same tokenizer** as Pythia (`AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")`).

---

## Evaluation metrics

| Metric | Definition |
|--------|------------|
| Reconstruction MSE | `mean((x - x̂)²)` on held-out shards |
| Variance explained | `1 - MSE / Var(x)` |
| L0 | Count of features with activation > ε per token |
| Dead features | Fraction of features that never fire > ε on sample |

---

## Feature dashboards (optional)

- [Neuronpedia](https://www.neuronpedia.org/) — host / browse SAE features when exported in supported format
- Local notebook: sort tokens by `f[i]` for feature index `i`

---

## Glossary

| Term | Meaning |
|------|---------|
| Residual stream | Main hidden state passed between sublayers |
| Feature | One dimension of SAE latent `f` |
| Activation cache | Disk store of vectors `x` used to train SAE |
| Pythia step | LM pretrain checkpoint, not SAE step |
