# Pythia-SAE

Sparse autoencoder experiments on residual stream activations from `EleutherAI/pythia-70m`.

## Goal

Train a sparse autoencoder on the residual stream after transformer block 4 of Pythia-70M.

Each token position gives one 512-dimensional activation vector. The SAE learns to reconstruct that vector using a sparse set of learned features.

## Default Experiment

- Model: `EleutherAI/pythia-70m`
- Pythia checkpoint: `step143000`
- Layer: 4
- Hook target: residual stream after block 4
- HuggingFace hidden state index: `hidden_states[5]`
- SAE input dimension: `512`
- SAE hidden dimension: `4096`
- Sparsity: L1 penalty

## Pipeline

1. Tokenize text with the Pythia tokenizer.
2. Run Pythia and cache residual activations from layer 4.
3. Train the SAE on cached activation vectors.
4. Evaluate reconstruction quality and sparsity.
5. Inspect max-activating token examples for individual features.

## Commands

```bash
python scripts/cache_activations.py --config configs/pythia70m_layer4.yaml
python scripts/train_sae.py --config configs/pythia70m_layer4.yaml
python scripts/eval_sae.py --config configs/pythia70m_layer4.yaml