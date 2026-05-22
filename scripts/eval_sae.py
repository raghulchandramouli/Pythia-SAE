import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sae import SparseAutoencoder


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_activation_shards(cache_dir: Path) -> torch.Tensor:
    shard_paths = sorted(cache_dir.glob("shard_*.pt"))

    if not shard_paths:
        raise FileNotFoundError(f"No activation shards found in {cache_dir}")

    chunks = []
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu")
        chunks.append(shard["activations"])

    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/pythia70m_layer4/sae_final.pt",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    d_in = config["activation"]["d_in"]
    d_sae = config["sae"]["d_sae"]
    batch_size = config["training"]["batch_size"]
    cache_dir = Path(config["cache"]["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    activations = load_activation_shards(cache_dir)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = SparseAutoencoder(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_mse = 0.0
    total_var = 0.0
    total_l0 = 0.0
    total_tokens = 0

    active_features = torch.zeros(d_sae, dtype=torch.bool)

    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device, dtype=torch.float32)

            reconstruction, features = model(batch)

            mse_per_token = ((batch - reconstruction) ** 2).mean(dim=-1)
            var_per_token = ((batch - batch.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
            l0_per_token = (features > 0).float().sum(dim=-1)

            total_mse += mse_per_token.sum().item()
            total_var += var_per_token.sum().item()
            total_l0 += l0_per_token.sum().item()
            total_tokens += batch.shape[0]

            active_features |= (features.detach().cpu() > 0).any(dim=0)

    mean_mse = total_mse / total_tokens
    mean_var = total_var / total_tokens
    variance_explained = 1.0 - (mean_mse / mean_var)
    mean_l0 = total_l0 / total_tokens
    dead_feature_fraction = 1.0 - active_features.float().mean().item()

    print(f"checkpoint: {args.checkpoint}")
    print(f"tokens evaluated: {total_tokens}")
    print(f"reconstruction_mse: {mean_mse:.6f}")
    print(f"variance_explained: {variance_explained:.4f}")
    print(f"mean_l0: {mean_l0:.2f}")
    print(f"dead_feature_fraction: {dead_feature_fraction:.4f}")


if __name__ == "__main__":
    main()