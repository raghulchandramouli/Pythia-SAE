import argparse
import sys
from pathlib import Path

import torch, yaml
from torch.utils.data import DataLoader, TensorDataset


sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.sae import SparseAutoencoder, sae_loss


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_activation_shards(cache_dir: Path) -> torch.Tensor:
    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    
    if not shard_paths:
        raise FileNotFoundError("No activation shards found")
    
    chunks = []
    for path in shard_paths:
        shard = torch.load(path, map_location='cpu')
        chunks.append(shard['activations'])
        
    return torch.cat(chunks, dim = 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    d_in = config["activation"]["d_in"]
    d_sae = config["sae"]["d_sae"]
    l1_coeff = config["sae"]["l1_coeff"]

    batch_size = config["training"]["batch_size"]
    train_steps = config["training"]["train_steps"]
    lr = config["training"]["lr"]
    seed = config["training"]["seed"]

    cache_dir = Path(config["cache"]["output_dir"])
    run_dir = Path(config["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    activations = load_activation_shards(cache_dir)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SparseAutoencoder(d_in=d_in, d_sae=d_sae).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0

    while step < train_steps:
        for (batch,) in dataloader:
            batch = batch.to(device, dtype=torch.float32)

            reconstruction, features = model(batch)
            loss, metrics = sae_loss(
                x=batch,
                reconstruction=reconstruction,
                features=features,
                l1_coeff=l1_coeff,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"step {step:05d} "
                    f"loss={metrics['loss'].item():.6f} "
                    f"mse={metrics['mse_loss'].item():.6f} "
                    f"l1={metrics['l1_loss'].item():.6f} "
                    f"l0={metrics['mean_l0'].item():.2f}"
                )

            if step % 500 == 0 and step > 0:
                checkpoint_path = run_dir / f"sae_step_{step:05d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "step": step,
                    },
                    checkpoint_path,
                )
                print(f"saved {checkpoint_path}")

            step += 1

            if step >= train_steps:
                break

    final_path = run_dir / "sae_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
        },
        final_path,
    )
    print(f"saved {final_path}")


if __name__ == "__main__":
    main()
