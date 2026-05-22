import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        
        self.d_in = d_in
        self.d_sae = d_sae
        
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
def sae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    features: torch.Tensor,
    l1_coeff: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    mse_loss = F.mse_loss(reconstruction, x)
    l1_loss  = features.abs().mean()
    total_loss = mse_loss + l1_coeff * l1_loss
    
    metrics = {
        "loss"     : total_loss.detach(),
        "mse_loss" : mse_loss.detach(),
        "l1_loss"  : l1_loss.detach(),
        "mean_l0"  : (features > 0).float().sum(dim=-1).mean().detach(),
    }
    
    return total_loss, metrics










