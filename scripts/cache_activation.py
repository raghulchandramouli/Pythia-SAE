import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def collect_texts(max_examples: int = 128) -> list[str]:
    
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    
    texts = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            texts.append(text)
            
        if len(texts) >= max_examples:
            break
    return texts

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pythia.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    model_id = config['model']['id']
    revision = config['model']['pythia_step']
    hidden_state_index = config['activation']['hf_hidden_states_index']
    max_tokens_cache = config['cache']['max_tokens_cache']
    shard_size = config['cache']['shard_size']
    output_dir = Path(config['cache']['output_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,   
        revision=revision,
        output_hidden_states=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    
    texts = collect_texts()
    
    cache_chunks  = []
    cached_tokens = 0
    shard_index   = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length = 128,
            ).to(device)
            
            outpus = model(**tokens)
            hidden = outputs.hidden_states[hidden_state_index]
            
            # shape: [batch, seq, d_in] -> tokens, d_in
            activations = hidden.reshape(-1, hidden.shape[-1]).cpu()
            
            cached_chunks.append(activations)
            cached_tokens += activations.shape[0]
            
            if cached_tokens >= shard_size_tokens:
                shard = torch.cat(cached_chunks, dim=0)
                shard_path = output_dir / f"shard_{shard_index:04d}.pt"
                torch.save({'activations' : shard}, shard_path)
                
                print(f"saved {shard_path} with shape {tuple(shard.shape)}")
                
                cached_chunks = []
                cached_tokens = 0
                shard_index += 1
            if shard_index * shard_size_tokens >= max_tokens_cache:
                break
            
        if cached_chunks:
            shard = torch.cat(cached_chunks, dim=0)
            shard_path = output_dir / f"shard_{shard_index:04d}.pt"
            torch.save({'activations' : shard}, shard_path})
            print(f"saved {shard_path} with shape {tuple(shard.shape)}")


if __name__ == "__main__":
    main()
