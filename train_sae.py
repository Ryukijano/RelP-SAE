import torch
import transformer_lens
from datasets import load_dataset
from tqdm.auto import tqdm
import os

# Configuration
MODEL_NAME = "gpt2-small"
DATASET_NAME = "NeelNanda/c4-10k"
LAYER = 6
D_MODEL = 768  # gpt2-small
EXPANSION_FACTOR = 4
D_SAE = D_MODEL * EXPANSION_FACTOR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAE_PATH = "sae.pt"

class SAE(torch.nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_sae)
        self.relu = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(d_sae, d_model)

    def forward(self, x):
        features = self.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features

def main():
    """Trains and saves a sparse autoencoder."""
    print(f"Using device: {DEVICE}")

    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    sae = SAE(D_MODEL, D_SAE).to(DEVICE)
    print("SAE model created.")

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    # We will just take a small sample for this example training run
    activations_list = []
    num_tokens_to_grab = 100000 
    
    hook_name = f"blocks.{LAYER}.hook_resid_pre"

    print(f"Fetching activations from layer {LAYER}...")
    with torch.no_grad():
        for batch in tqdm(dataset.take(1000), total=1000):
            tokens = model.to_tokens(batch['text'], truncate=True, move_to_device=True)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            activations = cache[hook_name].view(-1, D_MODEL)
            activations_list.append(activations.cpu())
            if sum(len(acts) for acts in activations_list) > num_tokens_to_grab:
                break

    all_activations = torch.cat(activations_list, dim=0)
    print(f"Total activations fetched: {all_activations.shape[0]}")
    
    # Training loop
    l1_lambda = 1e-3
    learning_rate = 3e-4
    epochs = 3
    batch_size = 256

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    print("Starting SAE training...")
    for epoch in range(epochs):
        sae.train()
        total_loss = 0
        
        # Shuffle data for each epoch
        indices = torch.randperm(all_activations.shape[0])
        all_activations = all_activations[indices]

        for i in tqdm(range(0, all_activations.shape[0], batch_size)):
            batch = all_activations[i:i+batch_size].to(DEVICE)
            
            optimizer.zero_grad()
            
            reconstruction, features = sae(batch)
            
            mse_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            l1_loss = l1_lambda * features.abs().sum()
            
            loss = mse_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / (all_activations.shape[0] / batch_size)}")

    print("Training complete.")
    
    print(f"Saving trained SAE to {SAE_PATH}")
    torch.save(sae.state_dict(), SAE_PATH)
    print("SAE saved.")


if __name__ == "__main__":
    main()
