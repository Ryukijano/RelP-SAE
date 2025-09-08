import torch
import transformer_lens

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the GPT-2 Small model
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", device=device)

# Enable LRP
model.cfg.use_lrp = True
# The RelP paper uses a specific set of rules, let's use a simple one for now
# to confirm it works. We will use the full set in the actual experiment.
model.cfg.LRP_rules = ['Identity-rule']

print("Model loaded and LRP enabled successfully.")

# A simple test run
prompt = "Hello, world!"
print(f"Running model with prompt: '{prompt}'")
logits = model(prompt)
print("Model run successful.")

print("\nSetup verification complete. You are ready to proceed with the experiments.")
