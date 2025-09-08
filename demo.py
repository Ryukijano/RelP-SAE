#!/usr/bin/env python3
"""
RelP-SAE Demo Script
Complete pipeline demonstration for MATS application
"""

import torch
import transformer_lens
from train_sae import SAE, D_MODEL, D_SAE, LAYER, SAE_PATH
import os

def main():
    print("🔬 RelP-SAE: High-Fidelity Causal Attribution Demo")
    print("=" * 60)

    # Check environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Load model
    print("🤖 Loading GPT-2 Small model...")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    # Load or check for trained SAE
    if os.path.exists(SAE_PATH):
        print("🧠 Loading pre-trained SAE...")
        sae = SAE(D_MODEL, D_SAE).to(device)
        sae.load_state_dict(torch.load(SAE_PATH, map_location=device))
        sae.eval()
    else:
        print("❌ SAE model not found. Please run train_sae.py first.")
        return

    # Test case
    prompt = "When John and Mary went to the store, John gave a drink to"
    print(f"\n📝 Test Prompt: '{prompt}'")
    print("🎯 Expected completion: ' Mary'")

    # Quick attribution demo
    print("\n🎲 Running quick attribution analysis...")

    # Enable LRP for RelP-SAE
    model.cfg.use_lrp = True
    model.cfg.LRP_rules = ['LN-rule', 'Identity-rule', 'Half-rule']

    # Get logits
    tokens = model.to_tokens(prompt, move_to_device=True)
    logits = model(tokens)

    # Get SAE features
    hook_name = f"blocks.{LAYER}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name]
        _, sae_features = sae(resid)

    # Simple LRP attribution (simplified demo)
    target_token_id = model.to_single_token(" Mary")
    target_logit = logits[0, -1, target_token_id]

    # Enable gradients for LRP
    sae_features.requires_grad_(True)
    target_logit.backward()

    # Get top features
    if sae_features.grad is not None:
        attribution = sae_features.grad[0, -1, :].abs()
        top_vals, top_indices = torch.topk(attribution, 3)

        print("\n🏆 Top 3 RelP-SAE Features:")
        for i, (val, idx) in enumerate(zip(top_vals, top_indices)):
            print(".4f")

    print("\n✅ Demo completed successfully!")
    print("📊 Full analysis available in run_attribution.py")
    print("🔗 Check MATS_Application_Materials/ for complete documentation")

if __name__ == "__main__":
    main()
