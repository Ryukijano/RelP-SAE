import torch
import transformer_lens
from train_sae import SAE, D_MODEL, D_SAE, LAYER, SAE_PATH

# Configuration
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """Loads models and runs attribution methods."""
    print(f"Using device: {DEVICE}")

    # Load the GPT-2 Small model
    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    # Load the trained SAE
    sae = SAE(D_MODEL, D_SAE).to(DEVICE)
    sae.load_state_dict(torch.load(SAE_PATH, map_location=DEVICE))
    sae.eval()
    print("Model and SAE loaded successfully.")

    # Define a test case (IOI task)
    prompt = "When John and Mary went to the store, John gave a drink to"
    correct_token = " Mary"
    incorrect_token = " John"

    correct_token_id = model.to_single_token(correct_token)
    incorrect_token_id = model.to_single_token(incorrect_token)

    print(f"Prompt: '{prompt}'")
    print(f"Correct continuation: '{correct_token}' (Token ID: {correct_token_id})")
    
    # Run Gradient Attribution
    print("\n--- Running Gradient Attribution (Baseline) ---")
    full_grad_attrib = get_gradient_attribution(model, sae, prompt, correct_token_id)
    
    # We only care about the gradients for the final token
    grad_attrib = full_grad_attrib[0, -1, :]

    # Get top 5 features
    top_k = 5
    top_grad_vals, top_grad_indices = torch.topk(grad_attrib.abs(), top_k)
    
    print(f"Top {top_k} features by gradient attribution:")
    for i in range(top_k):
        print(f"  Feature {top_grad_indices[i].item()}: Attribution = {top_grad_vals[i].item():.4f}")

    # Run Integrated Gradients Attribution
    print("\n--- Running Integrated Gradients Attribution (Strong Baseline) ---")
    ig_attrib = get_integrated_gradients_attribution(model, sae, prompt, correct_token_id)
    
    top_ig_vals, top_ig_indices = torch.topk(ig_attrib.abs(), top_k)
    
    print(f"Top {top_k} features by Integrated Gradients attribution:")
    for i in range(top_k):
        print(f"  Feature {top_ig_indices[i].item()}: Attribution = {top_ig_vals[i].item():.4f}")

    # Run RelP-SAE Attribution (Our Novel Method)
    print("\n--- Running RelP-SAE Attribution (Novel Method) ---")
    relp_sae_attrib = get_relp_sae_attribution(model, sae, prompt, correct_token_id)

    top_relp_vals, top_relp_indices = torch.topk(relp_sae_attrib.abs(), top_k)

    print(f"Top {top_k} features by RelP-SAE attribution:")
    for i in range(top_k):
        print(f"  Feature {top_relp_indices[i].item()}: Attribution = {top_relp_vals[i].item():.4f}")
    print("\nSetup complete. Ready to run attribution methods.")


def get_integrated_gradients_attribution(model, sae, prompt, token_id, steps=50):
    """Computes Integrated Gradients attribution for SAE features."""
    model.zero_grad()
    sae.zero_grad()

    hook_name = f"blocks.{LAYER}.hook_resid_pre"

    # Get the baseline (zero) and original activations
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(prompt, move_to_device=True), names_filter=[hook_name])
        original_resid = cache[hook_name]
        baseline_resid = torch.zeros_like(original_resid)

        # Get the SAE feature shapes
        _, original_features = sae(original_resid)
        integrated_grads = torch.zeros_like(original_features)
    sae_features_store = [None]

    for alpha in torch.linspace(0, 1, steps):

        def hook_fn_ig(resid, hook):
            # Interpolate between baseline and original
            interpolated_resid = baseline_resid + alpha * (original_resid - baseline_resid)

            sae_out, sae_features = sae(interpolated_resid)
            sae_features.retain_grad()
            # Store for gradient access later
            sae_features_store[0] = sae_features
            return sae_out

        logits = model.run_with_hooks(
            model.to_tokens(prompt, move_to_device=True),
            fwd_hooks=[(hook_name, hook_fn_ig)],
        )

        target_logit = logits[0, -1, token_id]
        target_logit.backward()

        # Accumulate the gradients
        integrated_grads += sae_features_store[0].grad

    # Average the gradients and multiply by the original activation difference
    _, original_features = sae(original_resid)
    attribution = (original_features - sae(baseline_resid)[1]) * (integrated_grads / steps)

    return attribution[0, -1, :]


def get_relp_sae_attribution(model, sae, prompt, token_id):
    """Computes RelP-SAE attribution using LRP relevance propagation."""
    model.zero_grad()
    sae.zero_grad()

    # Enable LRP
    model.cfg.use_lrp = True
    model.cfg.LRP_rules = ['LN-rule', 'Identity-rule', 'Half-rule']

    hook_name = f"blocks.{LAYER}.hook_resid_pre"

    # Store for gradients (which become LRP relevance when LRP is enabled)
    grad_store = [None]

    def hook_fn_relp(resid, hook):
        # Forward pass through SAE
        sae_out, sae_features = sae(resid)

        # Enable gradient computation for LRP relevance
        sae_features.retain_grad()
        grad_store[0] = sae_features

        return sae_out

    # Get the LRP relevance by running the model and doing backward pass
    logits = model.run_with_hooks(
        model.to_tokens(prompt, move_to_device=True),
        fwd_hooks=[(hook_name, hook_fn_relp)]
    )

    # Get the logit for the target token and compute backward pass
    target_logit = logits[0, -1, token_id]
    target_logit.backward()

    # The gradients now contain the LRP relevance scores
    if grad_store[0] is not None and grad_store[0].grad is not None:
        # Return the LRP relevance for the SAE features at the final token position
        return grad_store[0].grad[0, -1, :].detach()
    else:
        # Fallback to gradient-based attribution if LRP fails
        print("Warning: LRP relevance not captured, falling back to gradient attribution")
        return get_gradient_attribution(model, sae, prompt, token_id).detach()


def get_gradient_attribution(model, sae, prompt, token_id):
    """Computes gradient attribution for SAE features."""
    model.zero_grad()
    sae.zero_grad()

    hook_name = f"blocks.{LAYER}.hook_resid_pre"
    
    # We need to create a hook function that replaces the residual stream 
    # with the SAE's reconstruction and saves the features.
    sae_features_store = [None]
    def hook_fn(resid, hook):
        sae_out, sae_features = sae(resid)
        sae_features.retain_grad()
        sae_features_store[0] = sae_features
        return sae_out

    # Run the model with the hook
    logits = model.run_with_hooks(
        model.to_tokens(prompt, move_to_device=True),
        fwd_hooks=[(hook_name, hook_fn)]
    )

    # Get the logit for the target token
    target_logit = logits[0, -1, token_id]
    
    # Calculate gradients
    target_logit.backward()
    
    # The gradients are now stored in the sae_features tensor
    # We care about the attribution at the final token position
    return sae_features_store[0].grad


if __name__ == "__main__":
    main()
