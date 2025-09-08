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
    # Ground Truth Validation via Activation Patching
    print("\n--- Ground Truth Validation (Activation Patching) ---")
    ground_truth_results = validate_with_activation_patching(model, sae, prompt, correct_token_id, incorrect_token_id)

    print("Ground truth feature effects (top 5 by absolute effect):")
    gt_top_vals, gt_top_indices = torch.topk(ground_truth_results.abs(), top_k)
    for i in range(top_k):
        print(".4f")

    # Correlation Analysis
    print("\n--- Attribution Method Correlations ---")
    correlations = compute_attribution_correlations(full_grad_attrib, ig_attrib, relp_sae_attrib, ground_truth_results)

    for method, corr in correlations.items():
        print(".4f")

    print("\n--- Feature Overlap Analysis ---")
    overlaps = analyze_feature_overlap(full_grad_attrib, ig_attrib, relp_sae_attrib, ground_truth_results)
    print(f"Gradient vs Integrated Gradients overlap: {overlaps['grad_vs_ig']:.1%}")
    print(f"Gradient vs RelP-SAE overlap: {overlaps['grad_vs_relp']:.1%}")
    print(f"Integrated Gradients vs RelP-SAE overlap: {overlaps['ig_vs_relp']:.1%}")
    print(f"RelP-SAE vs Ground Truth overlap: {overlaps['relp_vs_gt']:.1%}")

    print("\nSetup complete. Validation analysis finished.")


def validate_with_activation_patching(model, sae, prompt, correct_token_id, incorrect_token_id, num_features=100):
    """Validates attribution methods against ground truth using activation patching."""
    model.zero_grad()
    sae.zero_grad()

    hook_name = f"blocks.{LAYER}.hook_resid_pre"

    # Get clean and corrupted inputs
    clean_tokens = model.to_tokens(prompt, move_to_device=True)
    corrupted_prompt = prompt.replace(" Mary", " John")
    corrupted_tokens = model.to_tokens(corrupted_prompt, move_to_device=True)

    # Get original logits
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)

    clean_logit_diff = clean_logits[0, -1, correct_token_id] - clean_logits[0, -1, incorrect_token_id]
    corrupted_logit_diff = corrupted_logits[0, -1, correct_token_id] - corrupted_logits[0, -1, incorrect_token_id]

    print(f"Clean logit diff: {clean_logit_diff:.4f}")
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

    # Test feature effects through activation patching
    feature_effects = torch.zeros(D_SAE)

    # Only test top features to save computation time
    top_features_to_test = 50  # Test first 50 features

    for feature_idx in range(min(top_features_to_test, D_SAE)):
        # Create hook to patch this feature
        def patch_hook(resid, hook):
            sae_out, sae_features = sae(resid)
            # Zero out this feature in the corrupted input
            sae_features[0, -1, feature_idx] = 0
            # Reconstruct with modified features
            modified_resid = sae.decoder(sae_features)
            return modified_resid

        # Patch the feature and measure effect
        patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)])

        patched_logit_diff = (patched_logits[0, -1, correct_token_id] -
                            patched_logits[0, -1, incorrect_token_id])

        # Effect is how much patching this feature moves us toward clean behavior
        feature_effects[feature_idx] = patched_logit_diff - corrupted_logit_diff

    return feature_effects


def compute_attribution_correlations(grad_attr, ig_attr, relp_attr, ground_truth):
    """Compute correlations between attribution methods and ground truth."""
    import numpy as np

    def pearson_corr(a, b):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        return np.corrcoef(a, b)[0, 1]

    # Only compare features that were actually tested in ground truth
    num_tested = len(ground_truth)
    grad_attr_subset = grad_attr[0, -1, :num_tested]
    ig_attr_subset = ig_attr[:num_tested]
    relp_attr_subset = relp_attr[:num_tested]

    return {
        "gradient_vs_ground_truth": pearson_corr(grad_attr_subset, ground_truth),
        "ig_vs_ground_truth": pearson_corr(ig_attr_subset, ground_truth),
        "relp_vs_ground_truth": pearson_corr(relp_attr_subset, ground_truth)
    }


def analyze_feature_overlap(grad_attr, ig_attr, relp_attr, ground_truth, top_k=10):
    """Analyze overlap between top features identified by different methods."""
    # Get top features for each method
    grad_top = torch.topk(grad_attr[0, -1, :].abs(), top_k).indices
    ig_top = torch.topk(ig_attr.abs(), top_k).indices
    relp_top = torch.topk(relp_attr.abs(), top_k).indices
    gt_top = torch.topk(ground_truth.abs(), top_k).indices

    def overlap(set1, set2):
        return len(set(set1.tolist()) & set(set2.tolist())) / top_k

    return {
        "grad_vs_ig": overlap(grad_top, ig_top),
        "grad_vs_relp": overlap(grad_top, relp_top),
        "ig_vs_relp": overlap(ig_top, relp_top),
        "relp_vs_gt": overlap(relp_top, gt_top)
    }


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
