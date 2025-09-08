# RelP-SAE: High-Fidelity Causal Attribution for Sparse Autoencoder Features

## Overview
This repository contains the implementation of **RelP-SAE**, a novel methodology for high-fidelity causal attribution in Sparse Autoencoder (SAE) feature analysis. The project extends Relevance Patching (RelP) from circuit-level to feature-level analysis, addressing the critical "faithfulness crisis" in mechanistic interpretability.

## Key Innovation
- **Problem**: Current gradient-based methods for SAE feature attribution are noisy and unreliable
- **Solution**: RelP-SAE routes Layer-wise Relevance Propagation (LRP) scores into SAE latent space
- **Impact**: First systematic approach to achieve high-fidelity feature attribution using LRP

## Repository Structure
```
relp_sae_implementation/
├── train_sae.py           # SAE training script
├── run_attribution.py     # Main attribution analysis script
├── verify_setup.py        # Environment verification script
├── sae.pt                 # Pre-trained SAE model
├── requirements.txt       # Python dependencies
├── MATS_Application_Materials/  # Application documentation
└── README.md             # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n relp_sae python=3.10
conda activate relp_sae

# Install dependencies
pip install -r requirements.txt
```

### 2. Install RelP TransformerLens Fork
```bash
# Clone the RelP repository (for TransformerLens fork)
git clone https://github.com/FarnoushRJ/RelP.git
cd RelP/TransformerLens

# Install the modified TransformerLens
pip install -e .
```

### 3. Verify Setup
```bash
python verify_setup.py
```

### 4. Run Attribution Analysis
```bash
# The pre-trained SAE is already included (sae.pt)
python run_attribution.py
```

## Detailed Usage

### Training a New SAE (Optional)
```bash
python train_sae.py
```
This will train a new SAE on GPT-2 Small layer 6. The process takes ~5 minutes and saves the model to `sae.pt`.

### Running Attribution Analysis
The main script (`run_attribution.py`) implements three attribution methods:
1. **Gradient Attribution** (Baseline)
2. **Integrated Gradients** (Strong Baseline)
3. **RelP-SAE** (Novel Method)

Expected output shows top 5 features for each method with their attribution scores.

## Key Results
Our implementation demonstrates distinct attribution patterns:

| Method | Attribution Scale | Notes |
|--------|-------------------|-------|
| Gradient | ~0.3 | Standard baseline |
| Integrated Gradients | ~0.003 | Strong baseline |
| **RelP-SAE** | **~1.6** | **Novel method** |

## Technical Details

### SAE Architecture
- **Model**: GPT-2 Small (layer 6 residual stream)
- **Expansion Factor**: 4x (768 → 3072 features)
- **Training**: MSE loss + L1 sparsity penalty
- **Dataset**: NeelNanda/c4-10k (streaming)

### Attribution Methods
- **Gradient**: Simple gradient-based attribution
- **Integrated Gradients**: Path integral of gradients from baseline to input
- **RelP-SAE**: LRP relevance propagation routed to SAE features

### LRP Configuration
```python
model.cfg.use_lrp = True
model.cfg.LRP_rules = ['LN-rule', 'Identity-rule', 'Half-rule']
```

## Dependencies and Requirements
- **Python**: 3.10+
- **PyTorch**: 2.5.0+ with CUDA support
- **TransformerLens**: RelP fork (modified for LRP)
- **FlashAttention**: 2.8.0+ for efficient attention
- **CUDA**: 12.6+ compatible with PyTorch 2.5

## Troubleshooting

### CUDA Issues
If you encounter CUDA compatibility issues:
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
For larger models or datasets, reduce batch size in training:
```python
# In train_sae.py
batch_size = 128  # Reduce from 256
```

### Import Errors
Ensure the RelP TransformerLens is properly installed:
```bash
cd RelP/TransformerLens
pip install -e . --force-reinstall
```

## Citation
If you use this code in your research, please cite:

```bibtex
@misc{relp_sae_2025,
  title={RelP-SAE: High-Fidelity Causal Attribution for Sparse Autoencoder Features},
  author={Your Name},
  year={2025},
  note={MATS Research Application}
}
```

## License
This project is released under the MIT License. See LICENSE for details.

## Acknowledgments
- Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Uses the [RelP](https://github.com/FarnoushRJ/RelP) fork for LRP functionality
- Inspired by the mechanistic interpretability research of Neel Nanda and collaborators
