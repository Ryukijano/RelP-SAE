#!/bin/bash

# RelP-SAE Setup Script
# Quick setup for the RelP-SAE implementation

echo "🔧 Setting up RelP-SAE environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create environment
echo "📦 Creating conda environment 'relp_sae'..."
conda create -n relp_sae python=3.10 -y

# Activate environment
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate relp_sae

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if RelP repository exists
if [ ! -d "../RelP" ]; then
    echo "📥 Cloning RelP repository..."
    cd ..
    git clone https://github.com/FarnoushRJ/RelP.git
    cd RelP/TransformerLens
    pip install -e .
    cd ../../relp_sae_implementation
else
    echo "✅ RelP repository found. Installing TransformerLens..."
    cd ../RelP/TransformerLens
    pip install -e .
    cd ../../relp_sae_implementation
fi

# Verify installation
echo "🔍 Verifying setup..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformer_lens; print('TransformerLens: ✓')"
python -c "import flash_attn; print('FlashAttention: ✓')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick start:"
echo "   conda activate relp_sae"
echo "   python demo.py"
echo ""
echo "📖 For full analysis:"
echo "   python run_attribution.py"
