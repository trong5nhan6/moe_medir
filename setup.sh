#!/bin/bash
# Run once: bash setup.sh

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install medmnist
pip install open_clip_torch
pip install pytorch-metric-learning
pip install scikit-learn matplotlib seaborn
pip install tqdm pandas numpy timm

# Verify BiomedCLIP
python -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
print('BiomedCLIP loaded OK')
"
