#!/bin/bash
# Run ONCE on a LOGIN node (it has internet; compute nodes may not):
#   source env.sh && bash slurm/download_weights.sh
# Downloads the ImageNet weights of both encoders into $TORCH_HOME / $HF_HOME (inside SOLAR_ROOT,
# so your partner does not have to download them again).
python -c "
from models import build_model
for name in ('unet', 'segformer'):
    build_model(name, pretrained=True)
    print(name, 'weights OK')
"
