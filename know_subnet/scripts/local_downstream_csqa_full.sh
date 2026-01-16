#!/usr/bin/env bash
set -euo pipefail

# Full finetuning (unfreeze transformer).
# Runs four setups in sequence:
# 1) Full (no mask removal)
# 2) Random Mask
# 3) Neuron Mask (mask removal)
# 4) Weight Mask (mask removal)
# Override checkpoint dirs if needed.
CHECKPOINT_DIR_WEIGHT="${CHECKPOINT_DIR_WEIGHT:-logs/subnet_weights_csqa/logs}"
CHECKPOINT_DIR_NEURON="${CHECKPOINT_DIR_NEURON:-logs/subnet_neurons_csqa/logs}"
MASK_NAME="${MASK_NAME:-csqa}"

python -m know_subnet.downstream_task.main \
  main=main \
  mask=none \
  freeze=false

# Random mask removal using the weight-mask checkpoint dir.
python -m know_subnet.downstream_task.main \
  main=main \
  mask=random \
  mask_name="${MASK_NAME}" \
  checkpoint_dir="${CHECKPOINT_DIR_WEIGHT}" \
  freeze=false

# Neuron mask removal using the neuron-mask checkpoint dir.
python -m know_subnet.downstream_task.main \
  main=main \
  mask=csqa \
  mask_name="${MASK_NAME}" \
  checkpoint_dir="${CHECKPOINT_DIR_NEURON}" \
  freeze=false

# Weight mask removal using the weight-mask checkpoint dir.
python -m know_subnet.downstream_task.main \
  main=main \
  mask=csqa \
  mask_name="${MASK_NAME}" \
  checkpoint_dir="${CHECKPOINT_DIR_WEIGHT}" \
  freeze=false
