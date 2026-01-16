import glob
import os
import torch

from know_subnet.lm.lm_utils import load_from_checkpoint
from know_subnet.utils import parse_json_to_dict

EXPR_NAME = {
    "checkpoint-csqa": "lm=gpt2-kg=csqa-seed=84-ablation=no_expression-date=2023_09_14_01:18:07",
    "checkpoint-csqa1": "lm=gpt2-kg=csqa1-seed=1318-ablation=no_expression-date=2023_09_14_01:18:12",
    "checkpoint-csqa2": "lm=gpt2-kg=csqa2-seed=84-ablation=no_expression-date=2023_09_14_01:18:14",
}

def get_model_and_config(
        exper_name,
        checkpoint_step,
        logs_path
    ):
    base_path =  os.path.join(logs_path, EXPR_NAME[exper_name])
    config_paths = list(glob.iglob('{}/**/config*'.format(base_path), recursive=True))

    if checkpoint_step == 0:
        checkpoint_paths = list(glob.iglob('{}/**/checkpoints/*'.format(base_path, checkpoint_step), recursive=True))
    else:
        checkpoint_paths = list(glob.iglob('{}/**/checkpoints/ckpt-step={}-*'.format(base_path, checkpoint_step), recursive=True))

    print(checkpoint_paths)
    assert (len(checkpoint_paths) == 1)

    for config_path in config_paths:
        config = parse_json_to_dict(config_path)

    for checkpoint_path in checkpoint_paths:
        model = load_from_checkpoint(config, checkpoint_path, torch.cuda.is_available())
    
    return model, 0.5, config
