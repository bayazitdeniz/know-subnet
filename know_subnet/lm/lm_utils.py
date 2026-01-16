import os
import argparse
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from know_subnet.lm.gpt2 import (
    GPT2LM,
    SelectivePrunedGPT2LM
)
from know_subnet.utils import str2moduleclass

def partial_state_dict(
    state_dict: Dict[str, Any]
): 
    """
    Creates a partial state dictionary that stores only the mask scores.

    Args:
        state_dict (Dict[str, Any]): The complete state dictionary.

    Returns:
        OrderedDict[str, Any]: A state dictionary containing only the selected parameters.
    """
    partial_state_dict = OrderedDict()
    for name in state_dict.keys():
        if 'mask_scores' in name:
            partial_state_dict[name] = state_dict[name]
    return partial_state_dict

def save_mask_scores(
    model: nn.Module,
    log_dict: Dict[str, Union[int, float]],
    base_path: str,
    accelerator: Any = None
):
    """
    Saves mask parameters to disk.

    Args:
        model (nn.Module): The model containing the mask scores.
        log_dict (Dict[str, Union[int, float]]): Dictionary containing logging information,
            such as 'step', 'top1acc', and 'pct_binary_produced_mask_0'.
        base_path (str): The directory where the checkpoint will be saved.
        accelerator (Any, optional): Accelerator object if available. Defaults to None.
    """
    fname = "ckpt-step={}-sparsity={:.5f}".format(
        int(log_dict['step']),
        log_dict['val/targetkg-pct_binary_produced_mask_0']
    )
    save_path = os.path.join(base_path, fname)
    state = partial_state_dict(model.state_dict())
    if accelerator is None:
        torch.save(state, save_path)
    else:
        accelerator.save(state, save_path)


def torch_load(use_cuda: bool, load_path: str) -> Any:
    """
    Loads a PyTorch checkpoint from the given path.

    Args:
        use_cuda (bool): If True, load the checkpoint on GPU; otherwise, load on CPU.
        load_path (str): The path to the checkpoint file.

    Returns:
        Any: The loaded checkpoint.
    """
    if use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def load_state_dict_incomplete(
    model: nn.Module,
    state_dict: Dict[str, Any],
    child: bool = False
):
    """
    Loads a partial state dictionary into a model, allowing for an incomplete match. 
    If `child` is True, the keys in the provided state dictionary are adjusted by removing the prefix up to and including the first '.'.
    If keys in the provided state dictionary are not found in the model's state dictionary, raises a ValueError.
    
    Args:
        model (nn.Module): The model into which the state dictionary will be loaded.
        state_dict (Dict[str, Any]): The state dictionary to load.
        child (bool): Flag indicating whether the state dict keys require adjustment.
    """
    # NOTE: Used to have to do this for accelerator, but it doesn't seem to be necessary anymore.
    #       Coulld be a python / transformer version issue.
    # if child:
    #     state_dict = {k[k.index('.')+1:]: v for k, v in state_dict.items() if '.' in k}
    
    model_dict = model.state_dict()
    not_in_model_dict = [k for k in state_dict.keys() if k not in model_dict]
    if len(not_in_model_dict) != 0:
        raise ValueError(f"The following found keys that are not in the skeleton model:\n{not_in_model_dict}")
    
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

def class_name_to_class(class_name_list: List[str]) -> List[type]:
    """Converts a list of class name strings to actual class objects."""
    class_list = []
    for i in class_name_list:
        class_name = i.split("'")[1].split('.')[4]
        class_list.extend(str2moduleclass(class_name))
    return class_list

def load_from_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: str,
    use_cuda: bool,
    verbose: bool = False,
    used_accelerator: bool = False
) -> nn.Module:
    """
    Loads a model from a checkpoint with a partially matching state dictionary 
    and freezes non-masked parameters.

    Args:
        config (Dict[str, Any]): Model config.
        checkpoint_path (str): Path to the checkpoint file.
        use_cuda (bool): Flag to indicate if CUDA should be used.
        verbose (bool): Enable verbose logging. Defaults to False.
        used_accelerator (bool): Indicates if an accelerator was used to save the checkpoint.
            Defaults to False.

    Returns:
        nn.Module: The model loaded with the checkpoint weights.
    """
    out_w_per_mask = config["params"][0]
    in_w_per_mask = config["params"][1]

    # Set default limits if not provided in config due to backward compatibility
    if ("top_limit" not in config.keys()) or ("bottom_limit" not in config.keys()):
        config["top_limit"] = -1
        config["bottom_limit"] = -1

    if config["lm"].startswith('gpt'):
        model = SelectivePrunedGPT2LM(
            out_w_per_mask,
            in_w_per_mask, 
            lm_name=config["lm"],
            top_k_layers=config["top_k_layers"],
            linear_types_to_mask=config["linear_types_to_mask"], 
            module_types_to_mask=class_name_to_class(config['module_types_to_mask']),
            use_dropout=config["use_dropout"],
            initial_mask_p=config["initial_mask_p"],
            top_limit=config["top_limit"],
            bottom_limit=config["bottom_limit"],
            verbose=verbose,
        )
    else:
        raise ValueError("Could not find matching model name for {}, only GPT2 family models are supported.".format(config["lm"]))

    state_dict = torch_load(use_cuda, checkpoint_path)
    load_state_dict_incomplete(model, state_dict, child=used_accelerator)
    model.freeze_params(exclude_name_list=["mask_scores"], verbose=False)
    return model

def load_lm(args: argparse.Namespace) -> nn.Module:
    """
    Loads a language model based on the provided argparse arguments.

    Depending on whether the model is pruned or not, this function creates an instance
    of either SelectivePrunedGPT2LM or GPT2LM. After model instantiation, it freezes the
    non-masked parameters.

    Args:
        args (argparse.Namespace): An object with attributes specifying model configurations. Expected
            attributes include: is_pruned, lm, params, top_k_layers, linear_types_to_mask,
            module_types_to_mask, use_dropout, initial_mask_p, top_limit, bottom_limit, verbose.

    Returns:
        nn.Module: The loaded language model.
    """
    if not args.test_full_model:
        if args.lm.startswith('gpt2'):
            out_w_per_mask, in_w_per_mask = args.params
            print("Prune - (out,in)_w_per_mask: {}".format((out_w_per_mask, in_w_per_mask)))
            model = SelectivePrunedGPT2LM(
                out_w_per_mask,
                in_w_per_mask, 
                lm_name=args.lm,
                top_k_layers=args.top_k_layers,
                linear_types_to_mask=args.linear_types_to_mask, 
                module_types_to_mask=args.module_types_to_mask,
                use_dropout=args.use_dropout,
                initial_mask_p=args.initial_mask_p,
                top_limit=args.top_limit,
                bottom_limit=args.bottom_limit,
                verbose=args.verbose,
            )
        else:
             raise ValueError("Could not find matching model name for {}, only GPT2 family models are supported.".format(args.lm))
    else:
        if args.lm.startswith('gpt2'):
            model = GPT2LM(
                use_dropout=args.use_dropout,
                lm_name=args.lm
            )
        else:
            raise ValueError("Could not find matching model name for {}, only GPT2 family models are supported.".format(args.lm))
        
    model.freeze_params(exclude_name_list=["mask_scores"], verbose=False)
    return model

def create_uniform_dist(tok_labels: torch.Tensor, lm_name: str) -> torch.Tensor:
    """
    Creates a uniform distribution tensor over the model vocabulary 
    for a specific batch of labels (as the number of masked tokens within a 
    sequence may vary in the batch).

    Args:
        tok_labels (torch.Tensor): A tensor containing token labels.
        lm_name (str): The name of the language model.

    Returns:
        torch.Tensor: A tensor of shape (num_masked_labels, vocab_size) with uniform probabilities.
    """
    config = AutoConfig.from_pretrained(lm_name)
    mask = tok_labels != -100
    masked_labels = tok_labels[mask]
    vocab_size = config.vocab_size    
    uni_prb = 1.0 / vocab_size
    uniform_prbs = torch.full((masked_labels.shape[0], vocab_size), uni_prb)
    return uniform_prbs
