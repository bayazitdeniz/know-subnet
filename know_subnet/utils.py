import os
import csv
import json
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import transformers
from collections import OrderedDict
from datetime import datetime

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

################################################################################
# REPRODUCIBILITY UTILITIES
################################################################################

def set_seed(seed: int):
    """Set random seed for all libraries for reproducibility

    Args:
        seed (int): the seed to set
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: if you will use distributed training 
    #       you might want to consider uncommenting the next line
    # torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    transformers.set_seed(seed)

def set_seed_and_device(args: argparse.Namespace) -> argparse.Namespace:
    """Set random seeds and determine device based on CUDA availability.
    Updates the args object with a `use_cuda` attribute.

    Args:
        args (argparse.Namespace): parsed args object
    Returns:
        argparse.Namespace: updated args objects
    """
    set_seed(args.seed)
        
    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    print("CUDA flag:", args.use_cuda)
    print("Device:", device)

    return args

def seed_worker():
    """
    Set seeds for dataloader worker processes.
    This function can be passed to the DataLoader worker_init_fn.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_rng_state(tag=""):
    """
    Print the current RNG states for torch, torch.cuda, numpy, and python's random.
    """
    print(f"\n=== RNG snapshot {tag} ===")
    print("torch:", torch.get_rng_state()[:5])  # show first 5 bytes
    print("torch.cuda:", torch.cuda.get_rng_state_all()[0][:5])  # first GPU
    print("numpy:", np.random.get_state()[1][:5])
    print("python:", random.getstate()[1][:5])


################################################################################
# LOGGING UTILITIES
################################################################################

def build_main_log_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Build and create the main logging directory.

    Depending on whether an NFS directory is provided in the args,
    this function sets up the base experiment directory for logs.

    Args:
        args: Parsed command-line arguments. Expected attributes:
            - nfs_dir: a string (empty if not provided)
            - verbose: a boolean flag for verbose output.

    Returns:
        The updated args with the 'exper_dir' attribute set.
    """
    # Use the NFS directory if provided, otherwise use the default DIR_PATH.
    base_dir = os.path.join(args.nfs_dir, "logs") if args.nfs_dir == '' else os.path.join(DIR_PATH, "logs")
    args.exper_dir = base_dir

    os.makedirs(args.exper_dir, exist_ok=True)

    if args.verbose:
        print("All logs will be located in:", args.exper_dir)

    return args


def create_experiment_folder(args: argparse.Namespace) -> argparse.Namespace:
    """
    Create an experiment folder to store configuration, checkpoints, etc.

    The folder path is constructed based on project name, logging directory,
    experiment name, and date. The date is set to the current timestamp if not provided.

    Args:
        args: Parsed command-line arguments. Expected attributes include:
            - date: a string (can be empty)
            - exper_dir: base experiment directory
            - project_name: name of the project
            - log_dir: name of the logging directory within the project
            - exper_name: name of the experiment
            - verbose: a boolean flag for verbose output
            - module_types_to_mask: a list of modules (or types) that need to be converted to strings

    Returns:
        The updated args with additional attributes and the experiment directory created.
    """
    # Set current date if not provided.
    if args.date == "":
        args.date = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    # Build the full experiment directory path.
    args.exper_dir = os.path.join(
        args.exper_dir,
        args.project_name,
        args.log_dir,
        f"{args.exper_name}-date={args.date}"
    )

    if args.verbose:
        print("This experiment's logs will be located in:", args.exper_dir)

    # Create necessary subdirectories like checkpoints.
    checkpoints_dir = os.path.join(args.exper_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save additional parameters to the args.
    args_dict = vars(args).copy()
    args_dict['module_types_to_mask'] = [str(moduleclass) for moduleclass in args_dict['module_types_to_mask']]

    # Save configuration to a JSON file.
    config_path = os.path.join(args.exper_dir, 'config.json')
    save_dict_to_json(args_dict, config_path)

    return args


################################################################################
# GENERAL PURPOSE UTILITIES
################################################################################

def read_lines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def write_lines(lines, filename):
    with open(filename, 'w') as f:
        f.writelines([line + "\n"  for line in lines])

def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def save_pickle(data, filename):
    with open(filename + ".pickle", 'wb') as fp:
        pickle.dump(data, fp)

def parse_json_to_dict(filename):
    with open(filename, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return data

def parse_json_to_dictlist(filename):
    mydictlist = []
    with open(filename) as f:
        for json_obj in f:
            mydict = json.loads(json_obj)
            mydictlist.append(mydict)
    return mydictlist

def parse_csv_to_dictlist(filename):
    mydictlist = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mydictlist.append(json.loads(json.dumps(row)))
    return mydictlist

def save_dict_to_json(mydict, filename, sort_keys=False):
    json_dict = json.dumps(mydict, sort_keys=sort_keys,  ensure_ascii=False, indent=4) 
    f = open(filename, "w")
    f.write(json_dict)
    f.close()

def save_dictlist_to_json(mydictlist, filename):
    f = open(filename, 'w', encoding='utf-8')
    for mydict in mydictlist:
        json.dump(mydict, f)
        f.write("\n")
    f.close()

def save_dictlist_to_json2(mydictlist, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(mydictlist, f, ensure_ascii=False, indent=4)

def save_dictlist_to_csv(mydictlist, filename):
    header = mydictlist[0].keys()
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(mydictlist)

def str2bool(v):
    # NOTE:
    # taken from
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2tuple(v):
    try:
        row, col = map(int, v.split(','))
        return (row, col)
    except:
        raise argparse.ArgumentTypeError('Expected comma-separated pair of integers.')

def str2moduleclass(v):
    module_class_dict = {
        'gpt2attention': GPT2Attention,
        'gpt2mlp': GPT2MLP, 
        'gpt2block': GPT2Block,
    }
    v = v.lower()
    try: 
        module_class_list = []
        for item in v.split(' '):
            module_class_list.append(module_class_dict[item])
        return module_class_list
    except:
        raise argparse.ArgumentTypeError('Expected valid GPT2 module class names.')

def shuffle_dict(input_dict):
    input_dict = OrderedDict(input_dict)
    data_len = len(list(input_dict.keys()))
    indices = list(range(data_len))
    np.random.shuffle(indices)
    input_dict_list = list(input_dict.items())
    input_dict_list = [input_dict_list[idx] for idx in indices]
    input_dict = OrderedDict(input_dict_list)
    return input_dict

def mycycle(dataloader):
    """
    Helper to infinitely yield batches from dataloader but also to reshuffle 
    whenever 1 epoch is finished. Otherwise, using cycle from itertools 
    will preserve order.
    """
    while True:
        temp_dataloader = dataloader
        for batch in temp_dataloader:
            yield batch
