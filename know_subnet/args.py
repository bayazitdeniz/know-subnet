import argparse
import json

from know_subnet.constants import (
    HYPERPARAMS,
    CONSTANTS
)
from know_subnet.utils import str2bool, str2tuple, str2moduleclass

def get_args(description='arguments for subnet_train.py main func', jupyter=False):
    parser = argparse.ArgumentParser(description=description)

    #######################################################################
    # 1) experiment variables
    #######################################################################
    parser.add_argument(
        "--exper_name",
        type=str,
        default='',
        help="experiment ID/name to separate it from other experiments running on" + \
            " the same device or project")
    parser.add_argument(
        "--project_name",
        type=str,
        default='',
        help="wandb project name")
    parser.add_argument(
        "--kg_type",
        type=str,
        default='wordnet',
        help="the type of KG to run the experiments on, options are: `wordnet` or `conceptnet`")
    parser.add_argument(
        "--targetkg_name",
        type=str,
        default='large',
        help="the name of the TargetKG to run the experiments on")
    parser.add_argument(
        "--is_csqa",
        type=str2bool,
        default=False,
        help="if learning a `conceptnet` type mask and this flag is true, the dataset will be a subset from CSQA rather than a ConceptNet targetkg")
    parser.add_argument(
        "--test_full_model",
        type=str2bool,
        default=False,
        help="True if testing the full model, False if training the mask")
    parser.add_argument(
        "--lm",
        type=str,
        default="gpt2",
        help="name of the language model")
    parser.add_argument(
        "--use_dropout",
        type=str2bool,
        default=False,
        help="a flag for using dropout in the pretrained LM")
    parser.add_argument(
        "--initial_mask_p",
        type=float,
        default=0.45,
        help="Initial mask probability, or in other words, the initial probability of keeping the parameter in the masked subnetwork")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="folder to store training logs / built automatically this is a placeholder")
    parser.add_argument(
        "--nfs_dir",
        type=str,
        default="",
        help="nfs root dir")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="date of the experiment, will be filled if nothing is passed")
    
    #######################################################################
    # 2) basic hyperparam options for training and optimization
    #######################################################################
    parser.add_argument(
        "--seed",
        type=int,
        default=HYPERPARAMS["seed"],
        help="the seed to set for torch, numpy, rand etc.")
    parser.add_argument(
        "--lr",
        type=float,
        default=HYPERPARAMS["learning_rate"],
        help="learning rate for mask parameter optimization")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=HYPERPARAMS["train_batch_size"],
        help="training batch size for verbalized KG datasets")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=HYPERPARAMS["eval_batch_size"],
        help="evaluation batch size for verbalized KG datasets")
    parser.add_argument(
        "--train_epoch",
        type=int,
        default=HYPERPARAMS["train_epoch"],
        help="number of epochs in training")
    parser.add_argument(
        "--lr_warmup_frac",
        type=float,
        default=HYPERPARAMS["lr_warmup_frac"],
        help="fraction of train epoch to warmup mask LR")
    parser.add_argument(
        "--log_step",
        type=int,
        default=HYPERPARAMS["log_step"],
        help="frequency of metric logging in steps")
    parser.add_argument(
        "--save_checkpoint_every",
        type=int,
        default=HYPERPARAMS["save_checkpoint_every"],
        help="frequency of checkpoint saving in epoch")
    parser.add_argument(
        "--min_step_save_checkpoint",
        type=int,
        default=6000,
        help="minimum number of steps before saving a checkpoint")
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="flag for verbosity")
    
    #######################################################################
    # 3) hyperparam options for subnetwork probing
    #######################################################################
    parser.add_argument(
        "--lambda_reg_init",
        type=float,
        default=HYPERPARAMS["lambda_reg_init"],
        help="initial sparsity regularization strength")
    parser.add_argument(
        "--lambda_reg_final",
        type=float,
        default=HYPERPARAMS["lambda_reg_final"],
        help="final sparsity regularization strength")
    parser.add_argument(
        "--mask_lr",
        type=float,
        default=HYPERPARAMS["mask_learning_rate"],
        help="learning rate for the mask params"
    )
    parser.add_argument(
        "--params", 
        type=str2tuple,
        default=(1,1),
        help="Param types to learn scores for where the first int of " + \
            "the tuple is the number of rows and the second int of the tuple is " + \
            " the number of columns per mask, thus for GPT2 small:\n" + \
            "   - (768, 768) masks entire matrices, in that case input 768,768\n" + \
            "   - (768, 1) masks neurons, in that case input 768,1\n" + \
            "   - (1, 1) masks weights, in that case input 1,1"
    )

    #######################################################################
    # 4) options for selective subnetwork probing
    #######################################################################
    parser.add_argument(
        "--top_k_layers",
        type=int,
        default=12,
        help="top k layers to mask"
    )
    parser.add_argument(
        "--top_limit",
        type=int,
        default=-1,
        help="top layer limit to mask, will ignore top_k_layers if set"
    )
    parser.add_argument(
        "--bottom_limit",
        type=int,
        default=-1,
        help="bottom layer limit to mask, will ignore top_k_layers if set"
    )
    parser.add_argument(
        "--linear_types_to_mask",
        type=lambda x: x.split(" "),
        default=[],
        help="types of linear layer, can be: (gpt2) c_attn, q_attn, c_proj, c_fc"
    )
    parser.add_argument(
        "--module_types_to_mask",
        type=str2moduleclass,
        default=[],
        help="types of module to mask, can be: (gpt2) GPT2Attention, GPT2MLP, GPT2Block"
    )
    
    #######################################################################
    # 5) options for multi-objective differentiable mask learning
    #######################################################################
    parser.add_argument(
        "--include_controllm_loss",
        type=str2bool,
        default=True,
        help="a flag for adding a term in the contrastive learning for language modeling")
    parser.add_argument(
        "--controllm_train_batch_size",
        type=int,
        default=HYPERPARAMS["train_batch_size"],
        help="training batch size for ControlLM dataset"
    )
    parser.add_argument(
        "--controllm_eval_batch_size",
        type=int,
        default=HYPERPARAMS["eval_batch_size"],
        help="evaluation batch size for ControlLM dataset"
    )
    parser.add_argument(
        "--include_controlkg_loss",
        type=str2bool,
        default=True,
        help="a flag for adding a term in the contrastive learning for random knowledge")
    parser.add_argument(
        "--include_targetkg_suppression_loss",
        type=str2bool,
        default=True,
        help="a flag for adding a term in the contrastive learning for maximizing loss on specialized knowledge")
    parser.add_argument(
        "--exclude_targetkg_expression_loss",
        type=str2bool,
        default=True,
        help="a flag for not adding the masked subnetwork terms for minimizing loss on specialized knowledge")
    parser.add_argument(
        "--lambda_targetkg_expression",
        type=float,
        default=1.0,
        help="the interpolation weight for TargetKG expression loss"
    )
    parser.add_argument(
        "--lambda_inverse_mask_controllm",
        type=float,
        default=1.0,
        help="the interpolation weight for ControlLM maintenance loss"
    )    
    parser.add_argument(
        "--lambda_inverse_mask_controlkg",
        type=float,
        default=1.0,
        help="the interpolation weight for ControlKG maintenance loss"
    )
    parser.add_argument(
        "--lambda_inverse_mask_targetkg",
        type=float,
        default=1.5,
        help="the interpolation weight for TargetKG suppression loss"
    )
    parser.add_argument(
        "--lambda_scheduler",
        type=str,
        default="constant",
        help="the type of scheduler that should be used for the objective interpolation"
    )
    
    #######################################################################
    # 6) Config file to update default argparse values
    #######################################################################
    parser.add_argument(
        "--subnet_config_file", 
        type=str, 
        default="", 
        help="JSON config file to update default values")
    args = None

    # Settings up arguments for Jupyter if called from a notebook:
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
        # Loading config file to override certain default values and command line args:
        # Priority: defined defaults -> config file -> commandline,
        # So commandline will always overwrite all of them
        if args.subnet_config_file != "":
            json_dict = json.load(open(args.subnet_config_file))
            defaults = {}
            defaults.update(json_dict)
            parser.set_defaults(**defaults)
            args = parser.parse_args() # Recall parse_args to overwrite it

    if args.kg_type == "wordnet":
        args.targetkg_name = "Synset('{}')".format(args.targetkg_name)

    # Setting default masked layer types as all types possible if none are given:
    if len(args.linear_types_to_mask) == 0:
        if args.lm.startswith("gpt2"):
            args.linear_types_to_mask = CONSTANTS["gpt2_linear_types_to_mask"]
        else:
            raise ValueError("No default linear types to mask for the given language model.")    
    if len(args.module_types_to_mask) == 0:
        if args.lm.startswith("gpt2"):
            args.module_types_to_mask = CONSTANTS["gpt2_module_types_to_mask"]
        else:
            raise ValueError("No default module types to mask for the given language model.")

    # NOTE: could change to "weight" or "neuron" since larger models may have diff dim
    #       currently handled under the hood in mask "from_layer" in mask.py
    assert args.params in [(768, 768), (768, 1), (1, 1)], \
        "The params argument should be one of the following: (768, 768), (768, 1), (1, 1)."

    return args
