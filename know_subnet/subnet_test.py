import os
import glob
import argparse
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import pandas as pd

from know_subnet.utils import *
from know_subnet.metrics import *
from know_subnet.constants import *
from know_subnet.lm.lm_utils import *
from know_subnet.lm.mask import *
from know_subnet.data.wordnet_dataloader import load_wordnet_targetkg, load_wordnet_controlkg
from know_subnet.data.conceptnet_dataloader import load_conceptnet
from know_subnet.data.wikitext_dataloader import (
    gpt2_infer_wiki_ppl, 
    load_wikitext2_test_dataloader
)
from know_subnet.subnet_train_utils import test_mask

def get_config(exper_dir):
    if not os.path.exists(exper_dir):
        print("Could not find experiment directory!")
        exit()
    config_paths = list(glob.iglob('{}/config.json'.format(exper_dir), recursive=True))
    config = None
    for config_path in config_paths:
        config = parse_json_to_dict(config_path)
    print("Found matching config!")
    return config

def get_model_ckpnt_list(exper_dir):
    if not os.path.exists(exper_dir):
        print("Could not find experiment directory!")
        exit()
    checkpoint_paths = list(glob.iglob('{}/checkpoints/ckpt-step=*-*'.format(exper_dir), recursive=True))
    checkpoint_steps = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_title = checkpoint_path.split("/")[-1].split("-")
        checkpoint_step = checkpoint_title[1].split("=")[1]
        checkpoint_steps.append(checkpoint_step)
    return checkpoint_steps

def get_model_and_config(
        exper_dir,
        checkpoint_step,
    ):
    config_paths = list(glob.iglob('{}/config*'.format(exper_dir), recursive=True))
    checkpoint_paths = list(glob.iglob('{}/checkpoints/ckpt-step={}-*'.format(exper_dir, checkpoint_step), recursive=True))
    assert (len(checkpoint_paths) == 1)
    for config_path in config_paths:
        config = parse_json_to_dict(config_path)
    for checkpoint_path in checkpoint_paths:
        checkpoint_title = checkpoint_path.split("/")[-1].split("-")
        checkpoint_step = checkpoint_title[1].split("=")[1]
        checkpoint_sparsity = checkpoint_title[2].split("=")[1]
        model = load_from_checkpoint(config, checkpoint_path, torch.cuda.is_available(), used_accelerator=True)
        model.eval()
    return model, checkpoint_sparsity, config


def get_best_checkpoint_for_random(nfs_dir, log_dir, project_name, kg, seed, best_ckpt_filename):
    # 1) Retrieve best checkpoint steps all (kg, seed) pair from the given file
    project_path = os.path.join(nfs_dir, log_dir, project_name)
    log_path = os.path.join(project_path, "logs")
    best_df = pd.read_csv(os.path.join(project_path, best_ckpt_filename))
    best_ckpnt_dict = {}
    for index, row in best_df.iterrows():
        best_ckpnt_dict["{}-{}".format(row['kg'], row['seed'])] = row['checkpoint step']
    print("Best checkpoints found: ", best_ckpnt_dict)

    # 2) Find the directory for the matching (kg, seed) pair
    config_dict = {}
    cnt = 0
    matching_exper_path = None
    exper_path = None
    for dirname in os.listdir(log_path):
        # print(dirname)
        exper_path = os.path.join(log_path, dirname)
        config_path = os.path.join(exper_path, "config.json")
        config_dict = parse_json_to_dict(config_path)
        # TODO: should we instead use this function? are they the same thing essentially or no?
        # get_config(exper_dir=exper_path) 
        
        args = argparse.Namespace(**config_dict)
        config_kg = config_dict["targetkg_name"]
        if config_dict["kg_type"] == "wordnet":
            config_kg = config_dict["targetkg_name"].split("'")[1]
        if config_dict["kg_type"] == "conceptnet":
            config_kg = config_dict["targetkg_name"]

        if config_kg == kg and config_dict["seed"] == seed:
            matching_exper_path = exper_path
            
            print("-" * 80)
            print(dirname)
            if cnt > 1: 
                print("Found more than 1 instance. Exiting...")
                exit()
            step = str(int(best_ckpnt_dict["{}-{}".format(config_dict["targetkg_name"],config_dict["seed"])]))
            model, checkpoint_sparsity, config = get_model_and_config(exper_path, step)
            model.freeze_params(exclude_name_list=[], verbose=False)
            model.eval()
            cnt += 1
    
    if cnt != 1:
        print("Did not find 1 instance. Exiting...")
        exit()

    return matching_exper_path, config, step

def parse_comma_list(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

def normalize_kg_name(kg_name, kg_type):
    if kg_type == "wordnet" and "'" in kg_name:
        return kg_name.split("'")[1]
    return kg_name

def format_kg_name(kg_name, kg_type, template):
    if kg_type == "wordnet" and "Synset" in template:
        return f"Synset('{kg_name}')"
    return kg_name

def load_best_checkpoint_dict(nfs_dir, log_dir, project_name, best_ckpt_filename):
    project_path = os.path.join(nfs_dir, log_dir, project_name)
    best_df = pd.read_csv(os.path.join(project_path, best_ckpt_filename))
    best_ckpnt_dict = {}
    for index, row in best_df.iterrows():
        best_ckpnt_dict["{}-{}".format(row['kg'], row['seed'])] = row['checkpoint step']
    return best_ckpnt_dict, project_path, os.path.join(project_path, "logs")

def data_loading(config, accelerator): 
    accelerator.print("Loading data...")

    with accelerator.main_process_first():
        targetkg_train_loader = None
        targetkg_val_loader = None
        controlkg_train_loader = None
        controlkg_val_loader = None

        ########################################################################
        # 1) TargetKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if config["kg_type"] == 'wordnet':
            accelerator.print("Loading WordNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader =  load_wordnet_targetkg(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_worse_format=False
            )
            accelerator.print("Loading WordNet TargetKG done.")
        elif config["kg_type"] == 'conceptnet':
            accelerator.print("Loading ConceptNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader = load_conceptnet(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_controlkg=False,
                is_csqa=config["is_csqa"]
            )
            accelerator.print("Loading ConceptNet TargetKG done.")
        
        ########################################################################
        # 2) ControlKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if config["kg_type"] == "wordnet":
            accelerator.print("Loading WordNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_wordnet_controlkg(
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_worse_format=False
            )
            accelerator.print("Loading WordNet ControlKG done.")
        elif config["kg_type"] == "conceptnet":
            accelerator.print("Loading ConceptNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_conceptnet(
                targetkg_name=config["targetkg_name"], 
                lm=config["lm"],
                train_batch_size=config["train_batch_size"],
                eval_batch_size=config["eval_batch_size"],
                is_controlkg=True,
                is_csqa=config["is_csqa"]
            )
            accelerator.print("Loading ConceptNet ControlKG done.")
            
    accelerator.print("Dataset loading and saving done.")
    
    return (
        targetkg_train_loader,
        targetkg_val_loader,
        controlkg_train_loader,
        controlkg_val_loader
    )


@torch.no_grad()
def zeroshot_log_loop(
    model,
    targetkg_val_loader,
    controllm_val_loader,
    controlkg_val_loader,
    lm_name="gpt2",
    epoch=0,
    step=0,
    processed=0,
    accelerator=None,
    verbose=True
):
    model.eval()
    
    log_dict = {
        'epoch': epoch,
        'step': step,
        'processed': processed
    }
    
    loaders_and_prefixes = [
        (targetkg_val_loader, 'targetkg'),
        (controlkg_val_loader, 'controlkg')
    ]
    
    for loader, prefix in loaders_and_prefixes:
        metric_dict = test_mask(
            model=model,
            dataset_loader=loader,
            lm_name=lm_name,
            do_sparsity_calc=False,
            do_rank_calc=True,
            accelerator=accelerator,
            verbose=verbose
        )
        
        # Prefix the keys and update the log dictionary.
        prefixed_metrics = {f"{prefix}-{k}": v for k, v in metric_dict.items()}
        log_dict.update(prefixed_metrics)
        
    log_dict["controllm-ppl"] = gpt2_infer_wiki_ppl(model, controllm_val_loader, is_accelerator=True)
    
    return log_dict

@torch.no_grad()
def validation_log_loop(
    config,
    model,
    log_dict,
    targetkg_val_loader,
    controllm_val_loader,
    controlkg_val_loader,
    accelerator
):
    model.eval()

    # 1) Define the three validation passes.
    loaders_and_prefixes = [
        ("targetkg-",    targetkg_val_loader, True),
        # ("controllm-",   controllm_val_loader, False),
        # ("controlkg-",   controlkg_val_loader, False),
    ]
    # 2) Run the normal passes
    for prefix, loader, do_sparsity in loaders_and_prefixes:
        prefix = f"masked-{prefix}"
        metrics = test_mask(
            model=model,
            dataset_loader=loader,
            do_sparsity_calc=do_sparsity,
            do_rank_calc=False,
            accelerator=accelerator
        )
        metrics = {prefix + k: v for k, v in metrics.items()}
        log_dict.update(metrics)

    # 3) If any of the include_* flags is set, run inverse-mask passes
    if (
        config["include_controllm_loss"]
        or config["include_controlkg_loss"]
        or config["include_targetkg_suppression_loss"]
    ):
        # flip on inverse mask
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=True)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=True)
            
        loaders_and_prefixes = [
            ("targetkg-",    targetkg_val_loader, True),
            ("controlkg-",   controlkg_val_loader, False),
        ]
        # run the same three jobs, but with an "inverse-" prefix on each
        for prefix, loader, do_sparsity in loaders_and_prefixes:
            inv_prefix = f"inverse-{prefix}"
            metrics = test_mask(
                model=model,
                dataset_loader=loader,
                do_sparsity_calc=do_sparsity,
                do_rank_calc=True,
                accelerator=accelerator
            )
            metrics = {inv_prefix + k: v for k, v in metrics.items()}
            log_dict.update(metrics)
            
        log_dict["inverse-controllm-ppl"] = gpt2_infer_wiki_ppl(model, controllm_val_loader, is_accelerator=True)

        # flip back
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=False)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=False)

    return log_dict

def generate_column_order(is_random_baseline):
    columns = ['checkpoint_step', 'checkpoint_sparsity']
    
    for d in ["targetkg", "controlkg", "controllm"]:
        columns += [f"{d}-(inverse - full)-ppl"]
    
    if not is_random_baseline:
        columns += ['targetkg-(inverse - masked)-ppl']
        columns += ['targetkg-(full - masked)-ppl']
        
    for mask_mode in ["full", "inverse"]:
        for d in ["targetkg", "controlkg", "controllm"]:
            columns += [f"{mask_mode}-{d}-ppl"]
    
    if not is_random_baseline:
        columns += ['masked-targetkg-ppl']

    for d in ["targetkg", "controlkg"]:
        for m in ["rank", "prb"]:
            for stat in ["mean", "min", "max"]:
                columns += [f"{d}-(inverse - full)-{m}-{stat}"]

    return columns

def test_subnet_post_train():
    set_seed(42)
    ############################################################################
    parser = argparse.ArgumentParser(description="args for subnet_test.py main func")
    ############################################################################
    # Joint args
    parser.add_argument(
        "--nfs_dir",
        type=str,
        default="",
        help="nfs root dir")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="folder to store training logs")
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="log directory")
    ############################################################################
    # Only subnetwork eval args
    parser.add_argument(
        "--exper_name",
        type=str,
        default="",
        help="experiment name ID/name to separate it from other experiments")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="date of the experiment, will be filled if nothing is passed")
    ############################################################################
    # Only random baseline args
    parser.add_argument(
        "--is_random_baseline",
        type=str2bool,
        default=False,
        help="whether doing random baseline calculation")
    parser.add_argument(
        "--random_baseline_seed",
        type=int,
        default=None,
        help="seed of model whose sparsity will be used for random baseline")
    parser.add_argument(
        "--random_baseline_kg",
        type=str,
        default=None,
        help="kg of model whose sparsity will be used for random baseline")
    ############################################################################
    # Only composition args
    parser.add_argument(
        "--is_composition",
        type=str2bool,
        default=False,
        help="whether to run composition experiments")
    parser.add_argument(
        "--composition_mode",
        type=str,
        default="seed",
        help="composition mode: seed or kg")
    parser.add_argument(
        "--composition_pattern",
        type=str,
        default="intersection",
        help="composition pattern: intersection, union, or floral")
    parser.add_argument(
        "--composition_kg",
        type=str,
        default=None,
        help="target KG for seed composition")
    parser.add_argument(
        "--composition_kg_list",
        type=str,
        default=None,
        help="comma-separated KG list for kg composition")
    parser.add_argument(
        "--composition_seed",
        type=int,
        default=None,
        help="seed for kg composition")
    parser.add_argument(
        "--composition_success_file",
        type=str,
        default="success_best_ckpnts-random=False.csv",
        help="best checkpoint file for composition")
    args, unknown = parser.parse_known_args()
    ############################################################################

    accelerator = Accelerator(
        log_with="wandb"
    )

    composition_eval_kgs = None
    composition_seed = None
    composition_model = None
    composition_checkpoint_sparsity = None
    composition_base_targetkg_name = None

    if args.is_composition:
        if args.composition_mode == "seed":
            if args.composition_kg is None:
                print("composition_kg is required for seed composition. Exiting...")
                exit()
            composition_eval_kgs = [args.composition_kg]
        elif args.composition_mode == "kg":
            if args.composition_seed is None:
                print("composition_seed is required for kg composition. Exiting...")
                exit()
            composition_eval_kgs = parse_comma_list(args.composition_kg_list)
            if len(composition_eval_kgs) != 3:
                print("Expected exactly 3 KGs for composition. Exiting...")
                exit()
            composition_seed = args.composition_seed
        else:
            raise ValueError("Invalid composition_mode. Use 'seed' or 'kg'.")

        best_ckpnt_dict, _, log_path = load_best_checkpoint_dict(
            nfs_dir=args.nfs_dir,
            log_dir=args.log_dir,
            project_name=args.project_name,
            best_ckpt_filename=args.composition_success_file
        )

        combined_models = []
        config = None
        for dirname in os.listdir(log_path):
            exper_path = os.path.join(log_path, dirname)
            if not os.path.isdir(exper_path):
                continue
            config_path = os.path.join(exper_path, "config.json")
            config_dict = parse_json_to_dict(config_path)
            config_kg = normalize_kg_name(config_dict["targetkg_name"], config_dict["kg_type"])
            if args.composition_mode == "seed":
                if config_kg != args.composition_kg:
                    continue
            else:
                if config_dict["seed"] != composition_seed or config_kg not in composition_eval_kgs:
                    continue
            ckpt_key = "{}-{}".format(config_dict["targetkg_name"], config_dict["seed"])
            if ckpt_key not in best_ckpnt_dict:
                fallback_key = "{}-{}".format(config_kg, config_dict["seed"])
                if fallback_key not in best_ckpnt_dict:
                    continue
                ckpt_key = fallback_key
            step = str(int(best_ckpnt_dict[ckpt_key]))
            model, checkpoint_sparsity, config = get_model_and_config(exper_path, step)
            model.freeze_params(exclude_name_list=[], verbose=False)
            model.eval()
            combined_models.append(model)

        if len(combined_models) != 3:
            print("Did not find all 3 models for composition. Exiting...")
            exit()

        if not config["lm"].startswith("gpt"):
            raise NotImplementedError("Only GPT2 model family is supported for now.")

        combined_model = GPT2LM(
            use_dropout=False,
            lm_name=config["lm"]
        )
        combined_model.freeze_params(exclude_name_list=[], verbose=False)
        combined_model.eval()
        checkpoint_sparsity = combined_model.remove_combined_mask(
            models_to_remove=combined_models,
            pattern=args.composition_pattern
        )
        composition_model = accelerator.prepare_model(combined_model, evaluation_mode=True)
        composition_checkpoint_sparsity = float(checkpoint_sparsity)
        composition_base_targetkg_name = config["targetkg_name"]

        save_project_name = (
            f"{args.project_name}_compose_seed"
            if args.composition_mode == "seed"
            else f"{args.project_name}_compose_kg"
        )
        args.project_name = save_project_name
        args.exper_name = f"composition-{args.composition_mode}-{args.composition_pattern}"
        args.date = "composition"
        args.exper_dir = os.path.join(
            args.nfs_dir,
            "logs",
            args.project_name,
            args.log_dir,
            f"{args.exper_name}-date={args.date}"
        )
        os.makedirs(args.exper_dir, exist_ok=True)
    
    if args.is_random_baseline and not args.is_composition:
        applied_seed_dict = {
            84: 231,
            735: 567,
            1318: 815
        }
        applied_seed = applied_seed_dict[args.random_baseline_seed]
        set_seed(applied_seed)

    ############################################################################
    # 1) Get config for data loading and checkpoint steps for model loading
    checkpoint_steps = []
    if args.is_composition:
        checkpoint_steps = [0]
    elif not args.is_random_baseline:
        # for subnet eval, we load the config and checkpoint steps from the experiment directory
        args.exper_dir = os.path.join(args.nfs_dir, "logs")
        args.exper_dir = os.path.join(
            args.exper_dir, 
            args.project_name, 
            args.log_dir,
            f"{args.exper_name}-date={args.date}"
        )
        accelerator.print("Requested exper dir is: ", args.exper_dir)
        accelerator.print("Retrieving config of experiment...")
        config = get_config(args.exper_dir)
        accelerator.print("Retrieval done.")
        
        checkpoint_steps = get_model_ckpnt_list(args.exper_dir)
    else:
        # for random baseline, we first find the best ckpt,
        # then load the config from the experiment directory
        exper_dir, config, checkpoint_step = get_best_checkpoint_for_random(
            nfs_dir=args.nfs_dir,
            log_dir=args.log_dir,
            project_name=args.project_name,
            kg=args.random_baseline_kg,
            seed=args.random_baseline_seed,
            best_ckpt_filename="success_best_ckpnts-random=False.csv"
        )
        args.exper_name = config["exper_name"]
        args.date = config["date"]
        args.exper_dir = exper_dir
        checkpoint_steps = [checkpoint_step]
            
    accelerator.print("Retrieved {} checkpoints in total.".format(len(checkpoint_steps)))
        
    config["train_batch_size"] = 250
    config["eval_batch_size"] = 250
    config["controllm_eval_batch_size"] = 8

    eval_kgs = [None]
    if args.is_composition:
        eval_kgs = composition_eval_kgs

    ############################################################################
    # 4) Setting up the full model for delta calculations
    if not config["lm"].startswith("gpt"):
        raise NotImplementedError("Only GPT2 model family is supported for now.")
    full_model = GPT2LM(
        use_dropout=False,
        lm_name=config["lm"]
    )
    full_model.freeze_params(exclude_name_list=[], verbose=False)
    full_model.eval()
    full_model = accelerator.prepare_model(full_model, evaluation_mode=True)
    
    ############################################################################
    # 6) Start wandb logging into a new project
    wandb_dir = None
    wandb_proj_name = args.project_name + "_test"
    accelerator.init_trackers(
        project_name=wandb_proj_name,
        config=vars(args),
        init_kwargs={"wandb": {
            "reinit": True, 
            "dir": wandb_dir, 
            "name": args.exper_name + "-date=" + args.date
        }}
    )
    keep_columns = generate_column_order(is_random_baseline=(args.is_random_baseline or args.is_composition))
    if args.is_composition and args.composition_mode == "kg":
        keep_columns = ["eval_kg", "composition_seed"] + keep_columns
    res_table = wandb.Table(columns=keep_columns)

    ############################################################################
    # 7) Define some general metric vars
    PPL_DATASETS = ["targetkg", "controlkg", "controllm"]
    RANK_PRB_DATASETS = ["targetkg", "controlkg"]
    TARGETKG_DELTA_PPL_PAIRS = {
        # name           (minuend, subtrahend)
        "(inverse - full)":    ("inverse", "full"),
    }
    if not (args.is_random_baseline or args.is_composition):
        TARGETKG_DELTA_PPL_PAIRS["(full - masked)"] = ("full",    "masked")
        TARGETKG_DELTA_PPL_PAIRS["(inverse - masked)"] = ("inverse", "masked")
        
    OTHERS_DELTA_PPL_PAIRS = {
        # name           (minuend, subtrahend)
        "(inverse - full)":    ("inverse", "full"),
    }
    
    ############################################################################
    # 8) Loop through checkpoints and evaluate
    model = None
    res_dictlist = []
    for eval_kg in eval_kgs:
        if args.is_composition and args.composition_mode == "kg":
            config["targetkg_name"] = format_kg_name(
                eval_kg,
                config["kg_type"],
                composition_base_targetkg_name
            )

        ############################################################################
        # 3) Data loading
        _, targetkg_val_loader, _, controlkg_val_loader = data_loading(config, accelerator)
        
        accelerator.print("-" * 50)
        accelerator.print("Loading wikitext2 as ControlLM...")
        controllm_val_loader = load_wikitext2_test_dataloader(
            lm_name=config["lm"],
            block_size=512,
            controllm_eval_batch_size=config["controllm_eval_batch_size"]
        ) 
        accelerator.print("Loading ControlLM done.")

        targetkg_val_loader, controlkg_val_loader, controllm_val_loader = accelerator.prepare(
            targetkg_val_loader, controlkg_val_loader, controllm_val_loader
        )

        ############################################################################
        # 5) Get full model results
        full_dict = zeroshot_log_loop(
            model=full_model,
            targetkg_val_loader=targetkg_val_loader,
            controllm_val_loader=controllm_val_loader,
            controlkg_val_loader=controlkg_val_loader,
            lm_name=config["lm"],
            epoch=0,
            step=0,
            processed=0,
            accelerator=accelerator,
            verbose=True
        )
        # add "full" prefix to all entries
        full_dict = {f"full-{k}": v for k, v in full_dict.items()}

        ckpnt_loop = tqdm(checkpoint_steps, desc='ckpnt step: ', leave=False, position=1)
        for checkpoint_step in ckpnt_loop:
            if not args.is_composition:
                del model
                accelerator.free_memory() 
                        
                model, checkpoint_sparsity, config = get_model_and_config(args.exper_dir, checkpoint_step)
                model.freeze_params(exclude_name_list=[], verbose=False)
                model.eval()
                model = accelerator.prepare(model)
            else:
                model = composition_model
                checkpoint_sparsity = composition_checkpoint_sparsity
            
            if args.is_random_baseline:
                random_model = GPT2LM(
                    use_dropout=False,
                    lm_name=config["lm"]
                )
                random_model.freeze_params(exclude_name_list=[], verbose=False)
                random_model = accelerator.prepare(random_model)
                random_model.eval()
                if config["params"] == [1,1]:
                    random_model.random_combined_mask([model])
                elif config["params"] == [768, 1]:
                    random_model.random_combined_mask_neuron([model])
                else:
                    raise NotImplementedError(
                        "Only config['params']=[1,1] (weights) and [768,1] (neurons) are supported for random baselines."
                    )
            
            log_dict = {
                'epoch': 0,
                'step': 0,
                'processed': 0,
            }
            
            if args.is_random_baseline or args.is_composition:
                eval_model = random_model if args.is_random_baseline else model
                log_dict = zeroshot_log_loop(
                    model=eval_model,
                    targetkg_val_loader=targetkg_val_loader,
                    controllm_val_loader=controllm_val_loader,
                    controlkg_val_loader=controlkg_val_loader,
                    lm_name=config["lm"],
                    epoch=0,
                    step=0,
                    processed=0,
                    accelerator=accelerator,
                    verbose=True
                )
                log_dict = {f"inverse-{k}": v for k, v in log_dict.items()}
            else:
                log_dict = validation_log_loop(
                    config=config,
                    model=model, 
                    log_dict=log_dict,
                    targetkg_val_loader=targetkg_val_loader,
                    controllm_val_loader=controllm_val_loader,
                    controlkg_val_loader=controlkg_val_loader,
                    accelerator=accelerator
                )

            # Gather PPLs per mask type / dataset
            ppl_dict = {
                "full":   { d: full_dict[f"full-{d}-ppl"]   for d in PPL_DATASETS },
                "inverse":{ d: log_dict[f"inverse-{d}-ppl"] for d in PPL_DATASETS },
            }
            
            if not (args.is_random_baseline or args.is_composition):
                ppl_dict["masked"] = { "targetkg": log_dict["masked-targetkg-ppl"] } # masked only has targetkg
            
            # Probably repetitive but doing it to be safe + to add to step_results
            step_results = {}
            for mask_mode, data_dict in ppl_dict.items():
                for d, value in data_dict.items():
                    step_results[f"{mask_mode}-{d}-ppl"] = value

            # Gather rank and prb metrics per mask type / dataset
            for d in RANK_PRB_DATASETS:
                step_results[f"full-{d}-rank"] = full_dict[f"full-{d}-gold_rank"]
                step_results[f"full-{d}-prb"]  = full_dict[f"full-{d}-gold_prb"]
                step_results[f"inverse-{d}-rank"] = log_dict[f"inverse-{d}-gold_rank"]
                step_results[f"inverse-{d}-prb"]  = log_dict[f"inverse-{d}-gold_prb"]

            # Compute deltas for PPLs
            for d in ["targetkg"]:
                for key, (minuend, subtrahend) in TARGETKG_DELTA_PPL_PAIRS.items():
                    step_results[f"{d}-{key}-ppl"] = ppl_dict[minuend][d] - ppl_dict[subtrahend][d]
            for d in ["controlkg", "controllm"]:
                for key, (minuend, subtrahend) in OTHERS_DELTA_PPL_PAIRS.items():
                    step_results[f"{d}-{key}-ppl"] = ppl_dict[minuend][d] - ppl_dict[subtrahend][d]

            # Compute deltas for ranks/prbs
            for d in RANK_PRB_DATASETS:
                for metric in ["rank", "prb"]:
                    full_value = step_results[f"full-{d}-{metric}"]
                    inv_value  = step_results[f"inverse-{d}-{metric}"]
                    diff_value = inv_value - full_value
                    step_results.update({
                        f"{d}-(inverse - full)-{metric}-mean": diff_value.mean().item(),
                        f"{d}-(inverse - full)-{metric}-min":  diff_value.min().item(),
                        f"{d}-(inverse - full)-{metric}-max":  diff_value.max().item(),
                    })

            if args.is_composition and args.composition_mode == "kg":
                step_results["eval_kg"] = eval_kg
                step_results["composition_seed"] = composition_seed

            # Add checkpoint metadata
            step_results["checkpoint_step"]     = int(checkpoint_step)
            step_results["checkpoint_sparsity"] = float(checkpoint_sparsity)

            # Log to wandb
            accelerator.log(step_results)
            myrowlist = [step_results[c] for c in keep_columns]
            res_table.add_data(*myrowlist)
            
            # Save to CSV locally as the checkpoint loop progresses
            res_dictlist.append(step_results)
            new_res_dictlist = []
            for mydict in res_dictlist:
                new_mydict = {k: mydict[k] for k in keep_columns}
                new_res_dictlist.append(new_mydict)
            save_dictlist_to_csv(
                mydictlist=new_res_dictlist, 
                filename=os.path.join(args.exper_dir, f"res-{args.exper_name}-date={args.date}-random={args.is_random_baseline}.csv"))
    
    accelerator.log({"results": res_table})
    accelerator.end_training()


if __name__ == '__main__':
    test_subnet_post_train()
