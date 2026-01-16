import os
import time
import wandb
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
# os.environ["WANDB_MODE"] = "offline"

# LM IMPORTS
from know_subnet.lm.lm_utils import load_lm

# DATASET IMPORTS
from know_subnet.data.wordnet_dataloader import (
    load_wordnet_targetkg,
    load_wordnet_controlkg
)
from know_subnet.data.conceptnet_dataloader import load_conceptnet
from know_subnet.data.wikitext_dataloader import load_wikitext2_dataloader

# EXPERIMENT CONFIG IMPORTS
from know_subnet.args import get_args
from know_subnet.utils import (
    write_lines,
    set_seed_and_device,
    build_main_log_paths,
    create_experiment_folder,
)

# EXPERIMENT RUNNER IMPORTS
from know_subnet.subnet_train_utils import (
    train_mask,
    zeroshot_log_loop
)

def data_loading(args, accelerator=None):
    if args.verbose:
        accelerator.print("Loading data...")

    with accelerator.main_process_first():
        targetkg_train_loader = None
        targetkg_val_loader = None
        controlkg_train_loader = None
        controlkg_val_loader = None
        controllm_train_loader = None
        controllm_val_loader = None

        ########################################################################
        # 1) TargetKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if args.kg_type == 'wordnet':
            accelerator.print("Loading WordNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader =  load_wordnet_targetkg(
                targetkg_name=args.targetkg_name, 
                lm=args.lm,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                is_worse_format=False
            )
            accelerator.print("Loading WordNet TargetKG done.")
        elif args.kg_type == 'conceptnet':
            accelerator.print("Loading ConceptNet TargetKG...")
            targetkg_train_loader, targetkg_val_loader = load_conceptnet(
                targetkg_name=args.targetkg_name, 
                lm=args.lm,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                is_controlkg=False,
                is_csqa=args.is_csqa
            )
            accelerator.print("Loading ConceptNet TargetKG done.")
        
        ########################################################################
        # 2) ControlKG loading
        ########################################################################
        accelerator.print("-" * 50)
        if args.kg_type == "wordnet":
            accelerator.print("Loading WordNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_wordnet_controlkg(
                lm=args.lm,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                is_worse_format=False
            )
            accelerator.print("Loading WordNet ControlKG done.")
        elif args.kg_type == "conceptnet":
            accelerator.print("Loading ConceptNet ControlKG...")
            controlkg_train_loader, controlkg_val_loader = load_conceptnet(
                targetkg_name=args.targetkg_name, 
                lm=args.lm,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                is_controlkg=True,
                is_csqa=args.is_csqa
            )
            accelerator.print("Loading ConceptNet ControlKG done.")
            
        ########################################################################
        # 3) ControlLM loading
        ########################################################################
        accelerator.print("-" * 50)
        accelerator.print("Loading wikitext2 as ControlLM...")
        controllm_train_loader, controllm_val_loader = load_wikitext2_dataloader(
            lm_name=args.lm,
            block_size=512,
            # mlm_probability=0.15,
            val_num=args.controllm_eval_batch_size,
            train_batch_size=args.controllm_train_batch_size, 
            eval_batch_size=args.controllm_eval_batch_size
        )
        accelerator.print("Loading ControlLM done.")

        ########################################################################
        # 4) Log datasets
        ########################################################################
        accelerator.print("-" * 50)
        accelerator.print("Logging dataset for quality check...")
        for dataset_name, dataloader in [
            ("targetkg-train", targetkg_train_loader),
            ("targetkg-val", targetkg_val_loader),
            ("controlkg-train", controlkg_train_loader),
            ("controlkg-val", controlkg_val_loader),
            ("controllm-train", controllm_train_loader),
            ("controllm-val", controllm_val_loader),

        ]:
            data = {}
            if dataset_name.startswith("controllm"):
                # if wikitext2, the dataset is a subset of a dataset, so need to do extra things
                data = {
                    'inputs_str': [],
                    'labels_str': []
                }
                
                for batch in dataloader:
                    temp_labels = batch["labels"].clone().detach()
                    temp_labels[temp_labels == -100] = dataloader.dataset.tokenizer.pad_token_id
                    data['inputs_str'].extend(dataloader.dataset.dataset.tokenizer.batch_decode(batch["input_ids"].clone().detach(), skip_special_tokens=False))
                    data['labels_str'].extend(dataloader.dataset.dataset.tokenizer.batch_decode(temp_labels, skip_special_tokens=False))
            else:
                dataset = dataloader.dataset
                data = {
                    'inputs_str': dataset.inputs_str,
                    'labels_str': dataset.labels_str
                }
            
            df = pd.DataFrame.from_dict(data)
            dataset_table = wandb.Table(dataframe=df)
            if accelerator is not None:
                accelerator.log({dataset_name: dataset_table})
            else:
                dataset_table_artifact = wandb.Artifact(dataset_name, type="dataset")
                dataset_table_artifact.add(dataset_table, dataset_name + "-table")
                wandb.log({dataset_name: dataset_table})
                wandb.log_artifact(dataset_table_artifact)
            
    if args.verbose:
        if accelerator is None:
            print("Dataset loading and saving done.")
        else:
            accelerator.print("Dataset loading and saving done.")
    
    return (
        targetkg_train_loader,
        targetkg_val_loader,
        controlkg_train_loader,
        controlkg_val_loader,
        controllm_train_loader, 
        controllm_val_loader
    )

def main():
    # 1) Loading
    args = get_args()
    args = set_seed_and_device(args)
    args = build_main_log_paths(args)
    # accelerator = None
    accelerator = Accelerator( log_with="wandb")
    
    # 2) Running experiments
    # [0] setting up logging + saving config
    time_start = time.time()
    args = create_experiment_folder(args)

    wandb_dir = None
    if args.nfs_dir != '':
        wandb_dir = os.path.join(args.nfs_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
    if accelerator is not None: 
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={"wandb": {
                "reinit": True, 
                "dir": wandb_dir, 
                "name": args.exper_name + "-date=" + args.date
            }}
        )
    else:
        run = wandb.init(finish_previous=True, project=args.project_name, dir=wandb_dir)
        wandb.run.name = args.exper_name + "-date=" + args.date
        wandb.config.update(args)

    # [1] load lms and save if randomly masked
    targetkg_train_loader, targetkg_val_loader, \
        controlkg_train_loader, controlkg_val_loader, \
        controllm_train_loader, controllm_val_loader = data_loading(args, accelerator)
    
    # [2] loading lang model
    model = load_lm(args)

    # [3] train/test if needed
    if not args.test_full_model:
        last_log_dict, model = train_mask(
            args=args, 
            model=model, 
            targetkg_train_loader=targetkg_train_loader, 
            targetkg_val_loader=targetkg_val_loader,
            controlkg_train_loader=controlkg_train_loader, 
            controlkg_val_loader=controlkg_val_loader,
            controllm_train_loader=controllm_train_loader, 
            controllm_val_loader=controllm_val_loader,
            accelerator=accelerator
        )
    else:
        last_log_dict = zeroshot_log_loop(
            model=model, 
            lm_name=args.lm,
            targetkg_val_loader=targetkg_val_loader,
            controllm_val_loader=controllm_val_loader,
            controlkg_val_loader=controlkg_val_loader,
            accelerator=accelerator,
            verbose=args.verbose
        )
    
    time_passed = (time.time() - time_start)
    write_lines(["Time passed in minutes: {}".format(time_passed / 60.0)], os.path.join(args.exper_dir, 'time.txt'))
    if accelerator is not None:
        accelerator.end_training()

if __name__ == '__main__':
    main()
