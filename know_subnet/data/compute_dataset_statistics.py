import torch

from know_subnet.utils import set_seed, parse_json_to_dict
from know_subnet.constants import *
from know_subnet.metrics import hf_perp_func

from know_subnet.data.wordnet_dataloader import load_wordnet_targetkg, load_wordnet_controlkg
from know_subnet.data.conceptnet_dataloader import load_conceptnet

from know_subnet.lm.gpt2 import GPT2LM

from tqdm import tqdm

wordnet_targetkg_name_list = [
    "Synset('building.n.01')", 
    "Synset('communication.n.02')", 
    "Synset('change.n.01')", 
    "Synset('statement.n.01')",
    "Synset('location.n.01')", 
    "Synset('representation.n.02')",
    "Synset('magnitude.n.01')"
]

conceptnet_targetkg_name_list = [
    "swimming", 
    "fruit", 
    "sun"
]

model_list = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl"
]

@torch.no_grad()
def simple_test(model, dataset_loader, use_cuda=False):
    """
    Runs through one epoch - all testing examples.
    """
    step = 0.0
    tot_loss = 0.0
    tot_hf_perp = 0.0

    test_loop = tqdm(dataset_loader, desc="Evaluating: ")
    model.eval()

    for batch in test_loop:
        # select relevant items from input batch
        input_ids = batch['input_ids'].detach()
        attention_mask = batch['attention_mask'].detach()
        labels = batch['labels'].detach()
        mask = labels != -100
        if use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        # run a pred loop
        output = model(input_ids, attention_mask=attention_mask, labels=labels)

        # select relevant items from the output
        loss = output.loss.detach()
        logits = output.logits.detach()
        labels = labels
        mask = mask

        # Shift the logits so it doesn't have prediction for what comes after the last token.
        # Shit masked_labels_ids and the mask that retrieves the label, as they include the first token (unlike logits).
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        mask = mask[..., 1:].contiguous()
        
        # calc metrics
        hf_perp = hf_perp_func(loss)
        
        # accumulate metrics
        tot_loss += loss
        tot_hf_perp += hf_perp
        step += 1
    
    avg_loss = tot_loss / step
    avg_hf_perp = tot_hf_perp / step
    
    metric_dict = {
        "val_loss": avg_loss.item(),
        "avg_hf_perp": avg_hf_perp
    }
    
    return metric_dict

def get_dataloader(kg_type, targetkg_name, lm, is_targetkg):
    if kg_type == 'wordnet' and is_targetkg:
        return load_wordnet_targetkg(
            targetkg_name=targetkg_name, 
            lm=lm,
            train_batch_size=250,
            eval_batch_size=250,
            is_worse_format=False
        )
    elif kg_type == "wordnet" and not is_targetkg:
        return load_wordnet_controlkg(
            lm=lm,
            train_batch_size=250,
            eval_batch_size=250,
            is_worse_format=False,
            do_train_shuffle=False
        )
    elif kg_type == 'conceptnet' and is_targetkg:
        return load_conceptnet(
            targetkg_name=targetkg_name, 
            lm=lm,
            train_batch_size=250,
            eval_batch_size=250,
            is_controlkg=False,
            is_csqa=False
        )
    elif kg_type == "conceptnet" and not is_targetkg:
        return load_conceptnet(
            targetkg_name=targetkg_name, 
            lm=lm,
            train_batch_size=250,
            eval_batch_size=250,
            is_controlkg=True,
            is_csqa=False,
            do_train_shuffle=False
        )

def get_original_losses(
        kg_type,
        kg_list,
        lm="gpt2", 
        include_controlkg=True, 
        include_targetkg=True
    ):
    full_model = GPT2LM(
        use_dropout=False,
        lm_name=lm
    )
    full_model.freeze_non_masked()
    full_model.cuda()
    targetkg_name = None
    if include_targetkg:
    
        for targetkg_name in kg_list:
            print("-" * 50)
            print("TargetKG: ", targetkg_name)
            _, targetkg_val_loader = get_dataloader(
                kg_type=kg_type,
                targetkg_name=targetkg_name,
                lm=lm,
                is_targetkg=True,
            )
            
            targetkg_metric_dict = simple_test(
                full_model, 
                targetkg_val_loader,
                use_cuda=True
            )
            print(targetkg_metric_dict)
    
    if include_controlkg:
        controlkg_train_loader, controlkg_val_loader = get_dataloader(
            kg_type=kg_type,
            targetkg_name=targetkg_name,
            lm=lm,
            is_targetkg=False,
        )

        print("-" * 50)
        print("Split name: train")
        controlkg_metric_dict = simple_test(
            model=full_model, 
            dataset_loader=controlkg_train_loader, 
            use_cuda=True
        )
        print(controlkg_metric_dict)

        print("-" * 50)
        print("Split name: val")
        controlkg_metric_dict = simple_test(
            model=full_model, 
            dataset_loader=controlkg_val_loader, 
            use_cuda=True
        )
        print(controlkg_metric_dict)
    
    del full_model


def wordnet_analysis():
    # 1) Get TargetKG triplet statistics
    set_seed(42)
    print("#" * 80)
    print("# WordNet Analysis")
    print("#" * 80)
    print("TargetKG statistics:")
    for targetkg_name in wordnet_targetkg_name_list:
        print("-" * 50)
        print("TargetKG: ", targetkg_name)
        print("")
        fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_targetkg"],  f"{targetkg_name}_best_format.json"))
        heads = list([fam_dict["head"] for fam_dict in fam_dict_list])
        tails = list([fam_dict["tail"] for fam_dict in fam_dict_list])
        print("amnt of tuples: ", len(fam_dict_list))
        print("amnt of unique heads: ", len(set(heads)))
        print("amnt of unique tails: ", len(set(tails)))
        print("unqiue tails: ",set(tails))
    
    # 2) Get TargetKG original loss/PPL values
    get_original_losses(
        lm="gpt2", 
        kg_type="wordnet",
        kg_list=wordnet_targetkg_name_list, 
        include_controlkg=False, 
        include_targetkg=True
    )
    
    # 3) Get ControlKG (train/val) triplet statistics
    set_seed(42)
    print("#" * 80)
    print("ControlKG statistics:")
    train_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_train_best_format.json"))
    val_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["wordnet_controlkg"], f"controlkg_val_best_format.json"))[:50]
    train_fam_dict_list = [mydict for mydict in train_fam_dict_list if mydict not in val_fam_dict_list]
    print("-" * 50)
    print("Split name: train")
    print("amnt of tuples: ", len(train_fam_dict_list))
    print("amnt of unique heads: ", len(set([mydict["head"] for mydict in train_fam_dict_list])))
    print("amnt of unique tails: ", len(set([mydict["tail"] for mydict in train_fam_dict_list])))

    print("-" * 50)
    print("Split name: val")
    print("amnt of tuples: ", len(val_fam_dict_list))
    print("amnt of unique heads: ", len(set([mydict["head"] for mydict in val_fam_dict_list])))
    print("amnt of unique tails: ", len(set([mydict["tail"] for mydict in val_fam_dict_list])))

    # 4) Get ControlKG original loss/PPL values
    get_original_losses(
        lm="gpt2", 
        kg_type="wordnet",
        kg_list=wordnet_targetkg_name_list, 
        include_controlkg=True, 
        include_targetkg=False
    )


def conceptnet_analysis():
    # 1) Get TargetKG triplet statistics
    set_seed(42)
    print("#" * 80)
    print("# ConceptNet Analysis")
    print("#" * 80)
    print("TargetKG statistics:")
    for targetkg_name in conceptnet_targetkg_name_list:
        print("-" * 50)
        print("TargetKG: ", targetkg_name)
        print("")
        fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_targetkg"], "{}_best_format.json".format(targetkg_name)))
        heads = list([fam_dict["head"] for fam_dict in fam_dict_list])
        tails = list([fam_dict["tail"] for fam_dict in fam_dict_list])
        rels = list([fam_dict["rel"] for fam_dict in fam_dict_list])

        print("amnt of tuples: ", len(list(fam_dict_list)))
        print("amnt of heads: ", len(heads))
        print("amnt of tails: ", len(tails))
        print("amnt of rels: ", len(rels))
        print("")
        print("amnt of unique unique tuples: ", len(set(["{}-{}-{}".format(fam_dict["head"], fam_dict["tail"], fam_dict["rel"]) for fam_dict in fam_dict_list])))
        print("amnt of unique heads: ", len(set(heads)))
        print("amnt of unique tails: ", len(set(tails)))
        print("amnt of unique rels: ", len(set(rels)))
        
    
    # 2) Get TargetKG original loss/PPL values
    get_original_losses(
        lm="gpt2", 
        kg_type="conceptnet", 
        kg_list=conceptnet_targetkg_name_list, 
        include_controlkg=False, 
        include_targetkg=True
    )
        
    # 3) Get ControlKG (train/val) triplet statistics
    set_seed(42)
    print("#" * 80)
    print("ControlKG statistics:")
    train_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_lama_dir"], 'controlkg_train_best_format.json'))
    val_fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_lama_dir"], 'controlkg_val_best_format.json'))
    
    print("-" * 50)
    print("Split name: train")
    print("amnt of unique tuples: ", len(set(["{}-{}-{}".format(fam_dict["head"], fam_dict["tail"], fam_dict["rel"]) for fam_dict in train_fam_dict_list])))
    print("amnt of unique heads: ", len(set([mydict["head"] for mydict in train_fam_dict_list])))
    print("amnt of unique tails: ", len(set([mydict["tail"] for mydict in train_fam_dict_list])))
    print("amnt of unique rels: ", len(set([mydict["rel"] for mydict in train_fam_dict_list])))
    print("-" * 50)
    print("Split name: val")
    print("amnt of unique tuples: ", len(set(["{}-{}-{}".format(fam_dict["head"], fam_dict["tail"], fam_dict["rel"]) for fam_dict in val_fam_dict_list])))
    print("amnt of unique heads: ", len(set([mydict["head"] for mydict in val_fam_dict_list])))
    print("amnt of unique tails: ", len(set([mydict["tail"] for mydict in val_fam_dict_list])))
    print("amnt of unique rels: ", len(set([mydict["rel"] for mydict in val_fam_dict_list])))

    # 4) Get ControlKG original loss/PPL values
    get_original_losses(
        lm="gpt2", 
        kg_type="conceptnet",
        kg_list=conceptnet_targetkg_name_list, 
        include_controlkg=True, 
        include_targetkg=False
    )


def main():
    set_seed(42)
    wordnet_analysis()
    conceptnet_analysis()


if __name__ == "__main__":
    main()