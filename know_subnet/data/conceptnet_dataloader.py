from know_subnet.utils import *
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader

from know_subnet.constants import PATH_DICT

from torch.nn import functional as F
from know_subnet.lm.lm_utils import create_uniform_dist


class ConceptNetDataset(Dataset):
    """
    A Dataset class that manages the LAMA ConceptNet subset.

    Args:
        fam_dict_list (list): A list of dictionaries containing the input and label data.
        lm (str, optional): Language model to use for tokenization. Defaults to 'gpt2'.
        is_controlkg (bool, optional): Flag indicating whether the dataset is the ControlKG. Defaults to False.

    Attributes:
        tokenizer: The tokenizer used for tokenization.
        inputs:  Tokenized inputs.
        labels: Tokenized labels.
        mask (torch.Tensor): Mask indicating ignored locations in labels.
        is_controlkg (bool): Flag indicating whether the dataset is ControlKG.
        uniform_kl_dist (torch.Tensor): The uniform distribution used as a reference in the suppression KLDiv loss.
        inputs_str (list): A list of str input sentences.
        labels_str (list): A list of str label sentences.
    """

    def __init__(self, fam_dict_list, lm='gpt2', is_controlkg=False):
        inputs = []
        labels = []

        for instance in fam_dict_list:     
            inp_sent = None
            label = None       
            if lm.startswith('gpt'):
                inp_sent = instance['best_gpt2_input']
                label = instance['best_gpt2_label']
            else:
                NotImplementedError("Only GPT2 is supported for now.")
            
            inputs.append(inp_sent)
            labels.append(label)

        self.inputs_str = inputs
        self.labels_str = labels
        assert len(self.inputs_str) == len(self.labels_str)

        if lm.startswith('gpt'):
            self.tokenizer = GPT2TokenizerFast.from_pretrained(lm)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
            self.labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)["input_ids"]

            final_ids = [len(ids) for ids in self.tokenizer(inputs)["input_ids"]]
            self.mask = torch.ones(self.labels.shape[0], self.labels.shape[1])
            for row, idx in zip(self.mask, final_ids):
                row[idx - 1] = 0
            self.mask = self.mask == 1
            self.labels[self.mask] = -100

            self.is_controlkg = is_controlkg
            if not self.is_controlkg:
                self.uniform_kl_dist = create_uniform_dist(self.labels, lm)
        else:
            NotImplementedError("Only GPT2 is supported for now.")
        
        inp_ids = self.inputs['input_ids']
        if inp_ids.shape[1] != self.labels.shape[1]:
            raise ValueError("The inputs and labels do not match in the second dimension after tokenization.")


    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.inputs.items()}
        item['inputs_str'] = self.inputs_str[idx]
        item['mask'] = self.mask[idx]
        if not(self.is_controlkg):
            item['uniform_kl'] = self.uniform_kl_dist[idx]

        if not(self.labels_str is None):
            item['labels'] = self.labels[idx]
            item['labels_str'] = self.labels_str[idx]
        return item

    def __len__(self):
        return len(self.inputs_str)


def load_conceptnet(
        targetkg_name: str, 
        lm: str,
        train_batch_size: int,
        eval_batch_size: int,
        use_cuda: bool = True,
        is_controlkg: bool = False, 
        is_csqa: bool = False,
        do_train_shuffle: bool = True
    ): 
    """
    Loads the ConceptNet TargetKG or ControlKG dataloaders for training and evaluation.

    Args:
        targetkg_name (str): The name of the TargetKG.
        lm (str): The language model to use for tokenization and reference distribution.
        train_batch_size (int): The batch size for training data.
        eval_batch_size (int): The batch size for evaluation data.
        use_cuda (bool, optional): Whether to use CUDA for GPU acceleration. Defaults to True.
        is_controlkg (bool, optional): Whether to load the ControlKG instead of TargetKG. Defaults to False.
        is_csqa (bool, optional): Whether to load the dataset for the CommonsenseQA task instead. Defaults to False.
        do_train_shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    train_dataset = None
    val_dataset = None
    
    if is_controlkg:
        if is_csqa:
            fam_dict_train_list = parse_json_to_dict(os.path.join(PATH_DICT["csqa_dir"], 'csqa_controlkg_best_format_train.json'))
            fam_dict_val_list = parse_json_to_dict(os.path.join(PATH_DICT["csqa_dir"], 'csqa_controlkg_best_format_val.json'))
        else:      
            fam_dict_train_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_lama_dir"], 'controlkg_train_best_format.json'))
            fam_dict_val_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_lama_dir"], 'controlkg_val_best_format.json'))
        
        print("Train ControlKG len: ", len(fam_dict_train_list))
        print("Val ControlKG len: ", len(fam_dict_val_list))
        
        train_dataset = ConceptNetDataset(
            fam_dict_list=fam_dict_train_list, 
            lm=lm,
            is_controlkg=True
        )
        val_dataset = ConceptNetDataset(
            fam_dict_list=fam_dict_val_list, 
            lm=lm,
            is_controlkg=True
        )
    else:
        fam_dict_list = None
        if is_csqa:
            fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["csqa_dir"], f"{targetkg_name}_targetkg_best_format.json"))
        else:
            fam_dict_list = parse_json_to_dict(os.path.join(PATH_DICT["conceptnet_targetkg"], f"{targetkg_name}_best_format.json"))
        data_len = len(fam_dict_list)
        
        print("Train TargetKG len: ", data_len)
        
        train_dataset = ConceptNetDataset(
            fam_dict_list=fam_dict_list, 
            lm=lm,
            is_controlkg=False
        )
        val_dataset = train_dataset

    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=do_train_shuffle, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        pin_memory=use_cuda, 
        num_workers=0,
        worker_init_fn=seed_worker
    )

    return train_loader, val_loader


def test():
    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    def print_batches(input_dataloader):
        cnt = 0
        for idx, batch in enumerate(input_dataloader):
            print("_" * 60)
            print(f"idx={idx}")
            print(f"cnt={cnt}")
            pp.pprint(batch)
            temp_labels = batch["labels"]
            temp_labels[temp_labels == -100] = input_dataloader.dataset.tokenizer.pad_token_id
            print("Decoded labels:")
            print(input_dataloader.dataset.tokenizer.batch_decode(temp_labels, skip_special_tokens=True))
            print(batch["labels_str"][:2])
            cnt += 1
            if  torch.all(batch["labels"] == 1):
                raise ValueError("This dataloader has 1s as labels!")
            if cnt == 2:
                break
    
    targetkg_name = "fruit"
    lm = "gpt2"

    targetkg_train_loader, targetkg_val_loader = load_conceptnet(
        targetkg_name=targetkg_name,
        lm=lm,
        train_batch_size=250,
        eval_batch_size=250,
        use_cuda=True,
        is_controlkg=False
    )
    print_batches(targetkg_train_loader)
    print_batches(targetkg_val_loader)

    controlkg_train_loader, controlkg_val_loader = load_conceptnet(
        targetkg_name=targetkg_name,
        lm=lm,
        train_batch_size=250,
        eval_batch_size=250,
        use_cuda=True,
        is_controlkg=True
    )
    print_batches(controlkg_train_loader)
    print_batches(controlkg_val_loader)

if __name__ == '__main__':
    test()