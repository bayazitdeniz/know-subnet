from itertools import chain
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, default_data_collator
import datasets
datasets.disable_caching()


def group_texts(examples, block_size):
    """
    Concatenates tokenized texts and chunks into blocks of a specified size.

    Args:
        examples: Tokenized inputs.
        block_size (int): The maximum length of each text block.
    
    Returns:
        Dict[str, List[List[int]]]: A dictionary containing the grouped texts (split into chunks of block_size).
            For certain models, a "labels" key is added as a copy of "input_ids".
    """
    # Concatenate all inputs.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the small remainder.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # Copy the same for labels.
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_and_chunk(
        wikitext_dataset,
        column_names,
        tokenizer,
        block_size=None
    ):
    """
    Tokenizes the texts and chunks the token IDs into blocks.

    Args:
        wikitext_dataset (Dataset): The dataset containing the texts to be tokenized and chunked.
        column_names (List[str]): The names of the columns in the dataset.
        tokenizer (Tokenizer): The tokenizer to be used for tokenization.
        block_size (int, optional): The maximum length of each block of token IDs. If not provided, it is determined
            based on the model's context length.

    Returns:
        Dataset: The dataset with tokenized and chunked texts.
    """
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # 1) Decide on block size depending on model context length if None is given
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with by changing this line."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            print(
                f"The max_seq_length passed ({block_size}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    # print(f"Using block size: {block_size}")
    
    # 2) Tokenize texts
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        desc="Running tokenizer on ControlLM dataset"
    )

    # 3) Chunk token IDs into blocks
    controllm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(
            examples=examples, 
            block_size=block_size
        ),
        batched=True,
        num_proc=1,
        desc=f"Grouping texts in chunks of {block_size}"
    )
    
    return controllm_datasets


def load_wikitext2_dataset(
        lm_name: str, 
        block_size=None
    ):
    """
    Preprocesses the WikiText-2 dataset for ControlLM train, val, and test splits.

    Args:
        lm_name (str): The name of the pretrained language model to use for tokenization.
        block_size (Optional[int]): The length of each text block. If not provided, the tokenizer's
            `model_max_length` is used (capped at 1024).

    Returns:
        datasets.DatasetDict: The tokenized and grouped dataset splits.
    """
    # 1) Load the dataset
    wikitext_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # 2) Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    column_names = list(wikitext_dataset["train"].features)
    
    # 3) Tokenize and chunk
    lm_datasets = tokenize_and_chunk(
        wikitext_dataset,
        column_names,
        tokenizer,
        block_size
    )

    # 6) Add tokenizer attribute to the dataset
    lm_datasets["train"].tokenizer = tokenizer
    lm_datasets["validation"].tokenizer = tokenizer
    lm_datasets["test"].tokenizer = tokenizer

    return lm_datasets

def load_wikitext2_dataloader(
        lm_name: str,
        block_size=None,
        train_num: int = 4656,
        val_num: int = 6,
        train_batch_size: int = 6,
        eval_batch_size: int = 6,
        do_train_shuffle: bool = True
    ):
    """
    Loads the WikiText-2 dataloader for ControlLM train and val splits.
    In particular takes a subset of the splits instead of the whole for 
    efficient training and validation.

    Args:
        lm_name (str): The name of the language model for tokenization.
        block_size (Optional[int]): The length of each text block. If not provided, the tokenizer's
            `model_max_length` is used (capped at 1024).
        train_num (int): The number of examples to include in the training subset. Defaults to 4656.
        val_num (int): The number of examples to include in the validation subset. Defaults to 6.
        train_batch_size (int): The batch size for the training dataloader. Defaults to 6.
        eval_batch_size (int): The batch size for the validation dataloader. Defaults to 6.
        do_train_shuffle (bool): Whether to shuffle the training data. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation dataloaders.
    """
    # 1) Load the dataset
    lm_datasets = load_wikitext2_dataset(lm_name, block_size)
    
    # 2) Get train / val splits
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    train_subset = Subset(train_dataset, np.arange(train_num).tolist())
    val_subset = Subset(val_dataset, np.arange(val_num).tolist())

    # 3) Add tokenizer attribute to the subsets
    train_subset.tokenizer = train_dataset.tokenizer
    val_subset.tokenizer = val_dataset.tokenizer

    print(f"Train ControlLM len: {len(train_subset)}")
    print(f"Val ControlLM len: {len(val_subset)}")
    
    # 4) Choose a data collator
    data_collator = default_data_collator

    # 5) Create dataloaders
    train_dataloader = DataLoader(
        train_subset, 
        shuffle=do_train_shuffle, 
        collate_fn=data_collator, 
        batch_size=train_batch_size
    )
    val_dataloader = DataLoader(
        val_subset, 
        shuffle=False,
        collate_fn=data_collator, 
        batch_size=eval_batch_size
    )

    return train_dataloader, val_dataloader


def load_wikitext2_test_dataloader(
        lm_name: str,
        block_size=512,
        controllm_eval_batch_size: int = 8
    ):
    """
    Loads the WikiText-2 dataloader for ControlLM *test split* (not train or val).

    Args:
        lm_name (str): The name of the pretrained language model to use for tokenization.
        block_size (Optional[int]): The length of each text block. If not provided, the tokenizer's
            `model_max_length` is used (capped at 1024).

    Returns:
        datasets.DatasetDict: The tokenized and grouped dataset splits.
    """
    # 1) Load the dataset
    wikitext_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:20%]")
    # print(wikitext_dataset.info.version)
    
    # 2) Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    column_names = list(wikitext_dataset.features)
    
    # 3) Tokenize and chunk
    lm_datasets = tokenize_and_chunk(
        wikitext_dataset,
        column_names,
        tokenizer,
        block_size
    )
    lm_datasets.tokenizer = tokenizer
    
    print("Test ControlLM len: ", len(lm_datasets))

    # 4) Create test dataloader
    test_dataloader = DataLoader(
        lm_datasets, 
        shuffle=False,
        collate_fn=default_data_collator, 
        batch_size=controllm_eval_batch_size
    )

    return test_dataloader

@torch.no_grad()
def gpt2_infer_wiki_ppl(
        model, 
        wiki_dataloader: DataLoader, 
        is_accelerator : bool = False
    ):
    """
    Computes the average perplexity of a GPT2 model on the WikiText dataset.

    Args:
        model: language model to evaluate with
        wiki_dataloader (DataLoader): A PyTorch DataLoader for the WikiText dataset.
        is_accelerator (bool, optional): If True, assumes data is already on the correct device and
            does not perform .cuda() transfers. Defaults to False.

    Returns:
        float: The average perplexity over the dataset.
    """
    sum_ppl = 0.0
    cnt = 0
    for batch in tqdm(wiki_dataloader, desc='wiki: ', leave=False):
        # 1) tokenize input
        if is_accelerator:
            batch["input_ids"] = batch["input_ids"].detach()
            batch["attention_mask"] = batch["attention_mask"].detach()
            tok_labels = batch["labels"].detach()
        else:
            batch["input_ids"] = batch["input_ids"].cuda().detach()
            batch["attention_mask"] = batch["attention_mask"].cuda().detach()
            tok_labels = batch["labels"].cuda().detach()
        
        # 2) get perplexity for each format
        with torch.no_grad():
            output = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=tok_labels)
            loss = output.loss.detach()
        
        sum_ppl += torch.exp(loss).item()
        cnt += 1

    return sum_ppl / float(cnt)

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
            cnt += 1
            if  torch.all(batch["labels"] == 1) or torch.any(batch["labels"] == -100):
                raise ValueError("This dataloader has all 1s or some -100s as labels!")
            if cnt == 10:
                break
    
    lm = "gpt2"
    
    train_dataloader, val_dataloader = load_wikitext2_dataloader(
        lm_name=lm,
        block_size=512
    )
    print_batches(train_dataloader)
    print_batches(val_dataloader)
    
    test_dataloader = load_wikitext2_test_dataloader(
        lm_name=lm,
        block_size=512
    )
    print_batches(test_dataloader)

if __name__ == '__main__':
    test()