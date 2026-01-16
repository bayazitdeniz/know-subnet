import os
import json
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer

CLS_TOKEN = "[CLS]"
PAD_TOKEN = "[PAD]"
LABELS_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

TEST_DATASETS = {
    "original": "test_rand_split.jsonl", 
    "csqa": "csqa_test.jsonl", 
    "csqa1": "csqa1_test.jsonl", 
    "csqa2": "csqa2_test.jsonl"
}

def parse_json_to_dict(filename):
    with open(filename, 'r') as read_file:
        data = [json.loads(line) for line in read_file.readlines()]
    return data

class CommonSenseQADataset(Dataset):
    """
    A Dataset class that manages CommonSenseQADataset.
    """
    def __init__(
            self, 
            filename,
            set_cls=False,
        ):
        self.inputs_str_list = []
        self.labels = []

        # build input strs + labels
        qa_dictlist = parse_json_to_dict(filename)

        for instance in qa_dictlist:
            curr_input_list = []
            question = instance["question"]["stem"]
            choice_list = instance["question"]["choices"]
            # choice_list = []
            new_choice_list = sorted(choice_list, key=lambda d: d['label']) 
            for choice in new_choice_list:
                choice_text = choice["text"]
                if set_cls:
                    choice_text = f"{choice_text} {CLS_TOKEN}"

                curr_input_list.append(f"{question} {choice_text}")

            self.inputs_str_list.append(curr_input_list)
            self.labels.append(LABELS_MAP[instance["answerKey"]])
            # self.labels.append(instance["question"]["answer_solution_num"])

    def __getitem__(self, idx):
        item = {'choices': self.inputs_str_list[idx],}
        if not(self.labels is None):
            item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.inputs_str_list)
    

class CommonSenseQADataLoader():
    def __init__(self, dataset, tokenizer, batch_size, use_cuda, is_training):
        # self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.evaluate = not is_training

        # tokenizer.add_special_tokens(
            # {"cls_token": CLS_TOKEN, "pad_token": PAD_TOKEN})
        tokenizer.pad_token = tokenizer.eos_token

        self.dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=use_cuda,
            collate_fn=self.mc_qa_collator,
        )

    def _tensorize(self, data, max_length=72):
        """Tensorize the input multiple choices"""
        choices = data['choices']
        encoded_choices = [self.tokenizer.encode(s) for s in choices]
        cls_token_location = [
            len(tokens) - 1 for tokens in encoded_choices
        ]
        
        encoded = self.tokenizer(
            choices, 
            padding="max_length", 
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return encoded.input_ids, encoded.attention_mask, cls_token_location

    def mc_qa_collator(self, batch):
        """
        Batch collator for multiple choice QA

        :param batch: an incoming batch
        """

        all_choices = []
        for data in batch:
            all_choices.extend(data['choices'])
        encoded_batch_choices = [self.tokenizer.encode(s) for s in all_choices]
        max_length = max([len(s) for s in encoded_batch_choices])

        tensorized = [self._tensorize(data, max_length) for data in batch]
        input_ids_batch = [x[0] for x in tensorized]
        attention_mask_batch = [x[1] for x in tensorized]
        cls_location_batch = [x[2] for x in tensorized]

        questions = [data['choices'] for data in batch]
        answers = [data['label'] for data in batch]
        mc_labels = torch.tensor(answers, dtype=torch.long)
        print_out = {
            # "guid": [data['guid'] for data in batch],
            "choices": questions,
            "answer": answers,
        }

        feature = {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "mc_token_ids": torch.tensor(cls_location_batch, dtype=torch.long),
            "mc_labels": mc_labels,
            "print_out": print_out,
            "evaluate": self.evaluate
        }

        return feature


def load_commonsensqa(
        data_dir,
        use_cuda,
        train_batch_size,
        eval_batch_size,
        set_cls=False,
        tokenizer=None):

    train_dataset = CommonSenseQADataset(
        filename=os.path.join(
        data_dir, "train_rand_split.jsonl"),
        set_cls=set_cls,
    )
    val_dataset_rand = CommonSenseQADataset(
        filename=os.path.join(
        data_dir, "dev_rand_split.jsonl"),
        set_cls=set_cls,
    )
    test_datasets = {d_name: CommonSenseQADataset(
        filename=os.path.join(
        data_dir, TEST_DATASETS[d_name]),
        set_cls=set_cls,
    ) for d_name in TEST_DATASETS}
    
    for d_name in test_datasets:
        print(f"# examples in {d_name}: ", len(test_datasets[d_name]))

    train_loader = CommonSenseQADataLoader(
        train_dataset,
        tokenizer,
        train_batch_size,
        use_cuda,
        is_training=True,
    ).dataloader

    val_loader = CommonSenseQADataLoader(
        val_dataset_rand,
        tokenizer,
        eval_batch_size,
        use_cuda,
        is_training=False,
    ).dataloader

    test_loaders = {d_name: CommonSenseQADataLoader(
        test_dataset,
        tokenizer,
        eval_batch_size,
        use_cuda,
        is_training=False,
    ).dataloader for d_name, test_dataset in test_datasets.items()}

    return train_loader, val_loader, test_loaders


def main():
    train_loader, val_loader, test_loaders = load_commonsensqa(
        data_dir="./data/csqa",
        use_cuda=torch.cuda.is_available(),
        train_batch_size=4,
        eval_batch_size=4,
        tokenizer=AutoTokenizer.from_pretrained("gpt2")
    )
    
    batch = next(iter(val_loader))
    print(f"batch: {batch}")
    print(f"batch['input_ids'].size(): {batch['input_ids'].size()}")
    print(f"batch['mc_token_ids'].size(): {batch['mc_token_ids'].size()}")
    print(f"batch['mc_labels'].size(): {batch['mc_labels'].size()}")
    

if __name__ == '__main__':
    main()
