import torch
import logging
import pytorch_lightning as pl

from omegaconf import OmegaConf
from transformers import get_linear_schedule_with_warmup
from know_subnet.lm.gpt2 import GPT2QAModel
from know_subnet.data.csqa_dataloader import load_commonsensqa
from know_subnet.downstream_task.load_checkpoint import get_model_and_config

util_logger = logging.getLogger('csqa.runner')

class MCQARunner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams.update(
            OmegaConf.to_container(args))
        if args.checkpoint != "none":
            checkpoint_step=args.checkpoint_step
            self.model, _, _ = get_model_and_config(exper_name=args.checkpoint, checkpoint_step=checkpoint_step, logs_path="")
        else:
            self.model = GPT2QAModel(
                use_dropout=args.use_dropout,
                lm_name=args.lm_name,
                do_freeze=args.freeze
            )

            if args.dataset_name == "csqa":
                if args.mask != "none":
                    cpt_model, _, _ = get_model_and_config(f"checkpoint-{args.mask_name}", 0, args.checkpoint_dir)
                    if args.mask == "random":
                        self.model.random_combined_mask([cpt_model])
                    else:
                        self.model.remove_combined_mask([cpt_model])
            else:
                raise ValueError("Unknown dataset_name: {}".format(args.dataset_name))
        if args.freeze:
            self.model.freeze_params()
        self.dataloader_init(args)
        self.test_count = 0

    def dataloader_init(self, args):
        if args.dataset_name == "csqa":
            train_loader, val_loader, test_loaders = load_commonsensqa(
                data_dir=args.data_dir,
                use_cuda=torch.cuda.is_available(),
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                set_cls=args.set_cls,
                tokenizer=self.model.tokenizer
            )
        else:
            raise ValueError("Unknown dataset_name: {}".format(args.dataset_name))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loaders = test_loaders

        self.validation_step_outputs = []
        self.test_names = [k for k in test_loaders]
        self.test_step_outputs = {k: [] for k in test_loaders}

    def forward(self, input_ids, mc_token_ids, attention_mask=None, mc_labels=None):
        return self.model(
            input_ids=input_ids, 
            mc_token_ids=mc_token_ids, 
            attention_mask=attention_mask,
            mc_labels=mc_labels
        )
    
    def training_step(self, batch, batch_idx):
        output = self.forward(
            input_ids=batch["input_ids"],
            mc_token_ids=batch["mc_token_ids"],
            attention_mask=batch["attention_mask"],
            mc_labels=batch["mc_labels"]
        )
        loss = output.mc_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print_out = batch["print_out"]
        questions = [q for q in print_out["choices"]]
        answers = [a for a in print_out["answer"]]

        output = self.forward(
            input_ids=batch["input_ids"],
            mc_token_ids=batch["mc_token_ids"],
            attention_mask=batch["attention_mask"],
            mc_labels=batch["mc_labels"]
        )
        val_loss = output.mc_loss
        self.log(
            f"val_loss", val_loss, 
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        val_preds = output.mc_logits.argmax(dim=1)
        
        for (pred, answer, question) in zip(val_preds, answers, questions):        
            pred_dict = {
                "pred": pred,
                "answer": answer,
                "question": question
            }
            self.validation_step_outputs.append(pred_dict)
        return val_loss, val_preds
    
    def test_step(self, batch, batch_idx, dataloader_idx): # might need to set the default value to 0
        print_out = batch["print_out"]
        questions = [q for q in print_out["choices"]]
        answers = [a for a in print_out["answer"]]

        output = self.forward(
            input_ids=batch["input_ids"],
            mc_token_ids=batch["mc_token_ids"],
            attention_mask=batch["attention_mask"],
            mc_labels=batch["mc_labels"]
        )
        val_loss = output.mc_loss
        val_preds = output.mc_logits.argmax(dim=1)

        dataset_name = self.test_names[dataloader_idx]
        for (pred, answer, question) in zip(val_preds, answers, questions):        
            pred_dict = {
                "pred": pred,
                "answer": answer,
                "question": question
            }
            self.test_step_outputs[dataset_name].append(pred_dict)
        return val_loss, val_preds
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        all_preds = [x["pred"] for x in outputs]
        all_answers = [x["answer"] for x in outputs]        
        acc = torch.sum(torch.tensor(all_preds) == torch.tensor(all_answers)).item() / len(all_preds)
        self.log(f"val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        metrics = {}
        for dataset_name, outputs in self.test_step_outputs.items():
            all_preds = [x["pred"] for x in outputs]
            all_answers = [x["answer"] for x in outputs]        
            acc = torch.sum(torch.tensor(all_preds) == torch.tensor(all_answers)).item() / len(all_preds)
            metrics[f"test_acc_{dataset_name}"] = acc

        columns = ["dataset", "acc"]
        data = [[k, metrics[k]] for k in metrics]
        self.logger.log_table(
            key=f"metrics-v{self.test_count}", columns=columns, data=data
        )
        self.test_count += 1

        for k in self.test_step_outputs:
            if len(self.test_step_outputs[k]) > 0:
                self.test_step_outputs[k].clear()
    
    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        # update_params = [
        #     p for n, p in self.model.named_parameters() if n == "multiple_choice_head.weight"
        # ]
        # update_bias = [
        #     p for n, p in self.model.named_parameters() if n == "multiple_choice_head.bias"
        # ]

        optimizer_parameters = [
            {
                "params": parameters_first,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]
    
    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler

        """
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = self.hparams.train_batch_size * \
            self.hparams.gradient_accumulation_steps * num_devices

        total_steps = (len(self.train_dataloader().dataset) /
                       effective_batch_size) * self.hparams.num_train_epochs
        self.hparams.warmup_steps = (
            total_steps / effective_batch_size
        ) * self.hparams.warmup_proportion

        util_logger.info(
            'total_steps computed for scheduler: %s, warmup step: %s' % (
                total_steps, str(self.hparams.warmup_steps))
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return scheduler
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return [self.test_loaders[k] for k in self.test_loaders]
