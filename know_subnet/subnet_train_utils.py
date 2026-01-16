from tqdm import tqdm
import torch
import numpy as np
import wandb
import os
import torch.nn.functional as F
from ignite.handlers.param_scheduler import (
    PiecewiseLinear,
    LinearCyclicalScheduler
)

from know_subnet.utils import mycycle
from know_subnet.metrics import ( 
    acc_func,
    hf_perp_func,
    sparsity_func,
    rank_prob_func
)
from know_subnet.lm.lm_utils import save_mask_scores
from know_subnet.lm.gpt2 import GPT2LM

@torch.no_grad()
def test_mask(
        model, 
        dataset_loader, 
        lm_name="gpt2",
        do_sparsity_calc=True,
        do_rank_calc=False, 
        accelerator=None,
        verbose=True
    ):
    """
    Runs through one epoch - all testing examples.
    """
    tot_loss = 0.0
    tot_top1_acc = 0.0
    tot_top5_acc = 0.0
    tot_top10_acc = 0.0
    tot_masked_tokens = 0
    tot_hf_perp = 0.0
    label_rank, label_prb = None, None
    gold_label_ranks, gold_label_prbs = [], []
    
    # Set up progress loop if verbose.
    test_loop = tqdm(enumerate(dataset_loader), desc='eval: ', leave=False, position=2) if verbose else enumerate(dataset_loader)
    model.eval()

    step = 0.0
    for i, batch in test_loop:
        # Extract and detach inputs.
        input_ids = batch['input_ids'].detach()
        attention_mask = batch['attention_mask'].detach()
        labels = batch['labels'].detach()
        mask = labels != -100
        
        # Move data to GPU if needed and not using accelerator.
        if accelerator is None:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
        
        # Run model forward pass.
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss.detach()
        logits = output.logits.detach()
        labels = labels.detach()
        mask = mask.detach()
        if accelerator is None:
            loss = output.loss.cpu()
            logits = output.logits.cpu()
            labels = labels.cpu()
            mask = mask.cpu()

        # If using accelerator, gather the outputs.
        if accelerator is not None:
            loss = accelerator.gather(loss)
            loss = torch.mean(loss)
            logits = accelerator.gather(logits)
            labels = accelerator.gather(labels)
            mask = accelerator.gather(mask)

        # Adjust outputs for autoregressive model by shifting.
        if lm_name.startswith('gpt'):
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            mask = mask[..., 1:].contiguous()
        
        # Compute metrics.
        top1_acc, top5_acc, top10_acc, num_masked_tokens = acc_func(logits, labels, mask)
        hf_perp = hf_perp_func(loss)
        if do_rank_calc:
            label_rank, label_prb = rank_prob_func(logits, labels, mask)
        
        # Accumulate metrics.
        tot_loss += loss
        tot_top1_acc += top1_acc
        tot_top5_acc += top5_acc
        tot_top10_acc += top10_acc
        tot_masked_tokens += num_masked_tokens
        tot_hf_perp += hf_perp
        if do_rank_calc:
            gold_label_ranks.append(label_rank)
            gold_label_prbs.append(label_prb)

        step += 1
    
    # Compute average metrics.
    avg_loss = tot_loss / step
    avg_hf_perp = tot_hf_perp / step
    avg_top1 = tot_top1_acc / tot_masked_tokens
    avg_top5 = tot_top5_acc / tot_masked_tokens
    avg_top10 = tot_top10_acc / tot_masked_tokens
    
    metric_dict = {
        "loss": avg_loss,
        "top1acc": avg_top1,
        "top5acc": avg_top5,
        "top10acc": avg_top10,
        "ppl": avg_hf_perp
    }

    if do_sparsity_calc:
        if accelerator is None:
            sparsity_stats_dict = sparsity_func(model, lm_name=lm_name)
        else:
            sparsity_stats_dict = sparsity_func(accelerator.unwrap_model(model), lm_name=lm_name)
        metric_dict.update(sparsity_stats_dict)

    if do_rank_calc:
        metric_dict.update({
            "gold_rank": torch.cat(gold_label_ranks, dim=0), 
            "gold_prb": torch.cat(gold_label_prbs, dim=0)
        })
    
    return metric_dict

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
    
    # Define the loaders and their corresponding prefixes.
    # NOTE: might make more sense to do controlkg before controllm, 
    #       but original order was this, so keeping it like this for reproducibility
    loaders_and_prefixes = [
        (targetkg_val_loader, 'targetkg'),
        (controllm_val_loader, 'controllm'),
        (controlkg_val_loader, 'controlkg')
    ]
    
    # Process each validation loader.
    for loader, prefix in loaders_and_prefixes:
        metric_dict = test_mask(
            model=model,
            dataset_loader=loader,
            lm_name=lm_name,
            do_sparsity_calc=False,
            do_rank_calc=False,
            accelerator=accelerator,
            verbose=verbose
        )
        
        # Prefix the keys and update the log dictionary.
        prefixed_metrics = {f"{prefix}-{k}": v for k, v in metric_dict.items()}
        log_dict.update(prefixed_metrics)
    
    # Log the results.
    if accelerator is None:
        wandb.log(log_dict)
    else:
        accelerator.log(log_dict)

    return log_dict

def train_log(
        args,
        epoch,
        step,
        processed,
        expression_loss,
        controllm_loss, 
        controlkg_loss,
        targetkg_loss,
        clr_combined_loss,
        clr_params,
        lr_ratio,
        reg,
        lambda_reg,
        optimizer,
        accelerator
    ):
    lambda_expression_log = None if expression_loss is None else clr_params.param_groups["lambda_targetkg_expression"] * expression_loss 
    lambda_reg_log = None if reg is None else lambda_reg * reg 
    lambda_inverse_controllm_log = None if controllm_loss is None else clr_params.param_groups["lambda_inverse_mask_controllm"] * controllm_loss
    lambda_inverse_controlkg_log = None if controlkg_loss is None else clr_params.param_groups["lambda_inverse_mask_controlkg"] * controlkg_loss
    lambda_inverse_targetkg_log = None if targetkg_loss is None else clr_params.param_groups["lambda_inverse_mask_targetkg"] * targetkg_loss
    all_combined = None
    if lambda_reg_log is None and clr_combined_loss is not None:
        all_combined = clr_combined_loss
    elif lambda_reg_log is not None and clr_combined_loss is None:
        all_combined = lambda_reg_log
    else:
        all_combined = lambda_reg_log + clr_combined_loss

    log_dict = {
        'epoch': epoch,
        'step': step,
        'processed': processed,
        'train/lr_ratio': lr_ratio
    }

    log_dict.update({
        'train/reg_val': reg,
        'train/lambda_reg': lambda_reg,
        'train/lambda_reg_*_reg_val': lambda_reg_log
    })
    
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'mask':
            log_dict['train/lr'] = param_group['lr']
    log_dict.update(clr_params.param_groups)

    log_dict.update({
        'train/expression_loss': expression_loss,
        'train/lambda_*_expression_loss': lambda_expression_log,
        #
        'train/inverse_controlkg_train_loss': controlkg_loss,
        'train/lambda_*_inverse_controlkg_train_loss': lambda_inverse_controlkg_log,
        #
        'train/inverse_controllm_train_loss': controllm_loss,
        'train/lambda_*_inverse_controllm_train_loss': lambda_inverse_controllm_log,
        #
        'train/inverse_targetkg_train_loss': targetkg_loss,
        'train/lambda_*_inverse_targetkg_train_loss': lambda_inverse_targetkg_log,
        #
        'train/clr_combined_loss': clr_combined_loss,
        'train/combined_loss_with_reg': all_combined
    })
    
    return log_dict

@torch.no_grad()
def validation_log_loop(
    args,
    model,
    log_dict,
    targetkg_val_loader,
    controllm_val_loader,
    controlkg_val_loader,
    accelerator
):
    model.eval()

    # 1) Define the three validation passes.
    # NOTE: might make more sense to do controlkg before controllm, 
    #       but original order was this, so keeping it like this for reproducibility
    loaders_and_prefixes = [
        ("targetkg-",    targetkg_val_loader, True),
        ("controllm-",   controllm_val_loader, False),
        ("controlkg-",   controlkg_val_loader, False),
    ]

    # 2) Run the normal passes
    for prefix, loader, do_sparsity in loaders_and_prefixes:
        prefix = f"val/{prefix}"
        metrics = test_mask(
            model=model,
            dataset_loader=loader,
            do_sparsity_calc=do_sparsity,
            accelerator=accelerator
        )
        metrics = {prefix + k: v for k, v in metrics.items()}
        log_dict.update(metrics)

    # 3) If any of the include_* flags is set, run inverse-mask passes
    if (
        args.include_controllm_loss
        or args.include_controlkg_loss
        or args.include_targetkg_suppression_loss
    ):
        # flip on inverse mask
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=True)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=True)

        # run the same three jobs, but with an "inverse-" prefix on each
        for prefix, loader, do_sparsity in loaders_and_prefixes:
            inv_prefix = f"val/inverse-{prefix}"
            metrics = test_mask(
                model=model,
                dataset_loader=loader,
                do_sparsity_calc=do_sparsity,
                accelerator=accelerator
            )
            metrics = {inv_prefix + k: v for k, v in metrics.items()}
            log_dict.update(metrics)

        # flip back
        if accelerator is None:
            model.set_is_inverse_mask(is_inverse_mask=False)
        else:
            accelerator.unwrap_model(model).set_is_inverse_mask(is_inverse_mask=False)

    return log_dict


class ContrastiveParameters():
    def __init__(
        self,
        lambda_targetkg_expression,
        lambda_inverse_mask_controllm,
        lambda_inverse_mask_controlkg,
        lambda_inverse_mask_targetkg,
        lambda_reg
    ):
        self.param_groups = {
            "lambda_targetkg_expression":  lambda_targetkg_expression,
            "lambda_inverse_mask_controllm": lambda_inverse_mask_controllm,
            "lambda_inverse_mask_controlkg": lambda_inverse_mask_controlkg,
            "lambda_inverse_mask_targetkg": lambda_inverse_mask_targetkg,
            "lambda_reg": lambda_reg
        }
        self.schedule_lists = {
            "lambda_targetkg_expression":  [],
            "lambda_inverse_mask_controllm": [],
            "lambda_inverse_mask_controlkg": [],
            "lambda_inverse_mask_targetkg": [],
            "lambda_reg": []
        }
        self.timestamp = -1

    def step(self):
        self.timestamp += 1
        for k in self.param_groups.keys():
            self.param_groups[k] = self.schedule_lists[k][self.timestamp]

def set_clr_scheduler(args, accelerator):
    clr_params = ContrastiveParameters(
        lambda_targetkg_expression=args.lambda_targetkg_expression,
        lambda_inverse_mask_controllm=args.lambda_targetkg_expression,
        lambda_inverse_mask_controlkg=args.lambda_targetkg_expression,
        lambda_inverse_mask_targetkg=args.lambda_targetkg_expression,
        lambda_reg=args.lambda_reg_init
    )

    if args.lambda_scheduler not in ["constant", "piecewise1", "piecewise2", "piecewise3equal", "linearcyclical1", "linearcyclical2", "linearcyclical3equal"]:
        accelerator.print("No CLR scheduler was picked. Defaulting to constant.")
        args.lambda_scheduler = "constant"

    if args.lambda_scheduler == "constant": # in this case set the lambda args to be the same
        clr_params.schedule_lists["lambda_targetkg_expression"] = [args.lambda_targetkg_expression] * args.train_epoch
        clr_params.schedule_lists["lambda_inverse_mask_controllm"] = [args.lambda_inverse_mask_controllm] * args.train_epoch
        clr_params.schedule_lists["lambda_inverse_mask_controlkg"] = [args.lambda_inverse_mask_controlkg] * args.train_epoch
        clr_params.schedule_lists["lambda_inverse_mask_targetkg"] = [args.lambda_inverse_mask_targetkg] * args.train_epoch
    else:
        raise NotImplementedError(f"Only constant lambda scheduler is implemented for now.")
    
    # regularization scheduler
    milestone1epoch = 1
    # milestone2epoch = int((1.0 * args.train_epoch) / 4.0)
    milestone3epoch = int((1.0 * args.train_epoch) / 2.0)
    # milestone3epoch = int((3.0 * args.train_epoch) / 4.0)
    milestone4epoch = args.train_epoch
    milestones_values = [
        (milestone1epoch, args.lambda_reg_init),
        # (milestone2epoch, args.lambda_reg_init), 
        # (milestone3epoch, args.lambda_reg_final), 
        (milestone3epoch, args.lambda_reg_init), 
        (milestone4epoch, args.lambda_reg_final)
    ]
    clr_params.schedule_lists["lambda_reg"] = np.array(PiecewiseLinear.simulate_values(
        num_events=args.train_epoch,
        optimizer=clr_params, 
        param_name="lambda_reg", 
        milestones_values=milestones_values
    ))[:,1]
    
    return clr_params


def initialize_training_components(
        args, 
        model, 
        targetkg_train_loader, 
        targetkg_val_loader,
        controlkg_train_loader, 
        controlkg_val_loader,
        controllm_train_loader, 
        controllm_val_loader,
        accelerator
    ):
    # 0) Print hyperparameters
    if args.verbose:
        accelerator.print("lambda_reg_init: {}, lambda_reg_final: {}".format(
            args.lambda_reg_init, args.lambda_reg_final))
        accelerator.print("lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
            args.lr, args.mask_lr, args.lr_warmup_frac, args.train_epoch, args.train_batch_size))
    
    # 1) Prepare (for accelerate) the model
    model = accelerator.prepare(model)
    
    # 2) Group parameters for different learning rates if need be
    mask_params = [p for n, p in model.named_parameters() if 'mask_score' in n and p.requires_grad]
    model_params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    accelerator.print("Mask params are trainable: ", len(mask_params) != 0)
    assert len(mask_params) != 0
    accelerator.print("Model params are frozen: ", len(model_params) == 0)
    assert len(model_params) == 0

    # 3) Setup the optimizer
    optimizer = torch.optim.Adam([
        {'params': mask_params, 'lr': args.mask_lr, 'name': 'mask'},
        ], 
        lr = args.mask_lr)

    # 4) Setup the learning rate scheduler
    total_warmup_steps =  len(targetkg_train_loader) * args.train_epoch * args.lr_warmup_frac
    lr_scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, 
        total_iters=total_warmup_steps, last_epoch=-1, verbose=False)

    # 5) Setup KL Div Loss
    # NOTE: log_target=False means that 
    #       - input logits should be log_softmax-ed
    #       - target logits should be softmax-ed
    kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)

    # 6) Create schedulers for multi-objective loss lambda weighing
    clr_params = set_clr_scheduler(args, accelerator)

    # 7) Prepare (for accelerate) optimizer + lr scheduler + loss lambda scheduler
    optimizer, lr_scheduler, clr_params = accelerator.prepare(
        optimizer, lr_scheduler, clr_params)
    
    # 8) Prepare (for accelerate) dataloaders
    if accelerator is not None:
        targetkg_train_loader, targetkg_val_loader, \
            controlkg_train_loader, controlkg_val_loader, \
            controllm_train_loader, controllm_val_loader = accelerator.prepare(
                targetkg_train_loader, targetkg_val_loader, \
                controlkg_train_loader, controlkg_val_loader, \
                controllm_train_loader, controllm_val_loader
            )
    
    # 9) Create original model as KL div reference
    full_model = None
    if args.lm.startswith("gpt"):
        full_model = GPT2LM(
            use_dropout=False,
            lm_name=args.lm
        )
    else:
        raise NotImplementedError("Only GPT2 is supported for now.")
    full_model.freeze_params(exclude_name_list=[], verbose=False)
    full_model.eval()
    full_model = accelerator.prepare_model(full_model, evaluation_mode=True)
            
    return (
        model, 
        full_model,
        optimizer, 
        lr_scheduler, 
        kl_loss, 
        clr_params, 
        mask_params,
        targetkg_train_loader, 
        targetkg_val_loader, 
        controlkg_train_loader,
        controlkg_val_loader,
        controllm_train_loader,
        controllm_val_loader
    )

def compute_controllm_loss(
        model, 
        full_model, 
        controllm_loop, 
        kl_loss
    ):
    batch = next(controllm_loop)
    lm_input_ids = batch['input_ids']
    lm_attention_mask = batch['attention_mask']
    lm_labels = batch['labels'] # NOTE: complete chunks, don't need masking
    
    # Get output from mask
    controllm_output = model(lm_input_ids, attention_mask=lm_attention_mask, labels=lm_labels)
    controllm_logits = controllm_output.logits
    controllm_logits = controllm_logits.reshape(-1, controllm_logits.shape[-1])
    del controllm_output

    # Get output from original model
    with torch.no_grad():
        lm_input_ids = lm_input_ids.detach()
        lm_attention_mask = lm_attention_mask.detach()
        lm_labels = lm_labels.detach()
        full_controllm_output = full_model(lm_input_ids, attention_mask=lm_attention_mask, labels=lm_labels)
        full_controllm_logits = full_controllm_output.logits.detach()
        full_controllm_logits = full_controllm_logits.reshape(-1, full_controllm_logits.shape[-1])
        del full_controllm_output
    
    # KL loss
    controllm_loss = kl_loss(
        input = F.log_softmax(controllm_logits, dim=-1),
        target = F.softmax(full_controllm_logits, dim=-1)
    )
    del controllm_logits
    del full_controllm_logits
    del batch
    del lm_input_ids
    del lm_attention_mask
    del lm_labels
    
    return controllm_loss

def compute_controlkg_loss(
        model, 
        full_model, 
        controlkg_loop, 
        kl_loss
    ):
    batch = next(controlkg_loop)
    controlkg_input_ids = batch['input_ids']
    controlkg_attention_mask = batch['attention_mask']
    controlkg_labels = batch['labels']
    controlkg_mask = controlkg_labels != -100
    
    # Get output from mask
    controlkg_output = model(controlkg_input_ids, attention_mask=controlkg_attention_mask, labels=controlkg_labels)
    controlkg_logits = controlkg_output.logits
    controlkg_logits = controlkg_logits[controlkg_mask]

    # Get output from original model
    with torch.no_grad():
        controlkg_input_ids = controlkg_input_ids.detach()
        controlkg_attention_mask = controlkg_attention_mask.detach()
        controlkg_labels = controlkg_labels.detach()
        full_controlkg_output = full_model(controlkg_input_ids, attention_mask=controlkg_attention_mask, labels=controlkg_labels)
        full_controlkg_logits = full_controlkg_output.logits.detach()
        full_controlkg_logits = full_controlkg_logits[controlkg_mask]
        del full_controlkg_output
        
    # KL loss
    controlkg_loss = kl_loss(
        input = F.log_softmax(controlkg_logits, dim=-1),
        target = F.softmax(full_controlkg_logits, dim=-1)
    )
    del controlkg_logits
    del full_controlkg_logits
    del batch
    del controlkg_input_ids
    del controlkg_attention_mask
    del controlkg_labels
    
    return controlkg_loss


def combine_losses(accelerator, losses, clr_params, expression_loss=None):
    """
    losses: mapping from loss‐key (e.g. "controllm") to the raw Tensor.
    clr_params.param_groups must contain "lambda_inverse_mask_<key>"
    expression_loss: if not None, will log the masked‐target term
    """
    combined = torch.tensor(0.0, device=next(iter(losses.values())).device)
    sep = "-" * 40

    for key, loss in losses.items():
        lam = clr_params.param_groups[f"lambda_inverse_mask_{key}"]
        accelerator.print(f"Inverse {key.upper():<10} loss:", loss)
        accelerator.print(f"Lambda x Inverse {key.upper():<7} loss:", lam * loss)
        combined = combined + lam * loss
        accelerator.print(sep)

    if expression_loss is not None:
        lam_expr = clr_params.param_groups["lambda_targetkg_expression"]
        accelerator.print("Masked TARGETKG loss:", expression_loss)
        accelerator.print("Lambda x Masked TARGETKG loss:", lam_expr * expression_loss)
        combined = combined + lam_expr * expression_loss
        accelerator.print(sep)

    accelerator.print("Combined loss:", combined)
    accelerator.print(sep)
    return combined


def train_mask(
        args, 
        model, 
        targetkg_train_loader, 
        targetkg_val_loader,
        controlkg_train_loader=None, 
        controlkg_val_loader=None,
        controllm_train_loader=None, 
        controllm_val_loader=None,
        accelerator=None
    ):
    ############################################################################
    # 1) Setup: init training components + prepare with accelerate
    model, full_model, optimizer, lr_scheduler, \
        kl_loss, clr_params, mask_params, \
        targetkg_train_loader, targetkg_val_loader, \
        controlkg_train_loader, controlkg_val_loader, \
        controllm_train_loader, controllm_val_loader = initialize_training_components(
            args, 
            model, 
            targetkg_train_loader, 
            targetkg_val_loader,
            controlkg_train_loader, 
            controlkg_val_loader,
            controllm_train_loader, 
            controllm_val_loader,
            accelerator
        )
    
    ############################################################################
    # 2) Zeroshot eval: on init mask prior to updates
    log_dict = zeroshot_log_loop(
        model=model, 
        lm_name=args.lm,
        targetkg_val_loader=targetkg_val_loader,
        controllm_val_loader=controllm_val_loader,
        controlkg_val_loader=controlkg_val_loader,
        accelerator=accelerator,
        verbose=args.verbose
    )

    ############################################################################
    # 3) Initialize training vars
    step = 0.0
    processed = 0
    controllm_loss = None
    controlkg_loss = None
    targetkg_loss = None
    expression_loss = None
    clr_combined_loss = None
    sep = "-" * 40

    ############################################################################
    # 4) Make cycle iterators for controllm and controlkg loaders
    accelerator.free_memory()
    if args.verbose:
        epoch_loop = tqdm(range(1, args.train_epoch + 1), desc='epoch', leave=False, position=0)
    else:
        epoch_loop = range(1, args.train_epoch + 1)
    if args.include_controllm_loss:
        controllm_loop = mycycle(controllm_train_loader)
    if args.include_controlkg_loss:
        controlkg_loop = mycycle(controlkg_train_loader)    
    
    ############################################################################
    # 5) Main train loop
    for epoch in epoch_loop:
        train_loop = targetkg_train_loader
        clr_params.step()

        for batch in train_loop:
            accelerator.print("_" * 80)
            model.train()
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            mask = labels != -100
            uniform_kl_dist = batch['uniform_kl']
            batch_size = input_ids.shape[0]

            ####################################################################
            # 1) Expression Loss
            if not args.exclude_targetkg_expression_loss:
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                expression_loss = output.loss
                del output
            
            ####################################################################
            # 2) Inverse Mask Losses 
            if any([args.include_controllm_loss,
                    args.include_controlkg_loss,
                    args.include_targetkg_suppression_loss]):
                # inverse the mask
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.set_is_inverse_mask(is_inverse_mask=True)

                inverse_losses = {}
                
                # 2.1) ControlLM maintenance pass on inverse mask
                if args.include_controllm_loss:
                    inverse_losses["controllm"] = compute_controllm_loss(
                        model, 
                        full_model, 
                        controllm_loop, 
                        kl_loss
                    )

                # 2.2) ControlKG maintenance pass on inverse mask
                if args.include_controlkg_loss:
                    inverse_losses["controlkg"] = compute_controlkg_loss(
                        model, 
                        full_model, 
                        controlkg_loop, 
                        kl_loss
                    )

                # 2.3) TargetKG suppression pass on inverse mask
                if args.include_targetkg_suppression_loss:
                    targetkg_output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    targetkg_loss = None
                    targetkg_logits = targetkg_output.logits
                    log_probs = F.log_softmax(targetkg_logits, dim=-1)
                    masked_log_probs = log_probs[mask]
                    
                    # KL loss with uniform distribution
                    inverse_losses["targetkg"] = kl_loss(
                        input = masked_log_probs,
                        target = uniform_kl_dist
                    )
                    del targetkg_output
                    del targetkg_logits
                    del input_ids
                    del attention_mask
                    del labels

                # set mask back to normal
                unwrapped_model.set_is_inverse_mask(is_inverse_mask=False)
                
                clr_combined_loss = combine_losses(
                    accelerator, 
                    inverse_losses, 
                    clr_params, 
                    expression_loss if not args.exclude_targetkg_expression_loss else None
                )
                
                # backward pass on combined loss
                accelerator.backward(clr_combined_loss)
                
            else:
                if not args.exclude_targetkg_expression_loss:
                    lam_expr = clr_params.param_groups["lambda_targetkg_expression"]
                    accelerator.print("Masked TARGETKG loss:", expression_loss)
                    accelerator.print("Lambda x Masked TARGETKG loss:", lam_expr * expression_loss)
                    accelerator.print(sep)
                    # backward pass only on expression loss
                    accelerator.backward(lam_expr * expression_loss)
                
            ####################################################################
            # 3) Sparsity Regularization Loss
            reg = accelerator.unwrap_model(model).compute_total_regularizer()
            lam_reg = clr_params.param_groups["lambda_reg"]
            accelerator.print("Sparsity Reg:".ljust(15), reg)
            accelerator.print("Lambda x Sparsity Reg:".ljust(15), lam_reg * reg)
            accelerator.print(sep)
            # backward pass only on sparsity regularization loss
            accelerator.backward(lam_reg * reg)

            ####################################################################
            # 4) Grad norm + Step on schedulers and optimizers
            mask_grad_norm = None
            mask_grad_norm = accelerator.clip_grad_norm_(mask_params, np.inf)
            accelerator.print("mask grad norm is: ", mask_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            step += 1
            processed += batch_size

            ####################################################################
            # 5) Log all metrics
            
            # detach all losses for logging
            expression_loss = expression_loss if expression_loss is None else expression_loss.detach()
            targetkg_loss = None if targetkg_loss is None else targetkg_loss.detach()
            controlkg_loss = None if controlkg_loss is None else controlkg_loss.detach()
            controllm_loss = None if controllm_loss is None else controllm_loss.detach()
            clr_combined_loss = None if clr_combined_loss is None else clr_combined_loss.detach()
            reg = None if reg is None else reg.detach()

            # convert to Python numbers for logging
            expression_loss_log = expression_loss if expression_loss is None else expression_loss.item()
            targetkg_loss_log = None if targetkg_loss is None else targetkg_loss.item()
            controlkg_loss_log = None if controlkg_loss is None else controlkg_loss.item()
            controllm_loss_log = None if controllm_loss is None else controllm_loss.item()
            clr_combined_loss_log = None if clr_combined_loss is None else clr_combined_loss.item()
            reg_log = None if reg is None else reg.item()

            # do logging or checkpointing
            do_eval = (step == 1) or (step % args.log_step == 0)
            train_log_dict = train_log(
                args=args, 
                epoch=epoch, 
                step=step, 
                processed=processed, 
                expression_loss=expression_loss_log,
                controllm_loss=controllm_loss_log, 
                controlkg_loss=controlkg_loss_log,
                targetkg_loss=targetkg_loss_log,
                clr_combined_loss=clr_combined_loss_log,
                clr_params=clr_params,
                lr_ratio=0.0,
                reg=reg_log,
                lambda_reg=clr_params.param_groups["lambda_reg"],
                optimizer=optimizer,
                accelerator=accelerator
            )
            
            if not do_eval:
                accelerator.log(train_log_dict)
            else:
                log_dict = validation_log_loop(
                    args=args,
                    model=model, 
                    log_dict=train_log_dict,
                    targetkg_val_loader=targetkg_val_loader,
                    controllm_val_loader=controllm_val_loader,
                    controlkg_val_loader=controlkg_val_loader,
                    accelerator=accelerator
                )
                accelerator.log(log_dict)
                # checkpointing
                # NOTE: keep in mind that step=epoch because our dataset is small so if it gets bigger you will have to change this statement
                if (step > args.min_step_save_checkpoint) and (step % args.save_checkpoint_every == 0):
                    save_mask_scores(model, log_dict, os.path.join(args.exper_dir, 'checkpoints'), accelerator=accelerator)

    ############################################################################
    # 6) Final logging and checkpointing before finish
    train_log_dict = train_log(
        args=args, 
        epoch=epoch, 
        step=step, 
        processed=processed, 
        expression_loss=expression_loss_log,
        controllm_loss=controllm_loss_log, 
        controlkg_loss=controlkg_loss_log,
        targetkg_loss=targetkg_loss_log,
        clr_combined_loss=clr_combined_loss_log,
        clr_params=clr_params,
        lr_ratio=0.0,
        reg=reg_log,
        lambda_reg=clr_params.param_groups["lambda_reg"],
        optimizer=optimizer,
        accelerator=accelerator
    )
    log_dict = validation_log_loop(
        args=args,
        model=model, 
        log_dict=train_log_dict,
        targetkg_val_loader=targetkg_val_loader,
        controllm_val_loader=controllm_val_loader,
        controlkg_val_loader=controlkg_val_loader,
        accelerator=accelerator
    )
    save_mask_scores(model, log_dict, os.path.join(args.exper_dir, 'checkpoints'), accelerator=accelerator)
    
    return log_dict, model