import os
import logging
import pytorch_lightning as pl

from pprint import pformat
from omegaconf import OmegaConf
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    GradientAccumulationScheduler
)

from know_subnet.downstream_task.logger import init_wandb_logger

util_logger = logging.getLogger('commonsenseqa.trainer')

def setup_trainer(args) -> pl.Trainer:
    """Sets up the trainer and associated call backs from configuration 

    :param configuration: the target configuration 
    :rtype: a trainer instance 
    """
    mode = "max"
    util_logger.info('mode=%s via %s' % (mode, args.callback_monitor))

    if not os.path.isdir(args.output_dir):
        util_logger.info('making target directory: %s' % args.output_dir)
        os.mkdir(args.output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        monitor=args.callback_monitor,
        mode=mode,
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=True,
    )

    accumulator = GradientAccumulationScheduler(
        scheduling={0: args.gradient_accumulation_steps})
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
        monitor=args.callback_monitor,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=mode
    )

    callbacks = [
        lr_monitor,
        checkpoint_callback,
        early_stop_callback,
        accumulator
    ]

    # train parameters
    train_params = dict(
        devices=find_usable_cuda_devices(args.n_gpu),
        accelerator=args.accelerator,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        precision=32,
        callbacks=callbacks,
        num_sanity_val_steps=4,
        log_every_n_steps=5,
    )

    train_params['logger'] = init_wandb_logger(args)
    train_params['logger'].log_hyperparams(
        OmegaConf.to_container(args, resolve=True)
    )

    util_logger.info(
        "\n===========\n"+pformat(train_params)+"\n==========="
    )

    trainer = pl.Trainer(**train_params)

    return trainer
