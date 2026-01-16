import os
import logging
import pathlib

from pytorch_lightning.loggers import WandbLogger

util_logger = logging.getLogger('meta_knowledge.utils.wandb_utils')

try:
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())
except NotImplementedError:
    pathlib.PosixPath = pathlib.WindowsPath
    WANDB_CACHE = str(pathlib.PosixPath('~/.wandb_cache').expanduser())

def create_wandb_vars(config):
    """Creates special environment variables for trainers and other utilities 
    to use if such configuration values are provided

    :param config: the global configuration values 
    :raises: ValueError 
    """
    if config.wandb_name:
        os.environ["WANDB_NAME"] = config.wandb_name
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity

    if config.wandb_name or config.wandb_project or config.wandb_entity:
        util_logger.info(
            'WANDB settings (options), name=%s, project=%s, entity=%s' %
            (config.wandb_name, config.wandb_project, config.wandb_entity))

def init_wandb_logger(config):
    """Initializes the wandb logger 

    :param config: the global configuration 
    """
    if config.mask != "none":
        if config.mask == "random":
            config.wandb_name += "_mask_random_"
        else: 
            config.wandb_name += "_mask_"

        config.wandb_name += config.mask_name #+ "_" + config.wandb_name
        
    else:
         config.wandb_name += "_full_model"

    config.wandb_name += f"_lr_{config.learning_rate}_wu_{config.warmup_proportion}"
    config.wandb_name += f"_wd_{config.weight_decay}_grad_{config.max_grad_norm}"
    if config.set_cls:
        config.wandb_name += "_[CLS]"
    if not config.freeze:
        config.wandb_name += "_unfreeze"
    config.wandb_name += f"_seed_{config.seed}"
       
    log_model = "all" if config.wandb_checkpoint else False
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name,
        log_model=log_model
    )
    return wandb_logger

def setup_wandb(config):
    """Sets up wandb enviroment variables, downloads datasets, models, etc.. as needed

    :param config: the global configuration 
    :rtype: None 
    """
    if config.wandb_project or config.wandb_entity:
        create_wandb_vars(config)  