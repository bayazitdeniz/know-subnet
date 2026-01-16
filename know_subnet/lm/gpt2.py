import torch.nn as nn
import tqdm as tqdm
from typing import List, Type

from transformers import (
    GPT2LMHeadModel,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer
)

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
)

from transformers.models.gpt2.configuration_gpt2 import (
    GPT2Config
)

from know_subnet.lm.lm_mixins import (
    MaskOpsMixin,
    SelectivePruningMixin,
    MaskStatsMixin
)


class GPT2LM(nn.Module, MaskOpsMixin):
    """GPT2 LM Base Model

    Wraps a GPT2LMHeadModel to augment it with:
        - dropout configuration
        - methods for various mask operations
        - methods to freeze/unfreeze parameters
    """
    def __init__(self, use_dropout: bool = False, lm_name: str = 'gpt2'):
        """
        Args:
            use_dropout (bool, optional): Whether to use dropout in the model. Defaults to False.
            lm_name (str, optional): Pretrained model name to load.. Defaults to 'gpt2'.
        """
        super().__init__()
        self.config = GPT2Config.from_pretrained(lm_name)
        if not use_dropout:
            self.config.resid_pdrop=0.0
            self.config.embd_pdrop=0.0
            self.config.attn_pdrop=0.0
            self.config.summary_first_dropout=0.0

        self.lm = GPT2LMHeadModel.from_pretrained(lm_name, config=self.config)

        self.lm_name = lm_name
        self.num_layers = self.lm.config.n_layer
        self.num_heads = self.lm.config.n_head

    def forward(self, input_ids=None, attention_mask=None,  labels=None):
        return self.lm(input_ids, attention_mask=attention_mask, labels=labels)

    def freeze_params(
        self,
        is_freeze: bool = True,
        exclude_name_list: List[str] = ["mask_scores", "multiple_choice_head"],
        verbose: bool = True
    ):
        """
        Freeze or unfreeze model parameters except for those specified in the exclusion list.

        Args:
            is_freeze (bool): If True, freeze parameters; if False, unfreeze them.
            exclude_name_list (List[str]): List of parameter name substrings to exclude from freezing.
            verbose (bool): If True, print the names of parameters as they are frozen/unfrozen.
        """
        for name, param in self.named_parameters():
            if any(exclude_name in name for exclude_name in exclude_name_list):
                continue  # Skip freezing this parameter
            param.requires_grad = not is_freeze
            if verbose:
                if is_freeze:
                    print("Froze {}".format(name))
                else:
                    print("Unfroze {}".format(name))

class GPT2QAModel(nn.Module, MaskOpsMixin):
    """GPT2 QA Base Model

    Wraps a GPT2DoubleHeadsModel to augment it with:
        - dropout configuration
        - tokenization adjustments for special tokens
        - methods for various mask operations
        - methods to freeze/unfreeze parameters
    """
    def __init__(self, use_dropout: bool = False, lm_name: str = 'gpt2', do_freeze: bool = True):
        """
        Args:
            use_dropout (bool, optional): Whether to use dropout in the model. Defaults to False.
            lm_name (str, optional): Pretrained model name to load.. Defaults to 'gpt2'.
            do_freeze (bool): If True, freeze parameters after initialization. Defaults to True.
        """
        super().__init__()
        config = GPT2Config.from_pretrained(lm_name)
        if not use_dropout:
            config.resid_pdrop=0.0
            config.embd_pdrop=0.0
            config.attn_pdrop=0.0
            config.summary_first_dropout=0.0

        self.lm = GPT2DoubleHeadsModel.from_pretrained(lm_name, config=config)

        self.lm_name = lm_name
        self.num_layers = self.lm.config.n_layer
        self.num_heads = self.lm.config.n_head

        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]"})
        
        self.lm.resize_token_embeddings(len(self.tokenizer))
        if do_freeze:
            self.freeze_params()

    def forward(self, input_ids=None, mc_token_ids=None, attention_mask=None, mc_labels=None):
        return self.lm(
            input_ids, 
            mc_token_ids=mc_token_ids, 
            attention_mask=attention_mask, 
            mc_labels=mc_labels
        )
        
    def freeze_params(
        self,
        is_freeze: bool = True,
        exclude_name_list: List[str] = ["mask_scores", "multiple_choice_head"],
        verbose: bool = True
    ):
        """
        Freeze or unfreeze model parameters except for those specified in the exclusion list.

        Args:
            is_freeze (bool): If True, freeze parameters; if False, unfreeze them.
            exclude_name_list (List[str]): List of parameter name substrings to exclude from freezing.
            verbose (bool): If True, print the names of parameters as they are frozen/unfrozen.
        """
        for name, param in self.named_parameters():
            if any(exclude_name in name for exclude_name in exclude_name_list):
                continue  # Skip freezing this parameter
            param.requires_grad = not is_freeze
            if verbose:
                if is_freeze:
                    print("Froze {}".format(name))
                else:
                    print("Unfroze {}".format(name))


class SelectivePrunedGPT2LM(GPT2LM, SelectivePruningMixin, MaskStatsMixin):
    """
    GPT2 LM Pruned Model
    
    Extends the GPT2LM model by selectively pruning layers, 
    applying masks based on specified parameters, and tracking mask statistics.
    """
    def __init__(
        self,
        out_w_per_mask: int,
        in_w_per_mask: int,
        lm_name: str = 'gpt2',
        top_k_layers: int = 12,
        linear_types_to_mask: List[str] = ['c_attn', 'q_attn', 'c_proj', 'c_fc'],
        module_types_to_mask: List[Type] = [GPT2Attention, GPT2MLP, GPT2Block],
        use_dropout: bool = False,
        initial_mask_p: float = 0.88,
        top_limit: int = -1,
        bottom_limit: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            out_w_per_mask (int): Number of output features to mask per parameter.
            in_w_per_mask (int): Number of input features to mask per parameter.
            lm_name (str): Pretrained model name to load. Defaults to 'gpt2'.
            top_k_layers (int): Number of top layers. Defaults to 12.
            linear_types_to_mask (List[str]): List of linear layer types to mask. Defaults to ['c_attn', 'q_attn', 'c_proj', 'c_fc'].
            module_types_to_mask (List[Type]): List of module types to apply masks to. Defaults to [GPT2Attention, GPT2MLP, GPT2Block].
            use_dropout (bool): Whether to use dropout in the model. Defaults to False.
            initial_mask_p (float): Initial probability for applying a mask. Defaults to 0.88.
            top_limit (int): Upper limit for mask pruning (-1 indicates no limit). Defaults to -1.
            bottom_limit (int): Lower limit for mask pruning (-1 indicates no limit). Defaults to -1.
            verbose (bool): If True, enable verbose logging during layer replacement. Defaults to False.
        """
        super().__init__(use_dropout, lm_name)
        
        # Validate input parameters
        self._validate_params(top_k_layers, linear_types_to_mask, module_types_to_mask, top_limit, bottom_limit)
        
        # Replace layers with masked versions
        self.replace_layers_with_masked(
            out_w_per_mask, 
            in_w_per_mask, 
            top_k_layers, 
            linear_types_to_mask, 
            module_types_to_mask, 
            initial_mask_p,
            top_limit,
            bottom_limit,
            verbose
        )

class SelectivePrunedGPT2QAModel(GPT2QAModel, SelectivePruningMixin, MaskStatsMixin):
    """
    GPT2 QA Pruned Model
    
    Extends the GPT2QAModel model by selectively pruning layers, 
    applying masks based on specified parameters, and tracking mask statistics.
    """
    def __init__(
        self,
        out_w_per_mask: int,
        in_w_per_mask: int,
        lm_name: str = 'gpt2',
        top_k_layers: int = 12,
        linear_types_to_mask: List[str] = ['c_attn', 'q_attn', 'c_proj', 'c_fc'],
        module_types_to_mask: List[Type] = [GPT2Attention, GPT2MLP, GPT2Block],
        use_dropout: bool = False,
        initial_mask_p: float = 0.88,
        top_limit: int = -1,
        bottom_limit: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            out_w_per_mask (int): Number of output features to mask per parameter.
            in_w_per_mask (int): Number of input features to mask per parameter.
            lm_name (str): Pretrained model name to load. Defaults to 'gpt2'.
            top_k_layers (int): Number of top layers to consider for masking. Defaults to 12.
            linear_types_to_mask (List[str]): List of linear layer types to mask. Defaults to ['c_attn', 'q_attn', 'c_proj', 'c_fc'].
            module_types_to_mask (List[Type]): List of module types to apply masks to. Defaults to [GPT2Attention, GPT2MLP, GPT2Block].
            use_dropout (bool): Whether to use dropout in the model. Defaults to False.
            initial_mask_p (float): Initial probability for applying a mask. Defaults to 0.88.
            top_limit (int): Upper limit for mask pruning (-1 indicates no limit). Defaults to -1.
            bottom_limit (int): Lower limit for mask pruning (-1 indicates no limit). Defaults to -1.
            verbose (bool): If True, enable verbose logging during layer replacement. Defaults to False.
        """
        super().__init__(use_dropout, lm_name)
        
        # Validate input parameters
        self._validate_params(top_k_layers, linear_types_to_mask, module_types_to_mask, top_limit, bottom_limit)
        
        # Replace layers with masked versions
        self.replace_layers_with_masked(
            out_w_per_mask, 
            in_w_per_mask, 
            top_k_layers, 
            linear_types_to_mask, 
            module_types_to_mask, 
            initial_mask_p,
            top_limit,
            bottom_limit,
            verbose
        )
