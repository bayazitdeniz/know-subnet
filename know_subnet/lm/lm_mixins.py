import numpy as np
import torch
import math
import random
import copy

from itertools import product, groupby
from typing import Dict, List, Optional, Any, Type, Callable, Tuple

from transformers.modeling_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
)
from know_subnet.lm.mask import (
    MaskedHuggingfaceConv1D
)

class MaskOpsMixin:
    """
    Mixin class that provides utility functions for mask operations on LMs.
    """
    
    @staticmethod
    def get_pattern(pattern: str, mask1: torch.Tensor, mask2: torch.Tensor, mask3: torch.Tensor):
        """
        Combines three masks based on the specified pattern.

        Parameters:
            pattern (str): The pattern to apply ('intersection', 'union', or 'floral').
            mask1 (torch.Tensor): The first mask.
            mask2 (torch.Tensor): The second mask.
            mask3 (torch.Tensor): The third mask.

        Returns:
            torch.Tensor: The combined mask.
        """
        if pattern == "intersection":
            return torch.logical_and(torch.logical_and(mask1, mask2), mask3)
        elif pattern == "union":
            return torch.logical_or(torch.logical_or(mask1, mask2), mask3)
        elif pattern == "floral":
            intersection = torch.logical_and(torch.logical_and(mask1, mask2), mask3)
            petal1 = torch.logical_and(torch.logical_and(mask1, mask2), torch.logical_not(mask3)) # '110'
            petal2 = torch.logical_and(torch.logical_and(mask2, mask3), torch.logical_not(mask1)) # '011'
            petal3 = torch.logical_and(torch.logical_and(mask1, mask3), torch.logical_not(mask2)) # '101'
            return torch.logical_or(torch.logical_or(torch.logical_or(intersection, petal1), petal2), petal3)
        else:
            raise ValueError("Invalid pattern provided: " + pattern)

    @staticmethod
    @torch.no_grad()
    def get_combined_masks(models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
        """
        Computes the unioned masks from a list of models 
        (where each have modules of type MaskedHuggingfaceConv1D).

        Parameters:
            models (List[torch.nn.Module]): A list of models fromn which to extract masks.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping module names to their unioned mask.
        """
        mask_dict = {}
        for model_id, model in enumerate(models):
            model.eval()
            model_masks = {
                k: v.produce_mask_reshaped() > 0.5
                for k, v in model.named_modules()
                if type(v) == MaskedHuggingfaceConv1D
            }
            mask_dict[model_id] = model_masks

        unioned_masks = mask_dict[0]
        for i in range(1, len(models)):
            unioned_masks = {
                k: torch.logical_or(unioned_masks[k].data, mask_dict[i][k].data)
                for k in unioned_masks.keys()
            }
        return unioned_masks

    @staticmethod
    def get_combined_masks_with_pattern(
            models: List[torch.nn.Module], 
            pattern: str
        ) -> Dict[str, torch.Tensor]:
        """
        Computes a combined mask using a specified pattern.

        Args:
            models (List[torch.nn.Module]): List of models with masked layers.
            pattern (str): The pattern to apply ('intersection', 'union', or 'floral').

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping module names to their combined masks.
        """
        if len(models) != 3:
            raise ValueError("Can only combine exactly 3 model masks in a pattern.")
        
        mask_dict = {}
        for model_id, model in enumerate(models):
            model.eval()
            model_masks = {
                k: v.produce_mask_reshaped() > 0.5
                for k, v in model.named_modules()
                if type(v) == MaskedHuggingfaceConv1D
            }
            mask_dict[model_id] = model_masks

        combined_masks = {}
        for mod_name in mask_dict[0].keys():
            combined_masks[mod_name] = MaskOpsMixin.get_pattern(
                pattern,
                mask_dict[0][mod_name],
                mask_dict[1][mod_name],
                mask_dict[2][mod_name]
            )
        return combined_masks

    def remove_combined_mask(
            self, 
            models_to_remove: List[torch.nn.Module], 
            pattern: Optional[str] = None
        ) -> float:
        """
        Removes masked weights from the model using either union or a specific 
        pattern and specifies the percentage removed.

        Args:
            models_to_remove (List[torch.nn.Module]): List of models with masked layers.
            pattern (Optional[str], optional): The pattern to apply ('intersection', 'union', or 'floral') when removing combined masks.
                If None, uses union. Defaults to None.

        Returns:
            float: The percentage of weights removed (as a fraction of total).
        """
        modules_dict = dict(self.named_modules())
        
        # Get combined masks
        if pattern is None:
            combined_masks = MaskOpsMixin.get_combined_masks(models_to_remove)
        else:
            combined_masks = MaskOpsMixin.get_combined_masks_with_pattern(models_to_remove, pattern)
            
        # Remove the combined masks
        tot_zeros, tot_size = 0.0, 0.0
        for module_name, curr_mask in combined_masks.items():
            v = curr_mask.float().cpu().numpy().flatten()
            tot_size += v.size
            tot_zeros += (v == 0.0).sum()
            modules_dict[module_name].weight.data *= (~curr_mask).float()
        
        return tot_zeros / tot_size

    def random_combined_mask(
        self, models_to_remove: List[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly removes at the same weight sparsity as the unioned mask 
        for each module in the given models.

        Args:
            models_to_remove (List[torch.nn.Module]): List of models with masked layers.

        Returns:
            Dict[str, torch.Tensor]: Updated boolean masks for each module.
        """
        modules_dict = dict(self.named_modules())
        unioned_masks = MaskOpsMixin.get_combined_masks(models_to_remove)
        returned_masks = {}
        
        for module_name, curr_mask in unioned_masks.items():
            num_removed = curr_mask.sum().item()
            
            if num_removed == 0:
                # no weights removed => just keep everything
                new_mask = torch.zeros_like(curr_mask, dtype=torch.bool)
                returned_masks[module_name] = new_mask
                continue
            
            rows, cols = curr_mask.shape
            indices = list(product(range(rows), range(cols)))
            selected = torch.LongTensor(random.sample(indices, int(num_removed)))
            
            # new_mask = (torch.zeros_like(curr_mask) > 0.0)
            new_mask = torch.zeros_like(curr_mask, dtype=torch.bool)
            row_indices = selected[:, 0]
            col_indices = selected[:, 1]
            new_mask[row_indices, col_indices] = True
            assert new_mask.sum().item() == num_removed
            
            to_keep_mask = (~new_mask).float().to(curr_mask.device)
            modules_dict[module_name].weight.data *= to_keep_mask
            returned_masks[module_name] = new_mask
        
        return returned_masks

    def random_combined_mask_neuron(
        self, models_to_remove: List[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly removes *input* neurons at the same neuron sparsity as the unioned mask 
        for each module in the given models. This requires removing removes 
        entire rows in the parameter as they map to input neurons.
        
        Args:
            models_to_remove (List[torch.nn.Module]): List of models with masked layers.

        Returns:
            Dict[str, torch.Tensor]: Updated boolean masks for each module.
        """
        model_named_modules = dict(self.named_modules())
        unioned_masks = self.get_combined_masks(models_to_remove)
        returned_boolean_masks = copy.deepcopy(unioned_masks)
        
        for module_name, curr_mask in unioned_masks.items():
            num_rows, num_cols = curr_mask.shape

            # Calculate row sparsity (percentage of weights already masked per row)
            row_sparsity = curr_mask.sum(dim=1) / num_cols

            # Compute overall sparsity level across all rows
            total_sparsity = row_sparsity.mean().item()
            
            # Determine number of rows to remove based on total sparsity
            num_rows_to_remove = int(total_sparsity * num_rows)
            if num_rows_to_remove == 0:
                # No rows to remove, we copied the unioned_masks so continue to next module
                continue

            # Select rows randomly to be removed
            rows_to_remove = random.sample(range(num_rows), num_rows_to_remove)

            # Create a new mask where selected rows are masked (False)
            new_mask = torch.ones_like(curr_mask, dtype=torch.bool)
            new_mask[rows_to_remove, :] = False 

            # Calculate the mask to keep weights
            to_keep_mask = new_mask.float().to(curr_mask.device)

            # Apply the mask to the model's weights
            model_named_modules[module_name].weight.data *= to_keep_mask
            
            # Update the returned mask
            returned_boolean_masks[module_name] = new_mask
            
        return returned_boolean_masks
    
    def random_remove_more(
            self, 
            boolean_masks: Dict[str, torch.Tensor], 
            pct_goal: float
        ) -> Dict[str, torch.Tensor]:
        """
        Removes additional parameters to reach a target sparsity percentage.

        Args:
            boolean_masks (Dict[str, torch.Tensor]): Dictionary of current boolean masks.
            pct_goal (float): Target sparsity level.

        Returns:
            Dict[str, torch.Tensor]: Updated boolean masks after additional removal.
        """
        # 1) Get the current sparsity level and decide if we can remove more
        model_named_modules = dict(self.named_modules())
        tot_mask_size = sum(mask.sum().item() for mask in boolean_masks.values())
        tot_param_size = sum(mask.numel() for mask in boolean_masks.values())

        sparsity = 1 - (tot_mask_size / tot_param_size)
        if pct_goal >= sparsity:
            raise ValueError(f"pct_goal ({pct_goal}) is higher than current sparsity ({sparsity}). Use random_add_more instead.")

        # 2) Calculate how many more parameters need to be removed
        tot_remaining_size = tot_param_size - tot_mask_size
        amount_to_remove = math.ceil((sparsity - pct_goal) * tot_param_size)
        amount_to_remove = min(amount_to_remove, tot_remaining_size)
        
        # 3) Randomly select parameters to remove from the ones that are still kept
        all_remaining_idx = []
        for module_name, curr_mask in boolean_masks.items():
            remaining_idx_list = [[module_name] + idx.tolist() for idx in (~curr_mask).nonzero(as_tuple=False)]
            all_remaining_idx.extend(remaining_idx_list)
        to_mask_idx_list = random.sample(all_remaining_idx, amount_to_remove)
        
        # 4) Group these by module and apply the removal
        to_mask_module_dict = {key: list(group) for key, group in groupby(sorted(to_mask_idx_list, key=lambda x: x[0]), key=lambda x: x[0])}
        returned_boolean_masks = copy.deepcopy(boolean_masks)
        for module_name, curr_mask in boolean_masks.items():
            module_mask_list = torch.LongTensor([[x[1], x[2]] for x in to_mask_module_dict[module_name]])
            row_indices, col_indices = module_mask_list[:, 0], module_mask_list[:, 1]
            # prev_sparsity = curr_mask.sum().item() / curr_mask.numel()
            curr_mask[row_indices, col_indices] = True
            # tot_sum += curr_mask.sum().item()
            # next_sparsity = curr_mask.sum().item() / curr_mask.numel()
            # print("-" * 40)
            # print("module: ", module_name)
            # print("prev density: ", 1 - prev_sparsity)
            # print("next density: ", 1 - next_sparsity)
            # print("difference: ", (1 - next_sparsity) - (1 - prev_sparsity))
            model_named_modules[module_name].weight.data *= (~curr_mask).float().to(curr_mask.device)
            returned_boolean_masks[module_name] = curr_mask
        
        # print("-" * 40)
        # print("Final reached density: ", 1 - (tot_sum / float(tot_param_size)))
        return returned_boolean_masks

    def random_add_more(self, boolean_masks: Dict[str, torch.Tensor], pct_goal: float) -> Dict[str, torch.Tensor]:
        """
        Adds back parameters to reach a target sparsity percentage.

        Args:
            boolean_masks (Dict[str, torch.Tensor]): Dictionary of current boolean masks.
            pct_goal (float): Target sparsity level.

        Returns:
            Dict[str, torch.Tensor]: Updated boolean masks after addition.
        """
        if pct_goal == 1.0:
            return {
                module_name: torch.zeros_like(curr_mask, dtype=torch.bool)
                for module_name, curr_mask in boolean_masks.items()
            }
        
        # 1) Get the current sparsity level and decide if we can add more
        model_named_modules = dict(self.named_modules())
        tot_mask_size = sum(mask.sum().item() for mask in boolean_masks.values())
        tot_param_size = sum(mask.numel() for mask in boolean_masks.values())

        sparsity = 1 - (tot_mask_size / tot_param_size)
        if pct_goal <= sparsity:
            raise ValueError(f"pct_goal ({pct_goal}) is lower than current sparsity ({sparsity}). Use random_remove_more instead.")

        # 2) Calculate how many more parameters need to be added        
        amount_to_add = math.ceil((pct_goal - sparsity) * tot_param_size)
        amount_to_add = max(min(amount_to_add, tot_mask_size), 0)
        if tot_mask_size == 0:
            return boolean_masks

        # 3) Randomly select parameters to add back from the ones that are masked
        all_remaining_idx = []
        for module_name, curr_mask in boolean_masks.items():
            remaining_idx_list = [[module_name] + idx.tolist() for idx in curr_mask.nonzero(as_tuple=False)]
            all_remaining_idx.extend(remaining_idx_list)
        to_unmask_idx_list = random.sample(all_remaining_idx, amount_to_add)
        
        # 4) Group these by module and apply the addition
        to_unmask_module_dict = {key: list(group) for key, group in groupby(sorted(to_unmask_idx_list, key=lambda x: x[0]), key=lambda x: x[0])}
        returned_boolean_masks = copy.deepcopy(boolean_masks)
        for module_name, curr_mask in boolean_masks.items():
            module_unmask_list = torch.LongTensor([[x[1], x[2]] for x in to_unmask_module_dict[module_name]])
            row_indices, col_indices = module_unmask_list[:, 0], module_unmask_list[:, 1]
            # prev_sparsity = curr_mask.sum().item() / curr_mask.numel()
            curr_mask[row_indices, col_indices] = False
            # tot_sum += curr_mask.sum().item()
            # next_sparsity = curr_mask.sum().item() / curr_mask.numel()
            # print("-" * 40)
            # print("module: ", module_name)
            # print("prev density: ", 1 - prev_sparsity)
            # print("next density: ", 1 - next_sparsity)
            # print("difference: ", (1 - next_sparsity) - (1 - prev_sparsity))
            model_named_modules[module_name].weight.data *= (~curr_mask).float().to(curr_mask.device)
            returned_boolean_masks[module_name] = curr_mask

        # print("-" * 40)
        # print("Final reached density: ", 1 - (tot_sum / float(tot_param_size)))
        return returned_boolean_masks
    

class SelectivePruningMixin:
    """
    Mixin class providing methods to replace model layers with masked ones.
    """
    def _validate_params(
        self, 
        top_k_layers: int, 
        linear_types_to_mask: List[str], 
        module_types_to_mask: List[Type[Any]], 
        top_limit: int, 
        bottom_limit: int
    ):
        """Validates the input parameters."""
        if top_k_layers < 1 or top_k_layers > self.num_layers:
            raise ValueError(f"The top k layer number must be between 1 and {self.num_layers} for {self.lm_name}.")
        
        if top_limit != -1 and bottom_limit != -1:
            if (top_limit > self.num_layers - 1) or (bottom_limit < 0) or bottom_limit > top_limit:
                raise ValueError(f"top_limit must be <= {self.num_layers - 1} and bottom_limit >= 0 for {self.lm_name}, got {top_limit} and {bottom_limit}.")

        for lin_type in linear_types_to_mask:
            if lin_type not in ['c_attn', 'q_attn', 'c_proj', 'c_fc']:
                raise ValueError("Linear type must be one of ['c_attn', 'q_attn', 'c_proj', 'c_fc'].")

        for mod_type in module_types_to_mask:
            if mod_type not in [GPT2Attention, GPT2MLP, GPT2Block]:
                raise ValueError("Module type must be one of [GPT2Attention, GPT2MLP, GPT2Block].")

    def replace_layers_with_masked(
        self, 
        out_w_per_mask: int,
        in_w_per_mask: int, 
        top_k_layers: int, 
        linear_types_to_mask: List[str], 
        module_types_to_mask: List[Type[Any]], 
        initial_mask_p: float, 
        top_limit: int, 
        bottom_limit: int, 
        verbose: bool
    ):
        """Replaces layers with their masked versions."""
        if top_limit == -1 or bottom_limit == -1:
            bottom_limit = self.num_layers - top_k_layers
            top_limit = self.num_layers - 1

        def replace_layers(
            linear_types: List[str], 
            parent_types: List[Type[Any]], 
            replacement: Callable[[Any], Any]
        ):
            for module_name, module in self.lm.transformer.named_modules():
                split_name = module_name.split('.')
                for lin_type in linear_types:
                    if hasattr(module, lin_type) \
                        and type(module) in parent_types \
                        and bottom_limit <= int(split_name[1]) <= top_limit:
                            layer = getattr(module, lin_type)
                            setattr(module, lin_type, replacement(layer))
                            if verbose:
                                print(f"Replaced {lin_type} in {module_name}")
        
        replace_layers(
            linear_types_to_mask, 
            module_types_to_mask, 
            lambda x: MaskedHuggingfaceConv1D.from_layer(
                layer=x, 
                out_w_per_mask=out_w_per_mask, 
                in_w_per_mask=in_w_per_mask,
                mask_p=initial_mask_p
            )
        )
    
    def set_is_inverse_mask(self, is_inverse_mask: bool = False):
        """Inverses the mask for all MaskedHuggingfaceConv1D layers."""
        model_modules = self.named_modules()
        for k, v in model_modules:
            if type(v) == MaskedHuggingfaceConv1D:
                v.mask.is_inverse_mask = is_inverse_mask
                    
    def set_is_bypass_mask(self, is_bypass_mask: bool = False):
        """Bypasses the mask for all MaskedHuggingfaceConv1D layers."""
        model_modules = self.named_modules()
        for k, v in model_modules:
            if type(v) == MaskedHuggingfaceConv1D:
                v.is_bypass_mask = is_bypass_mask

    def compute_total_regularizer(self) -> float:
        """Computes the total regularizer for all MaskedHuggingfaceConv1D layers."""
        total, n = 0, 0
        for name, module in self.named_modules():
            if hasattr(module, 'regularizer'):
                return_sum, num_elem = module.regularizer()
                total += return_sum
                n += num_elem
        return total / float(n)
    
    
class MaskStatsMixin:
    """
    Mixin class providing methods to compute and retrieve various mask statistics
    for masked modules within a model.
    """
    
    def compute_binary_pct_produced_mask_stats(self) -> Dict[str, float]:
        """
        Compute per-neuron and per-weight sparsity statistics on the mask scores 
        produced by masked convolutional layers (`MaskedHuggingfaceConv1D`).

        Returns:
            Dict[str, float]: A dictionary containing the computed statistics.
        """
        modules = self.named_modules()
        mask_modules = {k: v for k, v in modules if type(v) == MaskedHuggingfaceConv1D}
        stats = {
            '0': 0.0,
            '1': 0.0,
            '?': 0.0,
            '<=0.01': 0.0,
            '>=0.99': 0.0,
            'strict-binary': 0.0,
            'non-strict-binary': 0.0,
            '0.45-0.55': 0.0,
            '<=0.20': 0.0,
            '>=0.80': 0.0
        }
        total_neurons = 0
        total_weights = 0
        pruned_neurons = 0

        for k, mod in mask_modules.items():
            v_grad = mod.produce_mask_reshaped()
            sig_res = torch.sigmoid(mod.mask.mask_scores)
            if mod.mask.is_inverse_mask:
                sig_res = 1.0 - sig_res
            
            v = v_grad.detach().cpu()
            sig_res = sig_res.detach().cpu()

            # Neuron statistics
            transpose_v = v.t()
            in_feats_num = transpose_v.shape[1]
            for out_feat in transpose_v:
                if torch.equal(out_feat, torch.zeros(in_feats_num)): 
                    pruned_neurons += 1
                total_neurons += 1

            # Weight statistics
            v = v.numpy().flatten()
            sig_res = sig_res.numpy().flatten()

            zeros = np.sum(v == 0.0)
            ones = np.sum(v == 1.0)
            stats['0'] += zeros
            stats['1'] += ones
            stats['strict-binary'] += zeros + ones

            decimal_zeros = np.sum(sig_res <= 0.01)
            decimal_ones = np.sum(sig_res >= 0.99)
            stats['<=0.01'] += decimal_zeros
            stats['>=0.99'] += decimal_ones
            stats['non-strict-binary'] += decimal_zeros + decimal_ones

            decimal_twenties = np.sum(sig_res <= 0.2)
            decimal_eighties = np.sum(sig_res >= 0.8)
            stats['<=0.20'] += decimal_twenties
            stats['>=0.80'] += decimal_eighties

            danger_zone = (0.45 <= sig_res) & (sig_res <= 0.55)
            decimal_danger_zone = np.sum(danger_zone)
            stats['0.45-0.55'] += decimal_danger_zone

            size = v.size
            stats['?'] += size - decimal_zeros - decimal_ones
            total_weights += size
        
        all_total_weights = total_weights

        modules = self.named_modules()
        non_mask_modules = {k: v for k, v in modules if type(v) == Conv1D}
        for k, mod in non_mask_modules.items():
            all_total_weights += mod.weight.detach().cpu().numpy().flatten().size
        stats["unconditonal_sparsity"] =  stats['0'] / float(all_total_weights)
        
        # Normalize the computed stats by the total weights or total neurons
        stats = {k: float(v) / total_weights for k,v in stats.items()}
        stats["neuron-sparsity"] = float(pruned_neurons) / total_neurons
        return stats

    
    def conditional_produce_mask(
        self, 
        param: Any, 
        non_mask_counts: bool, 
        density: bool = False
    ):
        """
        Produce a mask for the given parameter based on its attributes.

        If the parameter has a 'produce_mask_reshaped' method, that method is used.
        Otherwise, as it is a non-masked parameters, if `non_mask_counts` is True, 
        then a default mask is returned based on the `density` flag (zeros if 
        density is True, ones otherwise). If `non_mask_counts` is False,
        returns None for non-masked params.

        Args:
            param (Any): The module or parameter for which to produce a mask.
            non_mask_counts (bool): Whether to produce a default mask when a custom mask is not available.
            density (bool): If True, produce a mask of zeros for non-masked parameters that count into the calculation; otherwise, a mask of ones.

        Returns:
            Optional[torch.Tensor]: The produced mask, or None.
        """
        param_mask = None
        if hasattr(param, 'produce_mask_reshaped'):
            param_mask = param.produce_mask_reshaped()
        else:
            if non_mask_counts:
                if density:
                    # non-masked params should have zero density masks as they are not removed
                    param_mask = torch.zeros(param.weight.size())
                else:
                    # non-masked params should have completely sparse masks as they are not removed
                    param_mask = torch.ones(param.weight.size())
            else:
                param_mask = None
        return param_mask
    
    @torch.no_grad()
    def get_mask_sparsity_head_qkv(
        self, layer_num: int, head: int, non_mask_counts: bool = True, density: bool = False
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Computes the sparsity (or density depending on the flag) for the query, key, and value masks
        for a given head in a specified layer.

        The method splits the combined q, k, v mask (from the c_attn module) into three parts,
        reshapes them to separate heads, and then extracts the mask for the specified head.

        Args:
            layer_num (int): The layer number (0-indexed).
            head (int): The head number (0-indexed).
            non_mask_counts (bool): Flag indicating whether to count non-masked parameters into the computation.
            density (bool): Flag indicating whether to calculate density (if True) or sparsity (if False).

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]:
                A tuple containing the sparsity fractions for query, key, and value masks.
        """
        layer = self.lm.transformer.h[layer_num].attn
        num_heads, embed_dim = self.lm.transformer.config.num_attention_heads, self.lm.transformer.config.hidden_size
        head_dim = embed_dim // num_heads
        target_mask_value = 1.0 if density else 0.0
        
        assert head_dim * num_heads == embed_dim, "Inconsistent head dimensions."
        
        # NOTE: c_attn is q, k, v combined 
        c_attn_mask = self.conditional_produce_mask(layer.c_attn, non_mask_counts, density=density) # size: (in_feats, 3 * hidden_dim)
        query_mask, key_mask, value_mask = c_attn_mask.split(embed_dim, dim=-1) # size: (in_feats, hidden_dim)
        
        if value_mask is not None and head is not None:
            new_shape = value_mask.size()[:-1] + (num_heads, head_dim) # size: (in_feats, num_heads, hidden_dim / num_heads)
            value_mask = value_mask.view(new_shape).permute(1, 0, 2) # size: (num_heads, in_feats, hidden_dim / num_heads)
            value_mask = value_mask[head]
        
        if query_mask is not None and head is not None:
            new_shape = query_mask.size()[:-1] + (num_heads, head_dim) # size: (in_feats, num_heads, hidden_dim / num_heads)
            query_mask =  query_mask.view(new_shape).permute(1, 0, 2) # size: (num_heads, in_feats, hidden_dim / num_heads)
            query_mask = query_mask[head]
        
        if key_mask is not None and head is not None:
            new_shape = key_mask.size()[:-1] + (num_heads, head_dim) # size: (in_feats, num_heads, hidden_dim / num_heads)
            key_mask =  key_mask.view(new_shape).permute(1, 0, 2) # size: (num_heads, in_feats, hidden_dim / num_heads)
            key_mask = key_mask[head]

        query_frac = torch.mean((query_mask == target_mask_value).type(torch.FloatTensor)).item() if query_mask is not None else None
        key_frac = torch.mean((key_mask == target_mask_value).type(torch.FloatTensor)).item() if key_mask is not None else None
        value_frac = torch.mean((value_mask == target_mask_value).type(torch.FloatTensor)).item() if value_mask is not None else None
        
        print(
            query_frac,
            key_frac,
            value_frac
        )
        
        return (
            query_frac,
            key_frac,
            value_frac
        )
    
    @torch.no_grad()
    def get_mask_sparsity_attn(
        self, layer_num: int, non_mask_counts: bool = True, density: bool = False
    ) -> Optional[float]:
        """
        Computes the sparsity (or density depending on the flag) of the attention sub-modules within a specified layer.

        Args:
            layer_num (int): The layer number (0-indexed).
            non_mask_counts (bool): Flag indicating whether to count non-masked parameters into the computation.
            density (bool): Flag indicating whether to calculate density (if True) or sparsity (if False).

        Returns:
            Optional[float]: The computed sparsity, or None if no masks are available.
        """
        layer = self.lm.transformer.h[layer_num]
        c_attn_mask = self.conditional_produce_mask(layer.attn.c_attn, non_mask_counts, density=density)
        c_proj_mask = self.conditional_produce_mask(layer.attn.c_proj, non_mask_counts, density=density)
        target_mask_value = 1.0 if density else 0.0
        
        if c_attn_mask is None and c_proj_mask is None:
            return None
        else:
            sparsity = 0.0
            total = 0.0
            if c_attn_mask is not None:
                sparsity += torch.sum(c_attn_mask == target_mask_value).item()
                total += c_attn_mask.numel()
            if c_proj_mask is not None:
                sparsity += torch.sum(c_proj_mask == target_mask_value).item()
                total += c_proj_mask.numel()
            sparsity = sparsity / total

            return sparsity
                
    @torch.no_grad()
    def get_mask_sparsity_mlp(self, layer_num: int, non_mask_counts: bool = True, density: bool = False
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Computes the sparsity (or density depending on the flag) of MLP (i.e. dense) sub-modules within a specified layer.

        Specifically, it calculates the sparsity for:
            - The c_proj module in the attention block.
            - The c_fc module in the MLP block.
            - The c_proj module in the MLP block.

        Args:
            layer_num (int): The layer number (0-indexed).
            non_mask_counts (bool): Flag indicating whether to count non-masked parameters into the computation.
            density (bool): Flag indicating whether to calculate density (if True) or sparsity (if False).

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]:
                Sparsity fractions for the three dense modules.
        """
        layer = self.lm.transformer.h[layer_num]
        mask1 = self.conditional_produce_mask(layer.attn.c_proj, non_mask_counts, density=density)
        mask2 = self.conditional_produce_mask(layer.mlp.c_fc, non_mask_counts, density=density)
        mask3 = self.conditional_produce_mask(layer.mlp.c_proj, non_mask_counts, density=density)
        target_mask_value = 1.0 if density else 0.0

        dense1_frac = torch.mean((mask1 == target_mask_value).type(torch.FloatTensor)).item() if mask1 is not None else None
        dense2_frac = torch.mean((mask2 == target_mask_value).type(torch.FloatTensor)).item() if mask2 is not None else None
        dense3_frac = torch.mean((mask3 == target_mask_value).type(torch.FloatTensor)).item() if mask3 is not None else None
        
        return (
            dense1_frac, 
            dense2_frac, 
            dense3_frac
        )

    @torch.no_grad()
    def get_mask_sparsity_layer(
        self, layer_num: int, non_mask_counts: bool = True, density: bool = False
    ) -> Optional[float]:
        """
        Compute the overall sparsity (or density depending on the flag) for a 
        specified layer by aggregating sparsity from both the MLP and attention sub-modules.

        Specifically, it calculates the sparsity for:
            - The c_fc and c_proj modules in the MLP block.
            - The c_attn and c_proj modules in the attention block.

        Args:
            layer_num (int): The layer number (0-indexed).
            non_mask_counts (bool): Flag indicating whether to count non-masked parameters into the computation.
            density (bool): Flag indicating whether to calculate density (if True) or sparsity (if False).

        Returns:
            Optional[float]: The overall sparsity fraction, or None if no masks are available.
        """
        layer = self.lm.transformer.h[layer_num]
        mask2 = self.conditional_produce_mask(layer.mlp.c_fc, non_mask_counts, density=density)
        mask3 = self.conditional_produce_mask(layer.mlp.c_proj, non_mask_counts, density=density)
        c_attn_mask = self.conditional_produce_mask(layer.attn.c_attn, non_mask_counts, density=density)
        c_proj_mask = self.conditional_produce_mask(layer.attn.c_proj, non_mask_counts, density=density)
        target_mask_value = 1.0 if density else 0.0

        if mask2 is None and mask3 is None and c_attn_mask is None and c_proj_mask is None:
            return None
        else:
            mask_list = [mask2, mask3, c_attn_mask, c_proj_mask]
            sparsity = 0.0
            total = 0.0
            for mask in mask_list:
                if mask is not None: 
                    sparsity += torch.sum(mask == target_mask_value).item()
                    total += mask.numel()
            return sparsity / total