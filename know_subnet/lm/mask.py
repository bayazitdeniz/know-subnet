from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.modeling_utils import Conv1D

class ModularMask(nn.Module):
    """
    A modular mask module inspired by:
        - https://github.com/RobertCsordas/modules/blob/master/framework/layers/gumbel_sigmoid.py
        - https://github.com/stevenxcao/subnetwork-probing/blob/main/code/masked_linear.py

    This module learns mask scores and produces a binary mask via a gumbel sigmoid trick.
    """
    def __init__(self, mask_dim: Tuple[int, int, int, int], mask_p: float):
        """
        Args:
            mask_dim (Tuple[int, int, int, int]): The shape of the mask scores.
            mask_p (float): The initial probability for the mask.
        """
        super().__init__()
        self.mask_scores = nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.tau = 1.0
        self.b = 1e-4
        self.init_weights()
        self.is_inverse_mask=False

    def init_weights(self):
        """Initialize the mask scores with the logit of mask_p."""
        mask_logit = torch.logit(torch.Tensor([self.mask_p])).item()
        init.constant_(self.mask_scores, mask_logit)

    def gumbel_sigmoid(
        self, 
        logits: torch.Tensor, 
        tau: float = 1, 
        hard: bool = True, 
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Applies the Gumbel Sigmoid trick to the provided logits.

        Args:
            logits (torch.Tensor): The input logits.
            tau (float): The temperature parameter.
            hard (bool): Whether to use the hard (discrete) version.
            eps (float): A small epsilon for numerical stability.

        Returns:
            torch.Tensor: The resulting mask after applying gumbel sigmoid.
        """
        if self.training:
            # Generate Gumbel uniform noise.
            uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)
            noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
            res = torch.sigmoid((logits + noise) / tau)
        else:
            res = torch.sigmoid(logits)

        if hard:
            if self.is_inverse_mask:
                res = 1.0 - res
            # Straight-through estimator: discretize while preserving gradients.
            res = ((res > 0.5).type_as(res) - res).detach() + res

        return res  
        
    def produce_mask(self) -> torch.Tensor:
        """Produce a binary mask by applying the gumbel sigmoid function."""
        mask_res = self.gumbel_sigmoid(logits=self.mask_scores, tau=self.tau, hard=True)
        return mask_res

    def regularizer(self) -> Tuple[torch.Tensor, int]:
        """
        Compute a regularization term based on the mask scores.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the sum of the sigmoid
            applied to mask scores and the total number of mask elements.
        """
        return torch.sum(torch.sigmoid(self.mask_scores)), self.mask_scores.numel()


class MaskedHuggingfaceConv1D(Conv1D):
    """
    A masked 1D convolutional layer compatible with Hugging Face's Conv1D.

    This layer acts like a linear layer with transposed weights and supports
    weight masking. The mask can be applied on the weights in different ways
    (input neuron masking, output neuron masking, or weight masking).

    Args:
        nf (int): Number of output features.
        nx (int): Number of input features.
        mask_p (float): The initial probability for masking.
        out_w_per_mask (int): Number of output features to mask per parameter.
        in_w_per_mask (int): Number of input features to mask per parameter.
        num_heads (int): Number of attention heads.
    """

    def __init__(
        self, 
        nf: int, 
        nx: int, 
        mask_p: float = 0.8808, 
        out_w_per_mask: int = 1,
        in_w_per_mask: int = 1,
        num_heads: int = 12,
    ):
        super().__init__(nf=nf, nx=nx)
        self.num_heads = num_heads
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask

        self.out_features = nf 
        self.in_features = nx
        
        assert nf % out_w_per_mask == 0, "{} % {} not 0".format(nf, out_w_per_mask)
        assert nx % in_w_per_mask == 0, "{} % {} not 0".format(nx, in_w_per_mask)

        # NOTE: The mask dimension is set as the opposite size of the linear case 
        # because the weight is transposed.
        mask_dim = (1, nx // in_w_per_mask, 1, nf // out_w_per_mask)
        self.mask = ModularMask(mask_dim=mask_dim, mask_p=mask_p)
        
        self.is_bypass_mask = False


    def produce_mask_reshaped(self) -> torch.Tensor:
        """
        Produce and reshape the mask to match the weight shape.

        The mask is repeated to match the dimensions of the weight matrix and
        then reshaped into [in_features, out_features].

        Returns:
            torch.Tensor: The reshaped mask.
        """
        mask = self.mask.produce_mask()
        mask = mask.repeat( self.in_w_per_mask, 1, self.out_w_per_mask, 1)
        mask = mask.reshape(self.in_features, self.out_features)
        return mask

    def produce_mask(self) -> torch.Tensor:
        """
        Produce the mask without reshaping.

        Returns:
            torch.Tensor: The produced mask.
        """
        mask = self.mask.produce_mask()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overwrites the forward to have a masked convolutional layer.

        If `is_bypass_mask` flag is false, the mask is applied to the weights
        before performing the matrix multiplication. Otherwise the mask is ignored.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the convolutional layer.
        """
        masked_weight = None
        if self.is_bypass_mask:
            masked_weight = self.weight
        else:
            # Reshape the weight to apply the mask
            # NOTE: Could also do "masked_weight = self.produce_mask_reshaped() * self.weight".
            #       Was noted as slower in the stevenxcao/subnetwork-probing implementation.
            #       Although that was for a Linear layer, efficiency might be different for Conv1D.
            masked_weight = self.produce_mask() * self.weight.reshape(
                self.in_w_per_mask, 
                self.in_features // self.in_w_per_mask,
                self.out_w_per_mask, 
                self.out_features // self.out_w_per_mask
            )
            masked_weight = masked_weight.reshape(self.in_features, self.out_features)
        
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), masked_weight)
        act = x.view(*size_out)
        return act

    @classmethod
    def from_layer(
        cls,
        layer: Conv1D,
        out_w_per_mask: int,
        in_w_per_mask: int,
        mask_p: float,
    ):
        """
        Create a MaskedHuggingfaceConv1D instance from an existing Conv1D layer.

        Depending on the masking configuration, this method sets up the
        instance for input neuron masking, output neuron masking, or weight masking.

        Args:
            layer (Conv1D): The original Conv1D layer.
            out_w_per_mask (int): Number of output features to mask per parameter.
            in_w_per_mask (int): Number of input features to mask per parameter.
            mask_p (float): The initial probability for masking.

        Returns:
            MaskedHuggingfaceConv1D: The new masked convolutional layer instance.
        """
        assert isinstance(layer, Conv1D), (
            f"layer provided is not Conv1D, but {type(layer)}"
        )
        
        if out_w_per_mask == 768 and in_w_per_mask == 1:
            # Input neuron masking
            res = cls(
                nf=layer.nf, 
                nx=layer.weight.shape[0], 
                out_w_per_mask=layer.nf, 
                in_w_per_mask=in_w_per_mask,
                mask_p=mask_p
            )
        elif out_w_per_mask == 1 and in_w_per_mask == 768:
            # Output neuron masking
            res = cls(
                nf=layer.nf,
                nx=layer.weight.shape[0], 
                out_w_per_mask=out_w_per_mask, 
                in_w_per_mask=layer.weight.shape[0],
                mask_p=mask_p
            )
        elif out_w_per_mask == 1 and in_w_per_mask == 1:
            # Weight masking
            res = cls(
                nf=layer.nf, 
                nx=layer.weight.shape[0], 
                out_w_per_mask=out_w_per_mask, 
                in_w_per_mask=in_w_per_mask,
                mask_p=mask_p
            )
        else:
            raise NotImplementedError(
                f"out_w_per_mask={out_w_per_mask} and in_w_per_mask={in_w_per_mask} not implemented."
            )

        res.weight = layer.weight
        res.bias = layer.bias
        return res

