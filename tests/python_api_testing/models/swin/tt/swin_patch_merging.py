from typing import Optional, Tuple
import torch
import torch.nn as nn


from python_api_testing.models.utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from python_api_testing.models.swin.swin_helper_funcs import linear as TtLinear
import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtSwinPatchMerging(nn.Module):
    def __init__(
        self,
        config,
        input_resolution: Tuple[int],
        dim: int,
        state_dict,
        base_address,
        device,
        host,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.device = device
        self.host = host
        self.config = config

        self.reduction_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.downsample.reduction.weight"], self.device
        )

        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.downsample.norm.weight"], self.device
        )
        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.downsample.norm.bias"], self.device
        )

        self.norm = fallback_ops.LayerNorm(
            self.gamma, self.beta, normalized_shape=4 * dim, eps=config.layer_norm_eps
        )

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = fallback_ops.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self, input_feature: tt_lib.tensor.Tensor, input_dimensions: Tuple[int, int]
    ) -> tt_lib.tensor.Tensor:
        height, width = input_dimensions
        _, batch_size, dim, num_channels = input_feature.shape()

        input_feature = fallback_ops.reshape(
            input_feature, batch_size, height, width, num_channels
        )

        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature = tt_to_torch_tensor(input_feature, self.host)

        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # [batch_size, height/2, width/2, 4*num_channels]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
        input_feature = torch_to_tt_tensor_rm(input_feature, self.device)
        input_feature = fallback_ops.reshape(
            input_feature, 1, batch_size, -1, 4 * num_channels
        )

        input_feature = self.norm(input_feature)
        input_feature = TtLinear(input_feature, self.reduction_weight)

        return input_feature
