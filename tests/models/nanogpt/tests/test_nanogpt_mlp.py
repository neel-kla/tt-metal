# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import pytest

from transformers import GPT2LMHeadModel

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import tests.models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
from tests.models.nanogpt.tt.nanogpt_config import GPTConfig

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_nanogpt_mlp(pcc, device):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.mlp"

    torch.manual_seed(0)

    test_in = torch.rand(1, 43, 768)

    tt_test_in = torch2tt_tensor(
        test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
    )

    model_type = "gpt2"

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        model_type: dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    }[model_type]

    config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    config_args["bias"] = True  # always True for GPT model checkpoints
    # we can override the dropout rate, if desired

    config = GPTConfig(**config_args)

    tt_mlp = nanogpt_mlp.TtMLP(base_address, config, sd, device)

    tt_out = tt_mlp.forward(tt_test_in)

    pt_mlp = model_hf.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in)

    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_mlp: Passed!")
    else:
        logger.warning("nanogpt_mlp: Failed!")

    assert does_pass
