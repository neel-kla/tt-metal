# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_reduce_scatter_post_commit import (
    is_unsupported_case,
    run_reduce_scatter_test,
)
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 64, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 128], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 1024], 3, ttnn.TILE_LAYOUT),
        # Has worker slice size warning - defaults to 1x1
        ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
        # Always fails with bfp8_b
        ([1, 1, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        # Has worker slice size warning - defaults to 1x1
        ([1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True, False])
def test_reduce_scatter_nightly(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_reduce_scatter_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        enable_async=enable_async,
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 64, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 128], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 1024], 3, ttnn.TILE_LAYOUT),
        # Has worker slice size warning - defaults to 1x1
        ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
        # Always fails with bfp8_b
        ([1, 1, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        # Has worker slice size warning - defaults to 1x1
        ([1, 1, 2048, 8192], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True, False])
def test_reduce_scatter_nightly(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_reduce_scatter_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        enable_async=enable_async,
    )
