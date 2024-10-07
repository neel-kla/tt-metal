# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from torch import nn

import ttnn
from models.utility_functions import (
    nearest_32,
)


class TtLlamaAttention(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = configuration.paged_attention_config

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.kv_seq_len = configuration.kv_seq_len
        self.sliding_window = configuration.sliding_window
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.model_config = configuration.get_model_config()

        layer_name = f"layers.{layer_num}.attention"
        if configuration.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv_list = []
        self.wo_list = []
        self.layer_past_list = []

        for i in range(self.num_devices):
            # wo: 4096 x 4096: width-sharded on 12 banks, 4224 over 12 banks.
            wqkv_shard_shape = (
                4096,
                6144 // 12,
            )
            wqkv_shard_spec = ttnn.ShardSpec(
                configuration.dram_weight_grid, wqkv_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False
            )
            wqkv_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, wqkv_shard_spec
            )
            wqkv = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.transpose(
                            torch.chunk(self.state_dict[wq_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wk_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wv_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                    ],
                    dim=-1,
                ),
                device=self.devices[i],
                dtype=self.dtype,
                memory_config=wqkv_mem_config,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wqkv_sharded"),
            )

            # wo: 4096 x 4096: width-sharded on 12 banks, 4224 over 12 banks.
            wo_shard_shape = (
                4096,
                4224 // 12,
            )  # (11 shards) 4224 - padded cols to divide by 12 dram cores (and divisible by tile size of 32)
            wo_shard_spec = ttnn.ShardSpec(
                configuration.dram_weight_grid, wo_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False
            )
            wo_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, wo_shard_spec
            )
            wo = ttnn.as_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                device=self.devices[i],
                memory_config=wo_mem_config,
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wo_sharded"),
            )

            if self.paged_attention_config:
                cache_k = torch.zeros(
                    (
                        self.paged_attention_config.max_num_blocks,
                        self.n_kv_heads // self.num_devices,
                        self.paged_attention_config.block_size,
                        self.head_dim,
                    )
                )
                cache_v = torch.zeros(
                    (
                        self.paged_attention_config.max_num_blocks,
                        self.n_kv_heads // self.num_devices,
                        self.paged_attention_config.block_size,
                        self.head_dim,
                    )
                )
            else:
                cache_k = torch.zeros(
                    (
                        self.max_batch_size,
                        self.n_kv_heads // self.num_devices,
                        self.sliding_window,
                        self.head_dim,
                    )
                )
                cache_v = torch.zeros(
                    (
                        self.max_batch_size,
                        self.n_kv_heads // self.num_devices,
                        self.sliding_window,
                        self.head_dim,
                    )
                )
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(
                    lp, device=self.devices[i], layout=self.model_config["ATTN_W_LAYOUT_TILE"], dtype=self.dtype
                )
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

        self.q_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.k_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.expand_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=4,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.reduce_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.attn_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
        )
        self.attention_grid = ttnn.CoreCoord(8, 4)
        self.scale = self.head_dim**-0.5

    def forward_decode(
        self,
        xs: List[ttnn.Tensor],
        current_pos,
        rot_mat=None,
        page_table=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            assert self.max_batch_size * self.n_kv_heads < 64

            ###
            # QKV matmuls
            # Use HiFi2 for DRAM-sharded matmuls as htey are otherwise flop-bound. Loses 1 bit of activation precision.
            ###
            x_sharded = ttnn.interleaved_to_sharded(x, self.model_config["SHARDED_SKIP_INPUT_MEMCFG"])
            xqkv_fused_sharded = ttnn.linear(
                x_sharded,
                wqkv,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.model_config["XQKV_DECODE_PROGCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
            )

            ttnn.deallocate(x_sharded)

            xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(xqkv_fused_sharded)
            # Reshape such that true unpadded batch is tracked in shape
            fqkv_shape = xqkv_fused.shape
            xqkv_fused = ttnn.reshape(
                xqkv_fused, ttnn.Shape((1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
            )

            ###
            # Reshape and rotary embeddings
            ###
            (
                q_heads_pre_rot_1BQD,
                k_heads_pre_rot_1BKD,
                v_heads_1BKD,
            ) = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            )

            ttnn.deallocate(xqkv_fused)

            q_heads_1BQD = ttnn.linear(
                q_heads_pre_rot_1BQD,
                rot_mat,
                program_config=self.model_config["ROT_MAT_BMM_PROGCFG"],
                memory_config=q_heads_pre_rot_1BQD.memory_config(),
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
            )

            k_heads_1BKD = ttnn.linear(
                k_heads_pre_rot_1BKD,
                rot_mat,
                program_config=self.model_config["ROT_MAT_BMM_PROGCFG"],
                memory_config=k_heads_pre_rot_1BKD.memory_config(),
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
            )

            ttnn.deallocate(q_heads_pre_rot_1BQD)
            ttnn.deallocate(k_heads_pre_rot_1BKD)

            ###
            # KV update
            ###
            keys = layer_past[0]
            values = layer_past[1]

            # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
            # v_heads [seqlen, n_kv_heads, bsz, head_dim]
            # keys, [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]
            ttnn.experimental.paged_update_cache(
                keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            self.layer_past_list[i] = [keys, values]

            ttnn.deallocate(k_heads_1BKD)
            ttnn.deallocate(v_heads_1BKD)

            if page_table:
                attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_heads_1BQD,
                    keys,
                    values,
                    cur_pos_tensor=current_pos,
                    page_table_tensor=page_table,
                    scale=self.scale,
                    program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                    compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            else:
                attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_heads_1BQD,
                    keys,
                    values,
                    cur_pos_tensor=current_pos,
                    scale=self.scale,
                    program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                    compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            ttnn.deallocate(q_heads_1BQD)
            attn_output_11BH = ttnn.to_memory_config(
                attn_output_1G4D, memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"]
            )
            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
                attn_output_11BH,
                num_heads=self.n_heads,
            )

            ttnn.deallocate(attn_output_11BH)
            attn_output_cat = ttnn.reshape(attn_output_cat, ttnn.Shape((1, 1, 32, self.hidden_size)))

            dense_out = ttnn.linear(
                attn_output_cat,
                wo,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.model_config["ATTN_OUTPUT_PROGCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )  # seqlen, 1, batch, hidden_size

            ttnn.deallocate(attn_output_cat)
            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return None  # tt_all_reduce(dense_outputs)
        else:
            return dense_outputs

    def forward_prefill(self, xs_11SH, rot_mats, transformation_mats, user_id: int = 0, page_table=None):
        seq_len = xs_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        wqkv = self.wqkv_list[0]
        wo = self.wo_list[0]
        self.layer_past = self.layer_past_list[0]
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            xs_11SH = ttnn.reshape(xs_11SH, [1, 2, seq_len // 2, -1])
        xqkv_fused = ttnn.linear(
            xs_11SH,
            wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        xs_11SH.deallocate(True)

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        q_heads_1QSD_pre_rot.deallocate(True)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        k_heads_1KSD_pre_rot.deallocate(True)

        # Fill KV-Cache
        keys_BKSD = self.layer_past[0]
        values_BKSD = self.layer_past[1]

        k_fill = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        # sharding k_fill to deal with update_cache memory limitation
        if seq_len > 128:
            k_fill = ttnn.interleaved_to_sharded(k_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        v_fill = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
        # sharding v_fill to deal with update_cache memory limitation
        if seq_len > 128:
            v_fill = ttnn.interleaved_to_sharded(v_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))

        if page_table:
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id,
            )

        self.layer_past = [keys_BKSD, values_BKSD]

        # SDPA

        # reshaping to put group in batch dim to do sdpa on 8 MQAs in parallel
        k_heads_K1SD = ttnn.reshape(k_heads_1KSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        v_heads_V1SD = ttnn.reshape(v_heads_1VSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        q_heads_84SD = ttnn.reshape(
            q_heads_1QSD, [self.n_local_kv_heads, self.n_local_heads // self.n_local_kv_heads, -1, self.head_dim]
        )
        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_84SD,
            k_heads_K1SD,
            v_heads_V1SD,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        # deallocate keys and values
        q_heads_84SD.deallocate(True)
        k_heads_K1SD.deallocate(True)
        v_heads_V1SD.deallocate(True)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_1QSD.deallocate(True)

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 2, seq_len // 2, -1])
        output_11SH = ttnn.linear(
            attn_output_11SH,
            wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        attn_output_11SH.deallocate(True)
        return [output_11SH]

    def forward(
        self, xs, current_pos, rot_mats=None, transformation_mats=None, user_id=0, mode="decode", page_table=None
    ):
        if mode == "prefill":
            return self.forward_prefill(xs[0], rot_mats, transformation_mats, user_id, page_table)
        else:
            return self.forward_decode(xs, current_pos, rot_mats, page_table)
