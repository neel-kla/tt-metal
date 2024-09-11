/**
 * @file histogram_compute1d.cpp
 * @author Neelakandan Ramachandran - neelakandan.ramachandran@kla-tencor.com
 * @brief
 * @version 0.1
 * @date 2024-09-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

// Compute Kernel : perform greater than logic
namespace NAMESPACE {

void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    const uint32_t oneTile = 1;

    unary_op_init_common(tt::CB::c_in0);
    unary_eq_tile_init();
    // heaviside_tile_init();
    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        cb_reserve_back(tt::CB::c_out0, oneTile);
        // Acquire the dest register file for updating them
        acquire_dst(tt::DstMode::Full);
        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CB::c_in0, 1);
        // copy the tile from input circular buffer to dst register index 0
        copy_tile(tt::CB::c_in0, 0, 0);
        // Comparison operation to find the mask of indices which are gt the value 728
        //unary_gt_tile(0, 0x44360000);
        unary_eq_tile(0, 0x44360000);
        // heaviside_tile(0, 5);
        // Pack tile from registers onto destination circular buffer
        pack_tile(0, tt::CB::c_out0);
        // move to next tile in circular buffer
        cb_pop_front(tt::CB::c_in0, 1);
        // release the destination register files for this core
        release_dst(tt::DstMode::Full); 
        // push the copied tile onto circular buffer slot
        cb_push_back(tt::CB::c_out0, oneTile);
    }
}
}  // namespace NAMESPACE
