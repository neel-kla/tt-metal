/**
 * @file histogram_compute1d.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    const uint32_t oneTile = 1;

    init_sfpu(tt::CB::c_in0);
    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        cb_reserve_back(tt::CB::c_out0, oneTile);

        acquire_dst(tt::DstMode::Full);

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CB::c_in0, 1);

        copy_tile(tt::CB::c_in0, 0, 0);

        pack_tile(0, tt::CB::c_out0);

        cb_pop_front(tt::CB::c_in0, 1);

        release_dst(tt::DstMode::Full);

        cb_push_back(tt::CB::c_out0, oneTile);
    }
}
}  // namespace NAMESPACE
