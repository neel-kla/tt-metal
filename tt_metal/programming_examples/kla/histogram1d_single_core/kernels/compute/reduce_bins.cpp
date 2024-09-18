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


#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"
#include "compute_kernel_api/tile_move_copy.h"



// Compute Kernel : perform the binning logic
namespace NAMESPACE {

void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    const uint32_t oneTile = 1;
    // reduce_init<true>(tt::CB::c_intermed0, tt::CB::c_intermed1);
    // cb_wait_front(tt::CB::c_intermed1, 1); // scaler tile from the reader
    // // heaviside_tile_init();
    // DPRINT << "Im inside Reduce bins kernel" << ENDL();
    // for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
    //     cb_reserve_back(tt::CB::c_out0, oneTile);
    //     // Acquire the dest register file for updating them
    //     acquire_dst(tt::DstMode::Full);
    //     // Pop tile after tile, copy to DST and pack
    //     cb_wait_front(tt::CB::c_intermed0, 1);
    //     // copy the tile from input circular buffer to dst register index 0
    //     // copy_tile(tt::CB::c_intermed0, 0, 0);
    //     // Comparison operation to find the mask of indices which are eq to the value 728
    //     // unary_eq_tile(0, 0x44360000);
    //     // reduce_tile_math(0, 32);
    //     // heaviside_tile(0, 5);
    //     // Pack tile from registers onto destination circular buffer
    //     pack_tile(0, tt::CB::c_out0);
    //     // move to next tile in circular buffer
    //     cb_pop_front(tt::CB::c_intermed0, 1);
    //     // release the destination register files for this core
    //     release_dst(tt::DstMode::Full); 
    //     // push the copied tile onto circular buffer slot
    //     cb_push_back(tt::CB::c_out0, oneTile);
    // }
}
}  // namespace NAMESPACE
