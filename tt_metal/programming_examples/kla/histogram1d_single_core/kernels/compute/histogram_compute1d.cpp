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
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tile.h"

namespace NAMESPACE {

void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    const uint32_t oneTile = 1;

    // Binning loop; perform the binning logic for a given comparison value for eg 728 here
    unary_op_init_common(tt::CB::c_in0);
    unary_eq_tile_init();  // initialize for unary eq tile operation
    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        // A blocking call that waits for 1 tile to be available in CB c_in0
        cb_wait_front(tt::CB::c_in0, 1);
        // Acquire an exclusive lock on the DST register for the MATH thread
        tile_regs_acquire();
        // copy the tile 0 from input CB c_in0 to dst register index 0
        copy_tile(tt::CB::c_in0, 0, 0);
        // Compare against hex value of 728 and get indices of all 728 in tile set as 1 (binning)
        unary_eq_tile(0, 0x44360000);
        // Release lock on DST register by MATH thread.
        tile_regs_commit();
        // A blocking call that waits for 1 tiles to be free in the CB c_intermed0
        cb_reserve_back(tt::CB::c_intermed0, oneTile);
        // Acquire an exclusive lock on the DST register for the PACK thread
        tile_regs_wait();
        // Copies a single tile from the DST register buffer at 0 to CB c_intermed0 at 0
        pack_tile(0, tt::CB::c_intermed0);
        // Release lock on DST register by PACK thread.
        tile_regs_release();
        // Pushe one tile  back of the c_intermed0 CB’s queue
        cb_push_back(tt::CB::c_intermed0, oneTile);
        // Pop one tile from the front of the c_in0 CB's queue
        cb_pop_front(tt::CB::c_in0, 1);
    }

    // Initialize reduction tile operation on the CBs
    reduce_init_short(tt::CB::c_intermed0, tt::CB::c_intermed1);
    cb_wait_front(tt::CB::c_intermed1, 1);  // scaler tile from the reader
    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        // A blocking call that waits for 1 tile to be available in CB c_intermed0
        cb_wait_front(tt::CB::c_intermed0, 1);
        // Acquire an exclusive lock on the DST register for the MATH thread
        tile_regs_acquire();
        // Perform the reduction on the c_intermed0 tile with c_intermed1 acting as scalar multiple
        // Perform a reduction operation B = reduce(A) using reduce_func for dimension reduction on a tile in the CB
        // at a given index and writes the result to the DST register at index dst_tile_index
        reduce_tile(tt::CB::c_intermed0, tt::CB::c_intermed1, 0, 0, 0);
        // Release lock on DST register by MATH thread.
        tile_regs_commit();
        // Pop one tile from the front of the c_intermed0 CB's queue
        cb_pop_front(tt::CB::c_intermed0, 1);
        // A blocking call that waits for 1 tiles to be free in the CB c_out0
        cb_reserve_back(tt::CB::c_out0, oneTile);
        // Acquire an exclusive lock on the DST register for the PACK thread
        tile_regs_wait();
        // Copies a single tile from the DST register buffer at 0 to CB c_out0 at 0
        pack_tile(0, tt::CB::c_out0);
        // Release lock on DST register by PACK thread.
        tile_regs_release();
        // push the copied tile onto circular buffer slot
        cb_push_back(tt::CB::c_out0, oneTile);
    }
}
}  // namespace NAMESPACE
