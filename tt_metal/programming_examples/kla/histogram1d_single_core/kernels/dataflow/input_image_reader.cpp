/**
 * @file input_image_reader.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdint.h>

#include "dataflow_api.h"


void generate_bcast_scaler() {
    constexpr uint32_t cb_intermed_1 = 25U; // CB id 25 reserved for intermediates
    uint32_t scaler = 1;
    union { float f; uint32_t u; } u; u.u = scaler;
    //DPRINT << "basic Scaler = " << F32(u.f) << ENDL();
    cb_reserve_back(cb_intermed_1, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_intermed_1));
    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    cb_push_back(cb_intermed_1, 1);
}

// kernel_0 : READER for Copying
void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0); // DRAM address
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;  // First input circular buffer

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

    generate_bcast_scaler();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);  // circular buffer address
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
