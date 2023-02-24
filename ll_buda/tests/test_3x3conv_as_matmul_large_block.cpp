//////////////////////////////////////////////////////////////////////////////////////////
// Tests a 3x3 convolution by implementing as a matmul.
// Converts the layout of activation from nchw to nhwc on the host and copies to DRAM.
// Converts the layout of weights to a 2d matrix and tilizes it on the host before copying to DRAM.
// Computes an address map on the host to copy the untilized activations from DRAM and tilize them in L1.
// Uses "generic_binary_reader_blocked" kernel to read untilized activations from DRAM using the address map computed on host.
// The "generic_binary_reader_blocked" kernel also reads the tilized weights from DRAM.
// Uses "matmul_large_block_zm" kernel to do the compute. Uses "writer_unswizzle" kernel to write tilized output to DRAM.
//////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <functional>
#include <random>
#include<chrono>
#include <tuple>
#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
// This file contains helper functions to do layout transformations (tilize, untilize) and
// to compute the address map for copying activations from DRAM to L1
#include "llrt/tests/test_libs/conv_pattern.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt;
using namespace tt::constants;
namespace matmul {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int in0_block_w;
    int in0_num_subblocks;
    int in0_block_num_tiles;
    int in0_subblock_num_tiles;
    int in1_num_subblocks;
    int in1_block_num_tiles;
    int in1_per_core_w;
    int num_blocks;
    int out_subblock_h;
    int out_subblock_w;
    int out_subblock_num_tiles;
};
}
int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ConvParameters conv_params = ConvParameters(3, 3, 1, 1, 0, 0);
        std::array<uint32_t, 4> act_shape = {1, 64, 10, 10};
        std::array<uint32_t, 4> weight_shape = {576, 64, 3, 3};
        tt::Tensor<bfloat16> tensor = tt::initialize_tensor<bfloat16>(act_shape, tt::Initialize::RANDOM, std::chrono::system_clock::now().time_since_epoch().count());
        std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
        bfloat16 pad_value = (uint32_t) 0;
        // convolution input is padded on the host. TODO: padding should be moved to device reader kernel
        tt::Tensor<bfloat16> tensor_padded = tt::pad(tensor, pad_size, pad_value);
        auto tensor_p = tt::permute(tensor_padded, {0, 2, 3, 1}); // NHWC
        // Overwrite the weight tensor with identity matrix after intializing it.
        tt::Tensor<bfloat16> weight_tensor = tt::initialize_tensor<bfloat16>(weight_shape, tt::Initialize::ZEROS);
        auto weight_tensor_p = tt::permute(weight_tensor, {0, 2, 3, 1}); // NHWC

        // generate address map to generic reader kernel
        std::tuple<uint32_t, uint32_t, uint32_t, std::vector<uint32_t>> addr_ = gen_source_addresses_for_conv_act_layout_transform(tensor_p.get_shape(), conv_params, sizeof(bfloat16));
        auto num_tiles_generated_with_source_addresses = std::get<0>(addr_);
        auto num_addresses_per_tile = std::get<1>(addr_);
        auto dram_read_size_bytes = std::get<2>(addr_);
        auto source_addresses = std::get<3>(addr_);
        // The source addresses are addresses for convolution activation in DRAM
        // It is used by the generic reader kernel. The source addresses are arranged in the order of tiles.
        // The dram read size is fixed to 16 elements which is one row of face within a tile .
        // The kernel determines the L1 address as it writes to contingous locations in L1 buffer.

        // vector to be copied to DRAM
        auto src_vec = tensor_p.get_values();
        // This will create the 2D matrix by modeling what dram to l1 read patterns are
        auto golden_act_matrix_tilized = move_act_dram_to_l1_tilized(tensor_p, dram_read_size_bytes, source_addresses);
        // This would be the actual golden that we compare the activation data
        auto golden_act_matrix = move_act_dram_to_l1(tensor_p, conv_params);
        auto golden_act_vector = flatten(golden_act_matrix);
        std::uint32_t act_rows = golden_act_matrix.size();
        std::uint32_t act_cols = golden_act_matrix.at(0).size();
        // Sanity check to verify address map.
        auto golden_act_untilized = untilize_act(golden_act_matrix_tilized, act_rows, act_cols);
        assert(golden_act_vector == golden_act_untilized);
        auto weight_matrix_ = move_weights_dram_to_l1_mm(weight_tensor_p);
        std::uint32_t weight_rows = weight_matrix_.size();
        std::uint32_t weight_cols = weight_matrix_.at(0).size();
        // For zero weight test -
        //auto weight_vector = flatten(weight_matrix_);
        // For identity test - Creating a new identity weight matrix
        auto weight_vector = create_identity_matrix(weight_rows, weight_cols, std::min(weight_rows, weight_cols));
        // tilize weights to be copied to DRAM
        // TODO: should we do this on device when reading from DRAM to L1?
        auto weights_tilized = tilize(weight_vector, weight_rows, weight_cols);
        std::array<uint32_t, 4> output_shape = {1, 1, act_rows, weight_cols};
        // For identity test -
        auto golden_output_vec = golden_act_vector;
        // For zero weight test -
        //tt::Tensor<bfloat16> golden_output_tensor = tt::initialize_tensor<bfloat16>(output_shape, tt::Initialize::ZEROS);
        //auto golden_output_vec = golden_output_tensor.get_values();

        uint32_t single_tile_size = 2 * 1024;
        assert(act_rows % TILE_HEIGHT == 0);
        assert(act_cols % TILE_WIDTH == 0);
        assert(weight_rows % TILE_HEIGHT == 0);
        assert(weight_cols % TILE_WIDTH == 0);
        std::uint32_t num_tiles_rows = act_rows / TILE_HEIGHT;
        std::uint32_t num_tiles_cols = act_cols / TILE_WIDTH;
        std::uint32_t num_tiles = num_tiles_rows * num_tiles_cols;
        std::uint32_t w_num_tiles_rows = weight_rows / TILE_HEIGHT;
        std::uint32_t w_num_tiles_cols = weight_cols / TILE_WIDTH;
        std::uint32_t w_num_tiles = w_num_tiles_rows * w_num_tiles_cols;
        assert(act_cols == weight_rows);
        assert(num_tiles == num_tiles_generated_with_source_addresses);
        uint32_t output_rows = act_rows;
        uint32_t output_cols = weight_cols;
        uint32_t output_tiles_rows = num_tiles_rows;
        uint32_t output_tiles_cols = w_num_tiles_cols;


        uint32_t M = num_tiles_rows;
        uint32_t K = num_tiles_cols;
        uint32_t N = w_num_tiles_cols;
        int out_subblock_h = 2;
        int out_subblock_w = 3;
        int in0_block_w = 1;


        int num_blocks = K/in0_block_w;
        uint32_t src0_num_tiles_per_block = M * in0_block_w;
        uint32_t src1_num_tiles_per_block = N * in0_block_w;
        // src0_num_reads_per_block is the number of DRAM reads issued to produce 1 block
        uint32_t src0_num_reads_per_block = src0_num_tiles_per_block * num_addresses_per_tile;
        assert(source_addresses.size() == num_blocks * src0_num_reads_per_block);
        uint32_t src0_num_bytes_per_block = src0_num_tiles_per_block * single_tile_size;
        uint32_t src1_num_bytes_per_block = src1_num_tiles_per_block * single_tile_size;

        ll_buda::Program *program = new ll_buda::Program();
        tt_xy_pair core = {0, 0};
        uint32_t dram_buffer_src0_size = tensor_p.get_volume() * sizeof(bfloat16);
        uint32_t dram_buffer_src1_size = weights_tilized.size() * sizeof(bfloat16);
        uint32_t dram_buffer_dst_size = M * N * single_tile_size;

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 0;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_src0_size, dram_buffer_src0_addr);
        auto src1_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_src1_size, dram_buffer_src1_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_dst_size, dram_buffer_dst_addr);

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t cb0_index = 0;
        uint32_t cb0_addr = 200 * 1024;
        uint32_t num_cb0_tiles = M * in0_block_w * 2;
        uint32_t cb0_size = num_cb0_tiles * single_tile_size;
        uint32_t source_addresses_in_l1_addr = cb0_addr + cb0_size;
        auto cb0 = ll_buda::CreateCircularBuffer(
            program,
            cb0_index,
            core,
            num_cb0_tiles,
            cb0_size,
            cb0_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t cb1_index = 1;
        uint32_t cb1_addr = 300 * 1024;
        uint32_t num_cb1_tiles = N * in0_block_w * 2;
        uint32_t cb1_size = num_cb1_tiles * single_tile_size;
        auto cb1 = ll_buda::CreateCircularBuffer(
            program,
            cb1_index,
            core,
            num_cb1_tiles,
            cb1_size,
            cb1_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        uint32_t cb_output_size = num_output_tiles * single_tile_size;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = ll_buda::CreateCircularBuffer(
            program,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );
        std::vector<uint32_t> generic_binary_reader_args {
            dram_buffer_src0_addr,
            (uint32_t)dram_src0_noc_xy.x,
            (uint32_t)dram_src0_noc_xy.y,
            dram_buffer_src1_addr,
            (uint32_t)dram_src1_noc_xy.x,
            (uint32_t)dram_src1_noc_xy.y,
            (uint32_t)source_addresses.size(),
            (uint32_t)source_addresses_in_l1_addr,
            (uint32_t)num_blocks,
            src0_num_reads_per_block,
            dram_read_size_bytes,
            src1_num_bytes_per_block,
            src0_num_tiles_per_block,
            src1_num_tiles_per_block};

        auto generic_binary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/generic_binary_reader_blocked.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        std::vector<uint32_t> writer_rt_args{
            dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            (std::uint32_t)out_subblock_h, // num tiles per sub block m
            (std::uint32_t)out_subblock_w, // num tiles per sub block n
            (std::uint32_t)M/out_subblock_h, // num sub blocks m
            (std::uint32_t)N/out_subblock_w, // num sub blocks n
            (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
            (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
            (std::uint32_t)out_subblock_w*single_tile_size}; // bytes offset to next sub-block

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unswizzle.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        int in0_num_subblocks = (M/out_subblock_h);
        int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        int in1_num_subblocks = (N/out_subblock_w);
        int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;

        int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

        void *hlk_args = new matmul::hlk_args_t{
            .in0_block_w = in0_block_w,
            .in0_num_subblocks = in0_num_subblocks,
            .in0_block_num_tiles = in0_block_num_tiles,
            .in0_subblock_num_tiles = in0_subblock_num_tiles,

            .in1_num_subblocks = in1_num_subblocks,
            .in1_block_num_tiles = in1_block_num_tiles,
            .in1_per_core_w = in1_per_core_w,

            .num_blocks = num_blocks,

            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .out_subblock_num_tiles = out_subblock_num_tiles
        };
        ll_buda::ComputeKernelArgs *mm_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(matmul::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto mm_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/matmul_large_block_zm.cpp",
            core,
            mm_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        std::cout << "GOING TO COMPILE PROGRAM." << std::endl;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);
        std::cout << "DONE COMPILING THE PROGRAM. GOING TO WRITE TO DRAM." << std::endl;
        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        auto activations = pack_bfloat16_vec_into_uint32_vec(src_vec);
        pass &= ll_buda::WriteToDeviceDRAM(device, src0_dram_buffer, activations);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tilized);
        pass &= ll_buda::WriteToDeviceDRAM(device, src1_dram_buffer, weights);
        std::cout << "DONE WRITING TO DEVICE. GOING TO CONFIGURE DEVICE WITH PROGRAM" << std::endl;
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            generic_binary_reader_kernel,
            core,
            generic_binary_reader_args);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            writer_rt_args);
        std::cout << "DONE DEVICE CONFIGURE. GOING TO WRITE address map TO DEVICE L1" << std::endl;
        ll_buda::WriteToDeviceL1(device, core, source_addresses, source_addresses_in_l1_addr);

        // DEBUG
        // Sanity check to verify address map in L1
        std::vector<uint32_t> source_addresses_in_l1;
        ll_buda::ReadFromDeviceL1(device, core, source_addresses_in_l1_addr, source_addresses_in_l1, source_addresses.size() * sizeof(uint32_t));
        assert(source_addresses == source_addresses_in_l1);
        // END DEBUG

        std::cout << "DONE WRITING address map TO DEVICE L1. GOING TO LAUNCH KERNELS" << std::endl;
        pass &= ll_buda::LaunchKernels(device, program);
        std::cout << "DONE KERNELS. GOING TO READ FROM DRAM." << std::endl;

        std::vector<uint32_t> result_uint32;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_uint32, dst_dram_buffer->size());
        auto result_vec_tilized = unpack_uint32_vec_into_bfloat16_vec(result_uint32);
        assert(golden_act_matrix_tilized.size() == result_vec_tilized.size());
        auto result_vec = untilize(result_vec_tilized, act_rows, weight_cols);
        std::cout << "DONE READING FROM DRAM. GOING TO VALIDATE NOW." << std::endl;
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        assert(golden_output_vec.size() == result_vec.size());
        pass &= (golden_output_vec == result_vec);
        pass &= ll_buda::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
