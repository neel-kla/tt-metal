#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"

using namespace tt;

namespace eltwise_binary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
    const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};
    auto ops = EltwiseOp::all();
    for (auto eltwise_op: ops) {
    log_info(LogTest, "====================================================================");
    log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);
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
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_size, dram_buffer_src0_addr);
        auto src1_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_size, dram_buffer_src1_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        auto cb_src1 = ll_buda::CreateCircularBuffer(
            program,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = 2;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto binary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            multibank ? "kernels/dataflow/reader_dual_8bank.cpp"
                      : "kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            multibank ? "kernels/dataflow/writer_unary_8bank.cpp"
                      : "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new eltwise_binary::hlk_args_t{
            .per_core_block_cnt = 2048,
            .per_core_block_size = 1
        };
        ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(eltwise_binary::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_binary.cpp",
            core,
            eltwise_binary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        eltwise_binary_kernel->add_define("ELTWISE_OP", op_id_to_op_define[eltwise_op]);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        if (multibank)
            pass &= ll_buda::WriteToDeviceDRAMChannelsInterleavedTiles(device, src0_vec, src0_dram_buffer->address());
        else
            pass &= ll_buda::WriteToDeviceDRAM(device, src0_dram_buffer, src0_vec);

        std::vector<uint32_t> src1_vec;
        if (eltwise_op == EltwiseOp::MUL)
            // TODO(AP): this doesn't provide very good coverage
            // switch to a better test with different values like in reduce
            src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
        else
            src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
        if (multibank)
            pass &= ll_buda::WriteToDeviceDRAMChannelsInterleavedTiles(device, src1_vec, src1_dram_buffer->address());
        else
            pass &= ll_buda::WriteToDeviceDRAM(device, src1_dram_buffer, src1_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles, 0});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        if (multibank)
            ll_buda::ReadFromDeviceDRAMChannelsInterleavedTiles(
                device, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        else
            ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src0_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }
    } // for EltwiseOp::all()

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
