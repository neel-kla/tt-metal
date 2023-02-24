#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tensor/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////////////
// All buffers are double buffered
// 1. Host writes data to buffer in DRAM
// 2. dram_loader_sync kernel on logical core {0, 0} BRISC copies data from buffer
//      in step 1. to buffer in L1
// 3. remote_read_remote_write_sync kernel on logical core {0, 1} NCRISC copies data
//      from L1 buffer on core {0, 0} to L1 buffer on core {0, 1}
// 4. remote_read_remote_write_sync copies data from L1 buffer to buffer in DRAM
// 5. Host reads from buffer written to in step 4.
//////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

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
        //                      Input Data Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 1, 32, 1024 * 32};

        uint32_t seed_from_systime = std::chrono::system_clock::now().time_since_epoch().count();
        Tensor<bfloat16> tensor = initialize_tensor<bfloat16>(
            shape, Initialize::RANDOM, 100, seed_from_systime);  // TODO: make randomized!
        auto golden = tensor.get_values();
        auto src_vec = pack_bfloat16_vec_into_uint32_vec(golden);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair loader_logical_core = {0, 0};
        tt_xy_pair writer_logical_core = {0, 1};
        auto loader_worker_core = device->worker_core_from_logical_core(loader_logical_core);
        auto writer_worker_core = device->worker_core_from_logical_core(writer_logical_core);

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_input_tiles = 1024 * 1;
        uint32_t num_output_tiles = num_input_tiles;
        uint32_t dram_buffer_size =
            single_tile_size * num_output_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_src_addr = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024;  // 512 MB (upper half)
        uint32_t loader_buffer_address1 = 200 * 1024;
        uint32_t loader_buffer_address2 = 400 * 1024;
        uint32_t writer_buffer_address1 = 200 * 1024;
        uint32_t writer_buffer_address2 = 400 * 1024;
        uint32_t transient_buffer_size_tiles = 4;
        uint32_t transient_buffer_size_bytes = transient_buffer_size_tiles * single_tile_size;
        uint32_t stream_register_address1 = STREAM_REG_ADDR(0, 12);
        uint32_t stream_register_address2 = STREAM_REG_ADDR(0, 24);
        int dram_channel_id = 0;

        TT_ASSERT(num_output_tiles % transient_buffer_size_tiles == 0);

        auto input_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_src_addr);

        auto l1_b0_a = ll_buda::CreateL1Buffer(program, loader_logical_core, transient_buffer_size_bytes, loader_buffer_address1);
        auto l1_b0_b = ll_buda::CreateL1Buffer(program, loader_logical_core, transient_buffer_size_bytes, loader_buffer_address2);

        auto l1_b1_a = ll_buda::CreateL1Buffer(program, writer_logical_core, transient_buffer_size_bytes, writer_buffer_address1);
        auto l1_b1_b = ll_buda::CreateL1Buffer(program, writer_logical_core, transient_buffer_size_bytes, writer_buffer_address2);

        auto output_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
        auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

        // Loader (producer kernel) running on BRISC on logical core {0, 0}
        auto producer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/dram_loader_sync_db.cpp",
            loader_logical_core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        // Writer (consumer kernel) running on NCRISC on logical core {0, 1}
        auto consumer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/remote_read_remote_write_sync_db.cpp",
            writer_logical_core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &=
            ll_buda::WriteToDeviceDRAM(device, input_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            producer_kernel,
            loader_logical_core,
            {dram_buffer_src_addr,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            loader_buffer_address1,
            loader_buffer_address2,
            (uint32_t)writer_worker_core.x,
            (uint32_t)writer_worker_core.y,
            stream_register_address1,
            stream_register_address2,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes}
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            consumer_kernel,
            writer_logical_core,
            {loader_buffer_address1,
            loader_buffer_address2,
            (uint32_t)loader_worker_core.x,
            (uint32_t)loader_worker_core.y,
            dram_buffer_dst_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            writer_buffer_address1,
            writer_buffer_address2,
            stream_register_address1,
            stream_register_address2,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes}
        );

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, output_dram_buffer, result_vec, output_dram_buffer->size());
        auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (dst_vec == golden);

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
