// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "impl/device/device.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

inline std::vector<bfloat16> create_random_vector_of_bfloat16_integer(uint32_t num_bytes, uint16_t rand_max_float, int seed) {
     auto rand_float = std::bind(std::uniform_real_distribution<float>(1000, rand_max_float), std::mt19937(seed));

    std::vector<bfloat16> vec(num_bytes/sizeof(bfloat16), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = (int)rand_float();
        vec[i] = bfloat16(num_1_float);
    }
    log_info(tt::LogVerif, "Created a random vector of size {}", vec.size());
    return vec;
}

inline void print_vec_of_bfloat16_NoComma(std::vector<bfloat16> vec, int num_tiles, std::string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << vec.at(idx).to_float() << " " ;
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Function to perform matrix multiplication
void golden_matmul(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output,
                   uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    // Indices for accessing elements in matrices
    std::uint32_t idx_c = 0; // Index for matrix C (output)
    std::uint32_t idx_a = 0; // Index for matrix A
    std::uint32_t idx_b = 0; // Index for matrix B

    // Temporary variables for floating-point calculations
    float c_f; // Accumulator for the dot product result
    float float_tmp; // Temporary variable for intermediate multiplication results

    // Initialize the output matrix with zeros
    std::vector<bfloat16> c_bf(M * N, 0);

    // Loop over each row of matrix A
    for (int i = 0; i < M; i++) {
        // Loop over each column of matrix B
        for (int j = 0; j < N; j++) {
            // Calculate the index for the current element in the output matrix
            idx_c = j + (i * N);
            // Initialize indices for the current row of A and column of B
            idx_a = i * K;
            idx_b = j;
            // Reset the accumulator for the current element in the output matrix
            c_f = 0;

            // Loop over each element in the current row of A and column of B
            for (int k_m = 0; k_m < K; k_m++) {
                // Convert bfloat16 to float and multiply the corresponding elements
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                // Accumulate the result
                c_f += float_tmp;
                // Move to the next element in the row of A and column of B
                idx_a += 1;
                idx_b += K;
            }

            // Convert the accumulated float result back to bfloat16 and store it in the output matrix
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}


void matmul_single_core(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    * Core range is just single core
    */
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    /*
    * EXtracting Matrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = 2 * 32 * 32;

    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    /* DRAM buffer size = input full size */
    /* limiting page_size = single tile size; to allow DRAM channels interleaving */

    tt_metal::InterleavedBufferConfig dram_config_A{
                    .device= device,
                    .size = dram_buffer_A_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_B{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_C{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/matmul_single_core_decode/kernels/dataflow/reader_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/matmul_single_core_decode/kernels/dataflow/writer_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        B, // B
        Mt, // Mt
        Kt, // Kt
        Nt // Nt
    };
    auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/matmul_single_core_decode/kernels/compute/bmm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}
    
    );


    /*
    * Kernels - Runtime arguments
    */
    tt_metal::SetRuntimeArgs(
        program, reader_id, core,
        {src0_addr, src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
    );

    tt_metal::SetRuntimeArgs(
        program, writer_id, core,
        {dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
    );

    /* Launch program & read in output buffer result into the host vector */
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}


// Function to print a bfloat16 vector as a 2D matrix
void printBFloat16Matrix(const std::vector<bfloat16>& vec, uint32_t M, uint32_t K) {
    if (vec.size() != M * K) {
        std::cerr << "Error: Vector size does not match the specified dimensions." << std::endl;
        return;
    }

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            // Convert bfloat16 to float for printing
            float value = vec[i * K + j].to_float();
            std::cout << std::setprecision(2) << value << " ";
        }
        std::cout << std::endl;
    }
}




///////////////////////////////////////



int main(int argc, char **argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        /* Create source data */
        constexpr uint32_t M = 64;  // user-defined
        constexpr uint32_t N = 64;  // user-defined
        constexpr uint32_t K = 64;  // user-defined
        constexpr uint32_t B = 1;  // user-defined

        // We are multiplying a matrix of size A(mxk) and B(kxn) to get a C(mxn) matrix 

        // Mt , Kt and Nt determine the number of tiles for given problem
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        // tile size assuming its in 16bits ie 2 bytes
        constexpr uint32_t single_tile_size = 2 * 1024;
         
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors with various ranges of values */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_integer(dram_buffer_A_size, 9999, 123);
        std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_integer(dram_buffer_B_size, 9999, 12522);

         /* Golden Matmul running on CPU (Float)*/
        vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, B);

        /* Input vector tilizing */
        // printf("Before tilize operations\n");
        // printBFloat16Matrix(src0_vec, M,K);
        tilize(src0_vec, M, K);
        printf("After tilize operations\n");
        print_vec_of_bfloat16_NoComma(src0_vec, Mt*Kt,"A tiles");

        tilize(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buffer_C_size/sizeof(bfloat16));
        matmul_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, B, device);
        untilize(result_vec, M, N);

        log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
