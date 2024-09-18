/**
 * @file histogrm1d_single_core.cpp
 * @author Neelakandan Ramachandran (neelakandan.ramachandran@kla-tencor.com)
 * @brief
 * @version 0.1
 * @date 2024-09-04
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

/**
 * @brief Create a random vector of bfloat16 integeral numbers
 *
 * @param num_bytes
 * @param rand_min_float
 * @param rand_max_float
 * @param seed
 * @return std::vector<bfloat16>
 */
inline std::vector<bfloat16> create_random_vector_of_bfloat16_integer(
    uint32_t num_bytes, float rand_min_float, float rand_max_float, int seed) {
    auto rand_float =
        std::bind(std::uniform_real_distribution<float>(rand_min_float, rand_max_float), std::mt19937(seed));
    std::vector<bfloat16> vec(num_bytes / sizeof(bfloat16), 0);
    for (int i = 0; i < vec.size(); i++) {
        // Casting to integer to get integral numbers
        float num_1_float = (int)rand_float();
        vec[i] = bfloat16(num_1_float);
    }
    log_info(tt::LogVerif, "Created a random vector of size {}", vec.size());
    return vec;
}

/**
 * @brief
 *
 * @param vec
 * @param num_tiles
 * @param name
 * @param tile_print_offset
 */
inline void print_vec_of_bfloat16_t(
    std::vector<bfloat16> vec, int num_tiles, string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << vec.at(idx).to_float() << " ";
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Assuming bfloat16 is a custom type with a method to convert to float.
int bfloat16_to_int(bfloat16 value) {
    // Convert bfloat16 to float first, then to int.
    return static_cast<int>(value.to_float());
}

/**
 * @brief
 *
 * @param flatImages
 * @param histogram
 * @param I
 * @param imageSize
 * @param imageIndex
 */
void calculateHistogram(
    const std::vector<bfloat16>& flatImages, std::vector<int>& histogram, int I, size_t imageSize, size_t imageIndex) {
    // Ensure the histogram is initialized with zeros.
    histogram.assign(I, 0);

    // Calculate the start index for the current image.
    size_t startIndex = imageIndex * imageSize;

    // Iterate over each pixel in the image and increment the corresponding histogram bin.
    for (size_t i = startIndex; i < startIndex + imageSize; ++i) {
        // Use the conversion function to get the integer value of the pixel.
        int pixelValue = bfloat16_to_int(flatImages[i]);
        histogram[pixelValue - 1]++;
    }
}

/**
 * @brief Calculate histogram of intensity values given an input flattened vector of Image intensitiies
 *
 * @param flatImages
 * @param goldenHistogram
 * @param I
 * @param imageSize
 * @param batchCount
 */
void calculateHistogramBatched(
    const std::vector<bfloat16>& flatImages,
    std::vector<int>& goldenHistogram,
    int I,
    size_t imageSize,
    size_t batchCount) {
    // Initialize the golden histogram with zeros.
    goldenHistogram.assign(I, 0);

    // Temporary histogram to store the histogram of a single image.
    std::vector<int> tempHistogram(I, 0);

    // Iterate over each image in the batch.
    for (size_t imageIndex = 0; imageIndex < batchCount; ++imageIndex) {
        // Calculate the histogram for the current image.
        calculateHistogram(flatImages, tempHistogram, I, imageSize, imageIndex);

        // Combine the current image's histogram with the golden histogram.
        for (int i = 0; i < I; ++i) {
            goldenHistogram[i] += tempHistogram[i];
        }
    }
}

/**
 * @brief Histogram1D device setup and kernel invocations
 *
 * @param inImage Input image
 * @param outputHistogram  Output histogram
 * @param W  : Width of image
 * @param H  : Hight of image
 * @param I  : Intensity levels : 1 to I
 * @param B  : Batch size
 * @param device : Target Device id
 */
void histogram1d_single_core(
    std::vector<bfloat16>& inImage,
    std::vector<bfloat16>& outputHistogram,
    std::vector<bfloat16>& outImage,
    uint32_t W,
    uint32_t H,
    uint32_t I,
    uint32_t B,
    float scaler,
    Device* device) {
    /*
     * Setup program to execute along with its buffers and kernels to use
     * Core range is just single core
     */
    CommandQueue& cq = device->command_queue();
    Program program{};
    constexpr CoreCoord core = {0, 0};

    /*
     * Extracting Matrix dimensions from input/output vectors
     */
    // inpImage = W*H
    // outHistogram = I
    uint32_t Wt = W / TILE_HEIGHT;
    uint32_t Ht = H / TILE_WIDTH;
    uint32_t It = I / TILE_HW;

    /*
     * Create DRAM Buffers for input and output vectors
     * Writing data from input vectors to source buffers
     */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = 2 * 32 * 32;

    // Setup src image and dst histogram sizes
    uint32_t dram_buffer_src_image_size = single_tile_size * Wt * Ht;
    uint32_t dram_buffer_dst_image_size = single_tile_size * Wt * Ht;
    uint32_t dram_buffer_dst_histogram_size = single_tile_size * It;

    /* DRAM buffer size = input full size */
    /* limiting page_size = single tile size; to allow DRAM channels interleaving */
    tt_metal::InterleavedBufferConfig dram_config_src_image{
        .device = device,
        .size = dram_buffer_src_image_size,
        .page_size = dram_buffer_src_image_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_dst_image{
        .device = device,
        .size = dram_buffer_dst_image_size,
        .page_size = dram_buffer_dst_image_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_dst_histogram{
        .device = device,
        .size = dram_buffer_dst_histogram_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create buffer for the src image and dst histogram on the device dram
    std::shared_ptr<tt::tt_metal::Buffer> src_image_dram_buffer = CreateBuffer(dram_config_src_image);
    std::shared_ptr<tt::tt_metal::Buffer> dst_image_dram_buffer = CreateBuffer(dram_config_dst_image);
    std::shared_ptr<tt::tt_metal::Buffer> dst_histogram_dram_buffer = CreateBuffer(dram_config_dst_histogram);

    // Get the src and dst dram address
    uint32_t src_image_addr = src_image_dram_buffer->address();
    uint32_t dst_image_addr = dst_image_dram_buffer->address();
    uint32_t dst_hist_addr = dst_histogram_dram_buffer->address();

    /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */
    uint32_t src0_cb_index = CB::c_in0;  // 0
    uint32_t num_input_tiles = 4;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_out0;  // 16
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CB::c_out1;  // 17 output operands start at index 17
    uint32_t num_output_tiles = 4;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    uint32_t intermed_cb_index0 = CB::c_intermed0;
    CircularBufferConfig cb_temp_reduce_tile_config0 = tt_metal::CircularBufferConfig(4 * single_tile_size, {{intermed_cb_index0, cb_data_format}})
        .set_page_size(CB::c_intermed0, single_tile_size);
    auto cb_temp_reduce_tile0 = tt_metal::CreateCircularBuffer(program, core, cb_temp_reduce_tile_config0);

    uint32_t intermed_cb_index1 = CB::c_intermed1;
    CircularBufferConfig cb_temp_reduce_tile_config1 = tt_metal::CircularBufferConfig(4 * single_tile_size, {{intermed_cb_index1, cb_data_format}})
        .set_page_size(CB::c_intermed1, single_tile_size);
    auto cb_temp_reduce_tile1 = tt_metal::CreateCircularBuffer(program, core, cb_temp_reduce_tile_config1);

    /*
     * Specify data movement kernels for reading/writing data to/from
     * DRAM.
     */
    KernelHandle input_image_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/kla/histogram1d_single_core/kernels/dataflow/input_image_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle output_image_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/kla/histogram1d_single_core/kernels/dataflow/output_image_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /*
     * Set the parameters that the compute kernel will use.
     */
    vector<uint32_t> compute_args = {Wt * Ht};

    auto histogram_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/kla/histogram1d_single_core/kernels/compute/histogram_compute1d.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}

    );

    // auto reduction_single_core_kernel_id = tt_metal::CreateKernel(
    //     program,
    //     "tt_metal/programming_examples/kla/histogram1d_single_core/kernels/compute/reduce_bins.cpp",
    //     core,
    //     tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}

    // );


    // std::cout << "Batch size" << B << std::endl;
    /*
     * Configure program and runtime kernel arguments, then execute.
     */
    // tt_metal::SetRuntimeArgs(program, input_image_reader_kernel_id, core, {src_image_addr, Wt, Ht, B});

    tt_metal::SetRuntimeArgs(
        program,
        input_image_reader_kernel_id,
        core,
        {
            src_image_dram_buffer->address(),
            static_cast<uint32_t>(src_image_dram_buffer->noc_coordinates().x),
            static_cast<uint32_t>(src_image_dram_buffer->noc_coordinates().y),
            Wt * Ht,
            *reinterpret_cast<uint32_t*>(&scaler)
        });

    SetRuntimeArgs(
        program,
        output_image_writer_kernel_id,
        core,
        {
            dst_image_addr,
            static_cast<uint32_t>(dst_image_dram_buffer->noc_coordinates().x),
            static_cast<uint32_t>(dst_image_dram_buffer->noc_coordinates().y),
            Wt * Ht,  // Total number of tiles to write to output image
        });

    // Host to Device DRAM Copy
    EnqueueWriteBuffer(cq, src_image_dram_buffer, inImage.data(), false);

    EnqueueProgram(cq, program, false);

    // Device to Host DRAM Copy
    EnqueueReadBuffer(cq, dst_image_dram_buffer, outImage.data(), true);
}

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        // Silicon accelerator setup
        constexpr int device_id = 0;
        Device* device = CreateDevice(device_id);
        // Configuration for image size and intensity levels.
        constexpr uint32_t W = 32;    // Width of each image.  (User defined)
        constexpr uint32_t H = 32;    // Height of each image. (User defined)
        constexpr uint32_t I = 1024;  // Number of intensity levels (e.g., 256 for 8-bit images).
        constexpr uint32_t B = 1;     // Batch Size
        float scaler = 1.0f;   // To be used with the reduction tile logic to multiply the input by scale and perform reduction
        bfloat16 bfloat_scaler_value = bfloat16(scaler);
        uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

        // Calculating the total number of tiles in both width and height directions
        uint32_t Wt = W / TILE_WIDTH;
        uint32_t Ht = H / TILE_HEIGHT;
        uint32_t It = I / TILE_HW;

        // tile size assuming its in 16bits ie 2 bytes
        constexpr uint32_t single_tile_size = 2 * 32 * 32;

        // Buffer size for Num_tiles of FP16_B
        uint32_t dram_buffer_src_size = B * single_tile_size * Wt * Ht;
        uint32_t dram_buffer_histogram_size = B * single_tile_size * It;

        // Create a batch of images with random pixel intensities.
        std::vector<bfloat16> flatImages = create_random_vector_of_bfloat16_integer(dram_buffer_src_size, 1, I, 123);

        // print the created vector
        print_vec_of_bfloat16_t(flatImages, Wt * Ht, "Input tiles");

        // Vector to store the golden histogram.
        std::vector<int> goldenHistogram(I, 0);

        // Calculate the golden histogram for the batched images. Running on CPU (Host)
        calculateHistogramBatched(flatImages, goldenHistogram, I, W * H, B);

        /* Calling the histogram host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buffer_histogram_size / sizeof(bfloat16), 0);
        vector<bfloat16> output_vec(dram_buffer_src_size / sizeof(bfloat16), 0);
        // Print the golden histogram.
        // for (int i = 0; i < I; ++i) {
        //     std::cout << "Intensity " << i << ": " << goldenHistogram[i] << std::endl;
        // }

        histogram1d_single_core(flatImages, result_vec, output_vec, W, H, I, B, scaler, device);

        // print the copied vector from device registers onto dram
        print_vec_of_bfloat16_t(output_vec, Wt * Ht, "Output tiles");

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}