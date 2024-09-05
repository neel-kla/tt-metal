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
        float num_1_float = (int)rand_float();
        vec[i] = bfloat16(num_1_float);
    }
    log_info(tt::LogVerif, "Created a random vector of size {}", vec.size());
    return vec;
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
    uint32_t W,
    uint32_t H,
    uint32_t I,
    uint32_t B,
    Device* device) {
    /*
     * Setup program to execute along with its buffers and kernels to use
     * Core range is just single core
     */
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0, 0}, {0, 0});

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
    uint32_t dram_buffer_dst_histogram_size = single_tile_size * It;

    /* DRAM buffer size = input full size */
    /* limiting page_size = single tile size; to allow DRAM channels interleaving */
    tt_metal::InterleavedBufferConfig dram_config_src_image{
        .device = device,
        .size = dram_buffer_src_image_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_dst_histogram{
        .device = device,
        .size = dram_buffer_dst_histogram_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    
    // Create buffer for the src image and dst histogram on the device dram
    std::shared_ptr<tt::tt_metal::Buffer> src_image_dram_buffer = CreateBuffer(dram_config_src_image);
    std::shared_ptr<tt::tt_metal::Buffer> dst_histogram_dram_buffer = CreateBuffer(dram_config_dst_histogram);

    // Get the src and dst dram address 
    uint32_t src0_addr = src_image_dram_buffer->address();
    uint32_t src1_addr = dst_histogram_dram_buffer->address();
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
        constexpr uint32_t W = 4096;  // Width of each image.  (User defined)
        constexpr uint32_t H = 4096;  // Height of each image. (User defined)
        constexpr uint32_t I = 1024;  // Number of intensity levels (e.g., 256 for 8-bit images).
        constexpr uint32_t batchSize = 1;

        // Calculating the total number of tiles in both width and height directions
        uint32_t Wt = W / TILE_WIDTH;
        uint32_t Ht = H / TILE_HEIGHT;
        uint32_t It = I / TILE_HW;

        // tile size assuming its in 16bits ie 2 bytes
        constexpr uint32_t single_tile_size = 2 * 32 * 32;

        // Buffer size for Num_tiles of FP16_B
        uint32_t dram_buffer_src_size = batchSize * single_tile_size * Wt * Ht;
        uint32_t dram_buffer_histogram_size = batchSize * single_tile_size * It;

        // Create a batch of images with random pixel intensities.
        std::vector<bfloat16> flatImages = create_random_vector_of_bfloat16_integer(dram_buffer_src_size, 1, I, 123);

        // print the created vector
        // print_vec_of_bfloat16(flatImages, Wt*Ht, "histogram tiles");

        // Vector to store the golden histogram.
        std::vector<int> goldenHistogram(I, 0);

        // Calculate the golden histogram for the batched images. Running on CPU (Host)
        calculateHistogramBatched(flatImages, goldenHistogram, I, W * H, batchSize);

        /* Calling the histogram host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buffer_histogram_size / sizeof(bfloat16));
        // Print the golden histogram.
        // for (int i = 0; i < I; ++i) {
        //     std::cout << "Intensity " << i << ": " << goldenHistogram[i] << std::endl;
        // }

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