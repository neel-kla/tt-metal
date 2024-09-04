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
    auto rand_float = std::bind(std::uniform_real_distribution<float>(rand_min_float, rand_max_float), std::mt19937(seed));
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

// Then in your calculateHistogram function, use this conversion function:
void calculateHistogram(const std::vector<bfloat16>& flatImages, std::vector<int>& histogram, int intensityLevels, size_t imageSize, size_t imageIndex) {
    // Ensure the histogram is initialized with zeros.
    histogram.assign(intensityLevels, 0);

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
 * @param intensityLevels 
 * @param imageSize 
 * @param batchCount 
 */
void calculateHistogramBatched(const std::vector<bfloat16>& flatImages, std::vector<int>& goldenHistogram, int intensityLevels, size_t imageSize, size_t batchCount) {
    // Initialize the golden histogram with zeros.
    goldenHistogram.assign(intensityLevels, 0);

    // Temporary histogram to store the histogram of a single image.
    std::vector<int> tempHistogram(intensityLevels, 0);

    // Iterate over each image in the batch.
    for (size_t imageIndex = 0; imageIndex < batchCount; ++imageIndex) {
        // Calculate the histogram for the current image.
        calculateHistogram(flatImages, tempHistogram, intensityLevels, imageSize, imageIndex);

        // Combine the current image's histogram with the golden histogram.
        for (int i = 0; i < intensityLevels; ++i) {
            goldenHistogram[i] += tempHistogram[i];
        }
    }
}

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
       
        // Silicon accelerator setup
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);
        // Configuration for image size and intensity levels.
        constexpr uint32_t W = 4096; // Width of each image.  (User defined)
        constexpr uint32_t H = 4096; // Height of each image. (User defined)
        constexpr uint32_t intensityLevels = 32; // Number of intensity levels (e.g., 256 for 8-bit images).
        constexpr uint32_t batchSize = 1;

        // Calculating the total number of tiles in both width and height directions
        uint32_t Wt = W / TILE_WIDTH;
        uint32_t Ht = H / TILE_HEIGHT;

        // tile size assuming its in 16bits ie 2 bytes
        constexpr uint32_t single_tile_size = 2 * 32 * 32;

        // Buffer size for Num_tiles of FP16_B
        uint32_t dram_buffer_src_size = batchSize * single_tile_size * Wt * Ht; 

        // Create a batch of images with random pixel intensities.
        std::vector<bfloat16> flatImages = create_random_vector_of_bfloat16_integer(dram_buffer_src_size, 1, intensityLevels, 123);

        // Vector to store the golden histogram.
        std::vector<int> goldenHistogram(intensityLevels, 0);

        // Calculate the golden histogram for the batched images.
        calculateHistogramBatched(flatImages, goldenHistogram, intensityLevels, W * H, batchSize);

      
        // Print the golden histogram.
        for (int i = 0; i < intensityLevels; ++i) {
            std::cout << "Intensity " << i << ": " << goldenHistogram[i] << std::endl;
        }
       
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

    TT_FATAL(pass);

    return 0;
}