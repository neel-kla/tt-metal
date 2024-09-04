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
 * @brief Create a random vector of bfloat16 with integral numbers
 * 
 * @param num_bytes 
 * @param rand_max_float 
 * @param seed 
 * @return std::vector<bfloat16> 
 */
inline std::vector<bfloat16> create_random_vector_of_bfloat16_integer(
    uint32_t num_bytes, uint16_t rand_max_float, int seed) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(1000, rand_max_float), std::mt19937(seed));
    std::vector<bfloat16> vec(num_bytes / sizeof(bfloat16), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = (int)rand_float();
        vec[i] = bfloat16(num_1_float);
    }
    log_info(tt::LogVerif, "Created a random vector of size {}", vec.size());
    return vec;
}

using namespace tt::tt_metal;

int main(int argc, char **argv) {
}