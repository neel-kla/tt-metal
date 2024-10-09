// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_coord.h"
#include "common/tt_backend_api_types.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class CircularBuffer {
   public:
    CircularBuffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);

    const CBHandle id() const { return id_; }

    const CoreRangeSet &core_ranges() const { return core_ranges_; }

    const CircularBufferConfig &config() const { return config_; }

    CircularBufferConfig &config() { return config_; }

    const std::unordered_set<uint32_t> &buffer_indices() const { return buffer_indices_; }

    uint32_t page_size(uint32_t buffer_index) const;

    bool globally_allocated() const { return this->config_.globally_allocated_address().has_value(); }

    uint32_t size() const { return this->config_.total_size(); }

    uint32_t num_pages(uint32_t buffer_index) const;

    DataFormat data_format(uint32_t buffer_index) const;

    const std::optional<Tile>& tile(uint32_t buffer_index) const;

    uint32_t address() const;

    bool is_on_logical_corerange(const CoreRange &logical_cr) const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    void assign_global_address();

    void set_locally_allocated_address(uint32_t address) {
        this->locally_allocated_address_ = address;
    }

   private:
    bool uses_buffer_index(uint32_t buffer_index) const;

    void invalidate_locally_allocated_address();

    const uintptr_t id_;
    const CoreRangeSet core_ranges_;
    CircularBufferConfig config_;
    std::unordered_set<uint32_t> buffer_indices_;
    // Circular buffers associated with dynamically allocated buffers pull address from `config_`
    // Locally allocated addresses are generated by the program this circular buffer belongs to
    std::optional<uint32_t> locally_allocated_address_;
    uint32_t globally_allocated_address_;
    // add a callback to invalidate circular buffer allocation
};

}  // namespace v0
}  // namespace tt::tt_metal
