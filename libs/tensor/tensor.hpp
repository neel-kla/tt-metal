#pragma once

#include <vector>
#include <array>
#include <random>
#include <tuple>

#include "tensor/types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/host.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"

namespace tt {

namespace tt_metal {

// TODO: this is duplicated
enum class Initialize
{
    ZEROS = 0,
    ONES = 1,
    INCREMENT = 2,
    RANDOM = 3
};

enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1,
    CHANNELS_LAST = 2
};

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3
};

struct MemoryConfig {
    bool interleaved = true;    // Interleave the data across multiple DRAM banks
    int bank_id = -1;           // If interleaved is false this has to be specified
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
};

// Forward declarations
class Tensor;

namespace tensor_impl {
    void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

    void allocate_contiguous_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

    template <typename T>
    void write_data(Tensor &tensor, std::vector<T> &data);
}

class Tensor {

    public:
        // ======================================================================================
        //                                  Hi Level APIs
        // ======================================================================================
        Tensor(const HostBuffer& host_buffer, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout);

        // TODO(arakhmati): modify this constructor to take device buffer
        Tensor(const HostBuffer& host_buffer, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config = {.interleaved=true});

        // TODO(arakhmati): delete
        Tensor(std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout);

        // TODO(arakhmati): delete
        Tensor(std::vector<uint32_t> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config={.interleaved=true});

        // TODO(arakhmati): delete
        Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout);

        // TODO(arakhmati): delete
        Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config={.interleaved=true});

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType dtype, Layout layout);

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config={.interleaved=true});

        Tensor(const std::array<uint32_t, 4> &shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config={.interleaved=true});

        Tensor(const Tensor &other) = default;
        Tensor& operator=(const Tensor &other) = default;

        Tensor(Tensor &&other) = default;
        Tensor& operator=(Tensor &&other) = default;

        ~Tensor();

        void deallocate();

        Tensor to(Device *target_device, const MemoryConfig &mem_config={.interleaved=true}) const;

        Tensor to(Host *host) const;

        Tensor to(Layout target_layout) const;

        Tensor pad(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) const;

        Tensor unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) const;

        Tensor pad_to_tile(float pad_value) const;

        Tensor unpad_from_tile(const std::array<uint32_t, 4> &output_tensor_shape) const;

        void print(Layout print_layout=Layout::ROW_MAJOR, bool pretty_print=false) const;

        // Prints like numpy print function to make it more readable. Only supports row major layout.
        void pretty_print() const;

        // ======================================================================================
        //                                  Low Level APIs
        // ======================================================================================
        const std::array<uint32_t, 4>& reshape(int N, int C, int H, int W);

        // ======================================================================================
        //                                      Getters
        // ======================================================================================
        const std::array<uint32_t, 4>& shape() const { return this->shape_; }

        const std::array<uint32_t, 4>& strides() const { return this->strides_; }

        uint32_t volume() const { return shape_[0] * shape_[1] * shape_[2] * shape_[3]; }

        DataType dtype() const { return dtype_; }

        Layout layout() const { return layout_; }

        Device *device() const { return device_; }

        Buffer *buffer() const { return buffer_.get(); }

        HostBufferDataType* data_ptr() const { return this->data_->data(); }

        const HostBuffer& host_buffer() const { return this->data_; }

        bool is_allocated_on_host() const { return bool(this->data_); }
        bool is_allocated_on_device() const { return bool(this->buffer_); }

        bool on_host() const { return device_ == nullptr; }

        bool interleaved() const;

        BufferType buffer_type() const { return mem_config_.buffer_type; };

        // Size in bytes of a single element held in tensor
        uint32_t element_size() const;

    private:
        const std::array<uint32_t, 4> compute_strides() const {
            return {shape_[1]*shape_[2]*shape_[3], shape_[2]*shape_[3], shape_[3], 1};
        }

        friend void tensor_impl::allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

        friend void tensor_impl::allocate_contiguous_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

        template <typename T>
        friend void tensor_impl::write_data(Tensor &tensor, std::vector<T> &data);

        std::array<uint32_t, 4> shape_;             // Outer-most dimension first
        std::array<uint32_t, 4> strides_;           // Outer-most dimension first
        DataType dtype_;
        Layout layout_;

        // Host Storage
        HostBuffer data_{};                 // Unpopulated if tensor is on device

        // Device Storage
        Device *device_ = nullptr;                  // Set if tensor is allocated on device
        std::shared_ptr<Buffer> buffer_{};  // Tensors on device are backed by an underlying buffer
        MemoryConfig mem_config_;
};

}  // namespace tt_metal

}  // namespace tt
