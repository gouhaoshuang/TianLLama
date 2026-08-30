#include "base/buffer.h"
#include <cuda_runtime_api.h>


#include <new>        // std::bad_alloc
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::move


// 拥有型 Buffer：通过 allocator 分配并管理资源
Buffer::Buffer(
    std::size_t byte_size,
    std::shared_ptr<DeviceAllocator> allocator
):
    byte_size_(byte_size), 
    allocator_(allocator) {
    
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
}

// 借用型 Buffer：只包装外部指针，不负责释放
Buffer::Buffer(
    std::size_t byte_size,
    void* external_ptr,
    DeviceType device_type
):
    byte_size_(byte_size),
    ptr_(external_ptr),
    use_external_(true),
    device_type_(device_type){
    if (byte_size_ != 0 && ptr_ == nullptr) {
        throw std::invalid_argument(
            "A non-empty external Buffer requires a pointer"
        );
    }

    if (ptr_ != nullptr && device_type_ == DeviceType::kDeviceUnknown) {
        throw std::invalid_argument(
            "An external Buffer requires a known device type"
        );
    }
}


Buffer::Buffer( Buffer&& other)
:   byte_size_(std::exchange(other.byte_size_ , 0)),
    ptr_(std::exchange(other.ptr_ , nullptr)),
    use_external_(std::exchange(other.use_external_ , false)),
    device_type_(
        std::exchange(other.device_type_ , DeviceType::kDeviceUnknown) 
    ),
    allocator_(std::move(other.allocator_)){
}


Buffer& Buffer::operator=( Buffer&& other) {
    if(this == & other){
        return * this;
    }
    
    release_owned_memory();

    byte_size_ = std::exchange(other.byte_size_ , 0);
    ptr_ = std::exchange(other.ptr_, nullptr);
    use_external_ = std::exchange(other.use_external_, false);
    device_type_ = std::exchange(
        other.device_type_,
        DeviceType::kDeviceUnknown
    );
    allocator_ = std::move(other.allocator_);

    return *this;
}

Buffer::~Buffer(){
    release_owned_memory();
}

void Buffer::release_owned_memory() noexcept{
    if(!use_external_ && ptr_ != nullptr && allocator_ != nullptr){
        allocator_->release(ptr_);
    }

    byte_size_ = 0;
    ptr_ = nullptr;
    use_external_ = false;
    device_type_ = DeviceType::kDeviceUnknown;
    allocator_.reset();
}

void * Buffer::ptr(){
    return ptr_;
}
const void * Buffer::ptr() const{
    return ptr_;
}

size_t Buffer::byte_size() const{
    return byte_size_;
}
DeviceType Buffer::device_type() const{
    return device_type_;
}

bool Buffer::is_external() const{
    return use_external_;
}
bool Buffer::owns_memory() const{
    return !use_external_ && ptr_ != nullptr;
}
bool Buffer::empty() const{
    return ptr_ == nullptr;
}
std::shared_ptr<DeviceAllocator> Buffer::allocator() const{
    return allocator_;
}

