#pragma once

#include <string>
#include <stdexcept>

template <typename T>
class SharedPtrGPU
{
public:
    SharedPtrGPU() :  raw_data_ptr_(nullptr), owners_(new unsigned long long(0)), size_(0) {};

    SharedPtrGPU(const size_t size_in_bytes) :  owners_(new unsigned long long(1)), size_(size_in_bytes)
    {
        auto status = cudaMalloc(reinterpret_cast<void**>(&this->raw_data_ptr_), this->size_);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to allocate " + std::to_string(this->size_) + " on device");
        }

        this->Increment();
    }

    SharedPtrGPU(SharedPtrGPU &another)
    {
        if (&another != this)
        {
            this->raw_data_ptr_ = another.raw_data_ptr_;
            this->size_ = another.size_;
            this->owners_ = another.owners_;

            this->Increment();
        }
    }

    SharedPtrGPU &operator=(const SharedPtrGPU &another)
    {
        this->Decrement();

        this->raw_data_ptr_ = another.raw_data_ptr_;
        this->owners_ = another.owners_;
        this->size_ = another.size_;

        this->Increment();

        return *this;
    }

    ~SharedPtrGPU()
    {
        this->Decrement();
    }

    inline T *get() const
    {
        return this->raw_data_ptr_;
    }

    T *operator->() const
    {
        return this->raw_data_ptr_;
    }

    T &operator*() const
    {
        return *this->raw_data_ptr_;
    }

    inline unsigned long long size() const
    {
        return this->size_;
    }

    SharedPtrGPU &CopyDataToHost(void *destination)
    {
        auto status = cudaMemcpy(destination, this->raw_data_ptr_, this->size(), cudaMemcpyDeviceToHost);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to copy data to host. Memcpy status: " + std::to_string(status));
        }

        return *this;
    }

    SharedPtrGPU &UploadHostData(const void *source, size_t size_in_bytes)
    {
        if (size_in_bytes > this->size_)
        {
            throw std::runtime_error("Size of the host data must be less or equal to size of the allocated memory on device");
        }

        auto status = cudaMemcpy(this->raw_data_ptr_, source, size_in_bytes, cudaMemcpyHostToDevice);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to copy data to device. Memcpy status: " + std::to_string(status));
        }

        return *this;
    }

private:
    T *raw_data_ptr_;
    unsigned long long *owners_;
    unsigned long long size_;

    void Decrement()
    {
        --(*this->owners_);

        if ((*owners_) == 0)
        {
            cudaFree(this->raw_data_ptr_);
            delete this->owners_;
            delete this;
        }
    }

    void Increment()
    {
        ++(*this->owners_);
    }
};

template <typename T>
SharedPtrGPU<T> make_shared_gpu(unsigned long long number_of_elements)
{
    return SharedPtrGPU<T>(number_of_elements * sizeof(T));
}