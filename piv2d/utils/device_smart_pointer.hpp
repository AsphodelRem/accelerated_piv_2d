#pragma once

#include <new>
#include <string>
#include <iostream>

template <typename T>
class SharedPtrGPU
{
public:
    SharedPtrGPU() : _owners(new unsigned long long(0)){};

    SharedPtrGPU(size_t size_in_bytes) : _size(size_in_bytes), _owners(new unsigned long long(1))
    {
        auto status = cudaMalloc((void **)(&this->_raw_data_ptr), this->_size);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to allocate " + std::to_string(this->_size) + " on device");
        }

        this->increment();
    }

    SharedPtrGPU(SharedPtrGPU &another)
    {
        if (&another != this)
        {
            this->_raw_data_ptr = another._raw_data_ptr;
            this->_size = another._size;
            this->_owners = another._owners;

            this->increment();
        }
    }

    SharedPtrGPU &operator=(const SharedPtrGPU &another)
    {
        this->decrement();

        this->_raw_data_ptr = another._raw_data_ptr;
        this->_owners = another._owners;
        this->_size = another._size;

        this->increment();

        return *this;
    }

    ~SharedPtrGPU()
    {
        this->decrement();
    }

    inline T *get() const
    {
        return this->_raw_data_ptr;
    }

    T *operator->() const
    {
        return this->_raw_data_ptr;
    }

    T &operator*() const
    {
        return this->_raw_data_ptr;
    }

    inline unsigned long long size() const
    {
        return this->_size;
    }

    SharedPtrGPU &copyDataToHost(void *destination)
    {
        auto status = cudaMemcpy(destination, this->_raw_data_ptr, this->size(), cudaMemcpyDeviceToHost);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to copy data to host. Memcpy status: " + std::to_string(status));
        }

        return *this;
    }

    SharedPtrGPU &uploadHostData(void *source, size_t size_in_bytes)
    {
        if (size_in_bytes > this->_size)
        {
            throw std::runtime_error("Size of the host data must be less or equal to size of the allocated memory on device");
        }

        auto status = cudaMemcpy(this->_raw_data_ptr, source, size_in_bytes, cudaMemcpyHostToDevice);

        if (status != cudaSuccess)
        {
            throw std::runtime_error("Unable to copy data to device. Memcpy status: " + std::to_string(status));
        }

        return *this;
    }

private:
    T *_raw_data_ptr;
    unsigned long long *_owners;
    unsigned long long _size;

    void decrement()
    {
        (*this->_owners)--;

        if ((*_owners) == 0)
        {
            cudaFree(this->_raw_data_ptr);
            delete this->_owners;
            delete this;
        }
    }

    void increment()
    {
        (*this->_owners)++;
    }
};

template <typename T>
SharedPtrGPU<T> make_shared_gpu(unsigned long long number_of_elements)
{
    return SharedPtrGPU<T>(number_of_elements * sizeof(T));
}