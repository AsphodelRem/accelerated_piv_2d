#include "image_container.cuh"
#include "parameters.cuh"

ImageContainer::ImageContainer(ListOfFiles &file_names, PIVParameters &params) : _file_names(file_names), _parameters(params)
{
    if (file_names.size() == 0 || file_names.size() % 2 != 0)
    {
        throw std::runtime_error("No files");
    }

    auto file_1 = this->_file_names.front();
    auto file_2 = this->_file_names.front();

    this->inputImages = ImagePair<unsigned char>(file_1, file_2);
    this->outputImages = SlicedImagePair<unsigned char, float>(this->inputImages, params);
}

bool ImageContainer::isEmpty()
{
    return this->_file_names.empty();
}

// template <typename T, typename T2>
SlicedImagePair<unsigned char, float> &ImageContainer::getImages()
{
    if (!this->isEmpty())
    {
        auto file_1 = this->_file_names.front();
        this->_file_names.pop();

        auto file_2 = this->_file_names.front();
        this->_file_names.pop();

        this->inputImages.uploadNewImages(file_1, file_2);
        this->outputImages.uploadNewImages(this->inputImages);
    }

    return this->outputImages;
}
