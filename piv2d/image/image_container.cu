#include <image_container.cuh>

ImageContainer::ImageContainer(ListOfFiles &file_names,
    const PIVParameters &parameters)
    : IDataContainer(parameters)
    , file_names_(file_names)

{
    if (file_names.size() == 0 || file_names.size() % 2 != 0)
    {
        throw std::runtime_error("No files");
    }

    const std::string file_1_name = this->file_names_.front();
    const std::string file_2_name = this->file_names_.front();

    this->input_images_ = ImagePair<unsigned char>(file_1_name, file_2_name);
    this->output_images_ = SlicedImagePair<unsigned char, float>(this->input_images_, parameters);
}

bool ImageContainer::IsEmpty() const
{
    return this->file_names_.empty();
}

// template <typename T, typename T2>
SlicedImagePair<unsigned char, float> &ImageContainer::GetImages()
{
    if (!this->IsEmpty())
    {
        const std::string file_1_name =  this->file_names_.front();
        this->file_names_.pop();

        const std::string file_2_name =  this->file_names_.front();
        this->file_names_.pop();

        this->input_images_.UploadNewImages(file_1_name, file_2_name);
        this->output_images_.UploadNewImages(this->input_images_);
    }

    return this->output_images_;
}
