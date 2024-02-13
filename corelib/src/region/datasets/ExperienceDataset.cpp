#include "rtabmap/core/region/datasets/ExperienceDataset.h"
#include "rtabmap/core/region/utils.h"

namespace region
{

    ExperienceDataset::ExperienceDataset(const std::vector<size_t> &ids,
                                         const std::vector<cv::Mat> &images,
                                         const std::vector<size_t> &labels,
                                         const int64_t &image_width,
                                         const int64_t &image_height) : ids_(ids), dataset_size_(labels.size()) /* : IdDataset::IdDataset(ids) */
    {
        std::vector<torch::Tensor> images_tensor(images.size());
        std::vector<torch::Tensor> labels_tensor(labels.size());

        for (int i = 0; i < images.size(); i++)
        {
            images_tensor[i] = image_to_tensor(images[i], image_width, image_height);
            labels_tensor[i] = torch::tensor(static_cast<float>(labels[i]));
        }

        this->images_ = torch::cat(images_tensor);
        this->labels_ = torch::cat(labels_tensor);
    }

    torch::data::Example<> ExperienceDataset::get(size_t index)
    {

        return torch::data::Example<>(this->images_[index], this->labels_[index]);
        // return torch::data::Example<torch::Tensor, torch::Tensor>(torch::Tensor({this->ids_[index], torch::Tensor({this->images_[index]})}), this->labels_[index];
        // torch::Tensor tensor = torch::tensor(static_cast<float>(this->ids_[index]));
        // return torch::data::Example<>(torch::Tensor({this->ids_[index], torch::Tensor({this->images_[index]})}), this->labels_[index]
        // );
    }
}
