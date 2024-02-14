#pragma once

#ifndef EXPERIENCE_DATASET_H
#define EXPERIENCE_DATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace rtabmap
{

    class ExperienceDataset : public torch::data::Dataset<ExperienceDataset> /* : public IdDataset */
    {
    public:
        ExperienceDataset(const std::vector<size_t> &ids,
                          const std::vector<cv::Mat> &images, // in rtabmap probably a vector of images if more stream are available
                          const std::vector<size_t> &labels,
                          const int64_t &image_width = 224,
                          const int64_t &image_height = 224);

        inline torch::optional<size_t> size() const override
        {
            return this->dataset_size_;
        }

        torch::data::Example<> get(size_t index) override;

        inline const std::vector<size_t> &ids() const { return this->ids_; }
        inline const torch::Tensor &images() const { return this->images_; }
        inline const torch::Tensor &labels() const { return this->labels_; }

    private:
        std::vector<size_t> ids_;
        torch::Tensor images_;
        torch::Tensor labels_;

        size_t dataset_size_;
    };

}

#endif