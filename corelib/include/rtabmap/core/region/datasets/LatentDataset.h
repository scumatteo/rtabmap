#pragma once

#ifndef LATENT_DATASET_H
#define LATENT_DATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace region
{
    class LatentDataset : public torch::data::Dataset<LatentDataset>
    {

    public:
        using Map = std::unordered_map<size_t, size_t>;

        LatentDataset();

        LatentDataset(const std::vector<size_t> &ids,
                      const torch::Tensor &freezed_features,
                      const torch::Tensor &labels);

        inline torch::optional<size_t> size() const override
        {
            return this->dataset_size_;
        }
        ExampleType get(size_t index) override;

        inline const Map &id_index() const { return this->id_index_; }
        inline const std::vector<size_t> &ids() const { return this->ids_; }
        inline const torch::Tensor &freezed_features() const { return this->freezed_features_; }
        inline const torch::Tensor &labels() const { return this->labels_; }

        // void update_x_at(size_t id, const torch::Tensor &x);
        void update_y_at(size_t id, const torch::Tensor &y);
        // void update_at(size_t id, const torch::Tensor &x, const torch::Tensor &y);

        std::shared_ptr<LatentDataset> concat(const std::shared_ptr<LatentDataset> &other);
        std::shared_ptr<LatentDataset> subset(const at::Tensor &indices);

    private:
        Map id_index_;
        std::vector<size_t> ids_;
        torch::Tensor freezed_features_;
        torch::Tensor labels_;

        size_t dataset_size_;
    };
}

#endif