#include "rtabmap/core/region/datasets/LatentDataset.h"

namespace region
{
    LatentDataset::LatentDataset() : dataset_size_(0)
    {
    }

    LatentDataset::LatentDataset(const std::vector<size_t> &ids,
                                 const torch::Tensor &freezed_features,
                                 const torch::Tensor &labels) : ids_(ids),
                                                                freezed_features_(freezed_features),
                                                                labels_(labels),
                                                                dataset_size_(ids.size())
    {
        for (int i = 0; i < ids.size(); i++)
        {
            this->id_index_.insert({ids[i], i});
        }
    }

    torch::data::Example<torch::Tensor, torch::Tensor> LatentDataset::get(size_t index)
    {
        return torch::data::Example<torch::Tensor, torch::Tensor>(this->freezed_features_[index], this->labels_[index]);
    }

    // void LatentDataset::update_x_at(size_t id, const torch::Tensor &x)
    // {
    //     this->freezed_features_[this->id_index_.at(id)] = x;
    // }

    void LatentDataset::update_y_at(size_t id, const torch::Tensor &y)
    {
        this->labels_[this->id_index_.at(id)] = y;
    }

    // void LatentDataset::update_at(size_t id, const torch::Tensor &x, const torch::Tensor &y)
    // {
    //     this->update_x_at(id, x);
    //     this->update_y_at(id, y);
    // }

    std::shared_ptr<LatentDataset> LatentDataset::concat(const std::shared_ptr<LatentDataset> &other)
    {

        std::vector<size_t> new_ids;
        std::vector<torch::Tensor> new_freezed_features;
        std::vector<torch::Tensor> new_labels;

        for (const auto &i : this->id_index_)
        {
            new_ids.emplace_back(i.first);
            auto data = this->get(i.second);
            new_freezed_features.emplace_back(data.data);
            new_labels.emplace_back(data.target);
        }

        for (const auto &i : other->id_index())
        {
            if (!this->id_index_.count(i.first))
            {
                new_ids.emplace_back(i.first);
                auto data = other->get(i.second);
                new_freezed_features.emplace_back(data.data);
                new_labels.emplace_back(data.target);
            }
        }

        // for (int i = 0; i < new_freezed_features.size(); i++)
        // {
        //     std::cout << new_freezed_features[i] << "\n";
        //     std::cout << new_labels[i] << "\n";
        // }

        // std::cout << new_freezed_features.size() << "\n";
        // std::cout << new_labels.size() << "\n";

        // auto new_freezed_features_tensor = torch::stack(new_freezed_features);
        // auto new_labels_tensor = torch::stack(new_labels);
        // auto new_data =
        // for(int i = 0; i < new_ids.size(); i++){
        //     std::cout << "IND " << i << "\n";
        // }
        return std::make_shared<LatentDataset>(new_ids, torch::stack(new_freezed_features), torch::stack(new_labels));
    }

    std::shared_ptr<LatentDataset> LatentDataset::subset(const at::Tensor &indices)
    {
        
        //std::cout << "LEN " << indices.sizes() << "\n";
    
        std::vector<size_t> new_ids(indices.size(0));
        std::vector<torch::Tensor> new_freezed_features(indices.size(0));
        std::vector<torch::Tensor> new_labels(indices.size(0));

        for (int i = 0; i < indices.size(0); i++)
        {
            
            new_ids[i] = this->ids_[indices[i].item().toLong()];
            new_freezed_features[i] = this->freezed_features_[indices[i].item().toLong()];
            new_labels[i] = this->labels_[indices[i].item().toLong()];
        }

        return std::make_shared<LatentDataset>(new_ids, torch::stack(new_freezed_features), torch::stack(new_labels));
    }

}
