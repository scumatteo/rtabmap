#include "rtabmap/core/region/models/IncrementalLinear.h"

namespace region
{
    IncrementalLinearImpl::IncrementalLinearImpl(size_t in_features,
                                                 size_t initial_out_features) : linear(torch::nn::Linear(in_features, initial_out_features))
    {
        this->register_all_();
    }

    torch::Tensor IncrementalLinearImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->linear->forward(input);
        x = torch::sigmoid(x);
        return x;
    }

    void IncrementalLinearImpl::adapt(const torch::Tensor &classes_in_this_experience)
    {
        if (classes_in_this_experience.sizes().empty())
        {
            return; // ERROR
        }
        size_t max_class_in_this_experience = torch::max(classes_in_this_experience).item().toLong();
        size_t old_n_classes = this->linear->options.out_features();
        size_t new_n_classes = std::max(old_n_classes, max_class_in_this_experience + 1);

        // no new region
        if (new_n_classes <= old_n_classes)
        {
            return;
        }

        torch::Tensor old_weights = this->linear->weight;
        torch::Tensor old_bias = this->linear->bias;

        this->linear = torch::nn::Linear(this->linear->options.in_features(), new_n_classes);

        this->linear->to(old_weights.device());

        for (size_t i = 0; i < old_n_classes; i++)
        {
            this->linear->weight[i].set_data(old_weights[i]);
            this->linear->bias[i].set_data(old_bias[i]);
        }
    }

    void IncrementalLinearImpl::reset()
    {

        this->rebuild_all_();
        this->register_all_();
    }

    void IncrementalLinearImpl::rebuild_all_()
    {
        this->linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(this->linear->clone());
    }

    void IncrementalLinearImpl::register_all_()
    {
        register_module("fc", this->linear);
    }

}