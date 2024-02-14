#include "rtabmap/core/region/models/Classifier.h"
#include "rtabmap/core/region/models/IncrementalLinear.h"

namespace rtabmap
{

    ClassifierImpl::ClassifierImpl(const std::vector<size_t> &layer_dims,
                                   size_t out_features) : output_layer(layer_dims.back(), out_features)

    {
        for (size_t i = 0; i < layer_dims.size() - 1; i++)
        {
            this->mlp->push_back(torch::nn::Linear(layer_dims[i], layer_dims[i + 1]));
            this->mlp->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        }

        this->register_all_();
    }

    torch::Tensor ClassifierImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->mlp->forward(input);
        x = this->output_layer->forward(x);
        return torch::sigmoid(x);
    }

    void ClassifierImpl::adapt(const torch::Tensor &classes_in_this_experience)
    {
        this->output_layer->adapt(classes_in_this_experience);
    }

    void ClassifierImpl::reset()
    {
        this->rebuild_all_();
        this->register_all_();
    }

    void ClassifierImpl::rebuild_all_()
    {
        this->mlp = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(this->mlp->clone());
        this->output_layer = std::dynamic_pointer_cast<IncrementalLinearImpl>(this->output_layer->clone());
    }

    void ClassifierImpl::register_all_()
    {
        register_module("mlp", this->mlp);
        for (int i = 0; i < this->mlp->children().size(); i += 2)
        {
            int idx = i / 2;
            register_module("linear" + std::to_string(idx), this->mlp->children().at(i));
            register_module("relu" + std::to_string(idx), this->mlp->children().at(i + 1));
        }
        register_module("output_layer", this->output_layer);

        // for (const auto &m : this->mlp->named_modules())
        // {
        //     m.value()->to(device, true);
        // }
        // for (const auto &m : this->output_layer->named_modules())
        // {
        //     m.value()->to(device, true);
        // }
    }
}