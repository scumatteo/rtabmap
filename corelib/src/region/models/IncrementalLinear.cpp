#include "rtabmap/core/region/models/IncrementalLinear.h"
#include "rtabmap/utilite/ULogger.h"

namespace rtabmap
{
    IncrementalLinearImpl::IncrementalLinearImpl(size_t in_features,
                                                 size_t initial_out_features) : linear(torch::nn::Linear(in_features, initial_out_features))
    {
        register_module("fc", this->linear);
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
        std::cout << "max_class_in_this_experience" << max_class_in_this_experience << "\n";
        size_t old_n_classes = this->linear->options.out_features();
        std::cout << "old_n_classes" << old_n_classes << "\n";
        size_t new_n_classes = std::max(old_n_classes, max_class_in_this_experience + 1);
        std::cout << "new_n_classes" << new_n_classes << "\n";

        // no new region
        if (new_n_classes <= old_n_classes)
        {
            return;
        }

        torch::Tensor old_weights = this->linear->weight;
        torch::Tensor old_bias = this->linear->bias;

        this->linear = torch::nn::Linear(this->linear->options.in_features(), new_n_classes);
        this->linear->to(old_weights.device());

        torch::Tensor new_weights = this->linear->weight.slice(0, old_n_classes, new_n_classes);
        std::cout << new_weights.sizes() << "\n";
        std::cout << old_weights.sizes() << "\n";
        new_weights = torch::cat({old_weights, new_weights});

        std::cout << new_weights.sizes() << "\n";

        torch::Tensor new_bias = this->linear->bias.slice(0, old_n_classes, new_n_classes);
        std::cout << new_bias.sizes() << "\n";
        std::cout << old_bias.sizes() << "\n";
        new_bias = torch::cat({old_bias, new_bias});

        std::cout << new_bias.sizes() << "\n";

        this->linear->weight.set_data(new_weights);
        this->linear->bias.set_data(new_bias);

        this->replace_module("fc", this->linear);
        ULOGGER_DEBUG("New model neurons:%d", this->linear->options.out_features());
    }

}