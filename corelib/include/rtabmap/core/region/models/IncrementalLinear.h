#pragma once

#ifndef INCREMENTAL_LINEAR_H
#define INCREMENTAL_LINEAR_H

#include <torch/torch.h>
#include <torch/script.h>

namespace region
{
    // Load jit script for feature extractor
    struct IncrementalLinearImpl : torch::nn::Cloneable<IncrementalLinearImpl>
    {

        torch::nn::Linear linear;

        IncrementalLinearImpl(size_t in_features,
                              size_t initial_out_features);

        torch::Tensor forward(const torch::Tensor &input);

        void adapt(const torch::Tensor &classes_in_this_experience);

        void reset() override;

    private:
        void rebuild_all_();
        void register_all_();
    };
    TORCH_MODULE(IncrementalLinear);
}

#endif