#pragma once

#ifndef INCREMENTAL_LINEAR_H
#define INCREMENTAL_LINEAR_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <torch/torch.h>
#include <torch/script.h>

namespace rtabmap
{
    // Load jit script for feature extractor
    struct RTABMAP_CORE_EXPORT IncrementalLinearImpl : torch::nn::Cloneable<IncrementalLinearImpl>
    {

        // std::string model_path;
        torch::nn::Linear linear;

        IncrementalLinearImpl(size_t in_features,
                              size_t initial_out_features);

        IncrementalLinearImpl(const std::string &model_path, 
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