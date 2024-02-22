#pragma once

#ifndef INCREMENTAL_LINEAR_H
#define INCREMENTAL_LINEAR_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <torch/torch.h>
#include <torch/script.h>

namespace rtabmap
{
    // Load jit script for feature extractor
    struct RTABMAP_CORE_EXPORT IncrementalLinearImpl : torch::nn::Module
    {

        torch::nn::Linear linear;

        IncrementalLinearImpl(size_t in_features = 512,
                              size_t initial_out_features = 0);

        torch::Tensor forward(const torch::Tensor &input);

        void adapt(const torch::Tensor &classes_in_this_experience);
    };
    TORCH_MODULE(IncrementalLinear);
}

#endif