#pragma once

#ifndef CUSTOM_LOSS_H
#define CUSTOM_LOSS_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
#include <torch/torch.h>

namespace rtabmap
{

    struct RTABMAP_CORE_EXPORT CustomLossImpl : public torch::nn::CrossEntropyLossImpl
    {
        CustomLossImpl(torch::nn::CrossEntropyLossOptions options = {}) : torch::nn::CrossEntropyLossImpl(options) {}

        virtual torch::Tensor compute(const torch::Tensor &input, const torch::Tensor &target) = 0;
    };

    TORCH_MODULE(CustomLoss);

}
#endif