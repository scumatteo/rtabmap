#pragma once

#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
#include <torch/torch.h>
#include "rtabmap/core/region/losses/CustomLoss.h"

namespace rtabmap
{

    struct RTABMAP_CORE_EXPORT CrossEntropyLossImpl : public CustomLossImpl
    {
        CrossEntropyLossImpl(torch::nn::CrossEntropyLossOptions options = {}) : CustomLossImpl(options) {}

        virtual torch::Tensor compute(const torch::Tensor &input, const torch::Tensor &target);
    };

    TORCH_MODULE(CrossEntropyLoss);

}
#endif