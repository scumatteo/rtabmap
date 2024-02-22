#pragma once

#ifndef TRAINABLE_PART_H
#define TRAINABLE_PART_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/BasicBlock.h"

namespace rtabmap
{
    struct RTABMAP_CORE_EXPORT TrainablePartImpl : torch::nn::Module
    {
        int64_t inplanes = 64;

        torch::nn::Sequential layer4;
        torch::nn::AdaptiveAvgPool2d avgpool;
        
        TrainablePartImpl();

        torch::Tensor forward(const torch::Tensor &input);
    };

    TORCH_MODULE(TrainablePart);

}
#endif