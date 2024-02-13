#pragma once

#ifndef TRAINABLE_PART_H
#define TRAINABLE_PART_H

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/BasicBlock.h"

namespace region
{
    struct TrainablePartImpl : torch::nn::Cloneable<TrainablePartImpl>
    {
        int64_t inplanes = 64;

        torch::nn::Sequential layer4;
        torch::nn::AdaptiveAvgPool2d avgpool;
        
        TrainablePartImpl();

        torch::Tensor forward(const torch::Tensor &input);
        void reset() override;

    private:
        void rebuild_all_();
        void register_all_();
    };

    TORCH_MODULE(TrainablePart);

}
#endif