#pragma once

#ifndef LOSS_H
#define LOSS_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace region_learner
{

    struct FocalLossImpl : torch::nn::CrossEntropyLossImpl
    {
        torch::Scalar gamma;

        FocalLossImpl(const size_t &gamma,
                      const torch::Tensor &weights);
                      
        virtual torch::Tensor forward(const torch::Tensor &input,
                                      const torch::Tensor &target);
    };

    TORCH_MODULE(FocalLoss);

}
#endif