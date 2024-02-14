#pragma once

#ifndef BASIC_BLOCK_H
#define BASIC_BLOCK_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
#include <torch/torch.h>

namespace rtabmap
{
    torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                          int64_t stride = 1, int64_t padding = 0, bool with_bias = false);

    struct RTABMAP_CORE_EXPORT BasicBlock : torch::nn::Cloneable<BasicBlock>
    {

        static const int expansion;

        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::ReLU relu;
        torch::nn::Conv2d conv2;
        torch::nn::BatchNorm2d bn2;
        torch::nn::Sequential downsample;

        BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
                   torch::nn::Sequential downsample = torch::nn::Sequential());

        torch::Tensor forward(torch::Tensor x);

        void reset() override;

    private:
        void rebuild_all_();
        void register_all_();
    };

    torch::nn::Sequential make_layer(int64_t planes, int64_t inplanes, int64_t blocks, int64_t stride = 1);

}

#endif