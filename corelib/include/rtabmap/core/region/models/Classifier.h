#pragma once

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/IncrementalLinear.h"

namespace rtabmap
{
    // Load jit script for feature extractor
    struct ClassifierImpl : torch::nn::Cloneable<ClassifierImpl>
    {
        torch::nn::Sequential mlp;
        IncrementalLinear output_layer;

        ClassifierImpl(const std::vector<size_t> &layer_dims,
                       size_t out_features);

        /* ClassifierImpl(const torch::nn::Sequential &mlp,
                       const IncrementalLinear &output_layer); */

        torch::Tensor forward(const torch::Tensor &input);
        void adapt(const torch::Tensor &classes_in_this_experience);

        void reset() override;

        /*         std::shared_ptr<ClassifierImpl> clone();
         */
    private:
        void rebuild_all_();
        void register_all_();
    };
    TORCH_MODULE(Classifier);
}

#endif