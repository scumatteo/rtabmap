#pragma once

#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/FeatureExtractor.h"
#include "rtabmap/core/region/models/Classifier.h"

namespace region
{

    struct ModelImpl : torch::nn::Cloneable<ModelImpl>
    {
        FeatureExtractor feature_extractor;
        Classifier classifier;

        ModelImpl(const FeatureExtractor &feature_extractor,
                  const Classifier &classifier);

        inline bool is_trained() const { return this->classifier->output_layer->linear->options.out_features() > 0; } // TODO

        torch::Tensor forward(const torch::Tensor &input);

        void adapt(const torch::Tensor &classes_in_this_experience);

        void reset() override;

    private:
        void rebuild_all_();
        void register_all_();
    };

    TORCH_MODULE(Model);

}

#endif