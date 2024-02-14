#pragma once

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/FreezedPart.h"
#include "rtabmap/core/region/models/TrainablePart.h"

namespace rtabmap
{
    struct RTABMAP_CORE_EXPORT FeatureExtractorImpl : torch::nn::Cloneable<FeatureExtractorImpl>
    {

        FeatureExtractorImpl();

        // std::string model_path;

        FreezedPart freezed_part;
        TrainablePart trainable_part;

        // FeatureExtractorImpl(const std::string &model_path);

        torch::Tensor extract_freezed_features(const torch::Tensor &input);
        torch::Tensor extract_features(const torch::Tensor &input);

        // torch::Tensor forward(const torch::Tensor &input);
        void reset() override;

        void train(bool on = true) override;

    private:
        void rebuild_all_();
        void register_all_();
    };

    TORCH_MODULE(FeatureExtractor);
}

#endif