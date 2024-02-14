#pragma once

#ifndef MODEL_H
#define MODEL_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <torch/torch.h>
#include <torch/script.h>
#include "rtabmap/core/region/models/FeatureExtractor.h"
#include "rtabmap/core/region/models/IncrementalLinear.h"

namespace rtabmap
{

    struct RTABMAP_CORE_EXPORT ModelImpl : torch::nn::Cloneable<ModelImpl>
    {

        std::string model_path;

        FeatureExtractor feature_extractor;
        IncrementalLinear classifier;

        ModelImpl();

        ModelImpl(const FeatureExtractor &feature_extractor,
                  const IncrementalLinear &classifier);

        ModelImpl(const FeatureExtractor &feature_extractor,
                  const IncrementalLinear &classifier,
                  const std::string &model_path);

        // ModelImpl(const std::string &model_path,
        //           size_t initial_out_features);

        inline bool is_trained() const { return this->classifier->linear->options.out_features() > 0; } // TODO

        torch::Tensor forward(const torch::Tensor &input);

        void adapt(const torch::Tensor &classes_in_this_experience);

        void set_freezed_part();

        void reset() override;

    private:
        void rebuild_all_();
        void register_all_();
    };

    TORCH_MODULE(Model);

}

#endif