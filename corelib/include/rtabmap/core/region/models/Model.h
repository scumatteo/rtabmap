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

    struct RTABMAP_CORE_EXPORT ModelImpl : torch::nn::Module
    {

        std::string model_path;

        FeatureExtractor feature_extractor;
        IncrementalLinear classifier;

        ModelImpl();

        ModelImpl(const FeatureExtractor &feature_extractor,
                  const IncrementalLinear &classifier,
                  const std::string &model_path = "");

        torch::Tensor forward(const torch::Tensor &input);
        void adapt(const torch::Tensor &classes_in_this_experience);
        void set_freezed_part();
        inline bool is_trained() const { std::cout << "out " << this->classifier->linear->options.out_features(); return this->classifier->linear->options.out_features() > 0; }

        std::shared_ptr<ModelImpl> clone();
        void save_state_dict(const std::string &model_path);
        void load_state_dict(const std::string &model_path);
    };

    TORCH_MODULE(Model);

}

#endif