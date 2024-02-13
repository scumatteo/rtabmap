#include "rtabmap/core/region/models/Model.h"

namespace region
{
    ModelImpl::ModelImpl(const FeatureExtractor &feature_extractor,
                         const Classifier &classifier) : feature_extractor(feature_extractor),
                                                         classifier(classifier)

    {
        this->register_all_();
    }

    torch::Tensor ModelImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->feature_extractor->extract_freezed_features(input);
        x = this->feature_extractor->extract_features(x);
        x = this->classifier->forward(x); 
        return x;
    }

    void ModelImpl::adapt(const torch::Tensor &classes_in_this_experience)
    {
        this->classifier->adapt(classes_in_this_experience);
    }

    void ModelImpl::reset()
    {
        this->rebuild_all_();
        this->register_all_();
    }

    void ModelImpl::rebuild_all_()
    {
        this->feature_extractor = std::dynamic_pointer_cast<FeatureExtractorImpl>(this->feature_extractor->clone());
        this->classifier = std::dynamic_pointer_cast<ClassifierImpl>(this->classifier->clone());
    }

    void ModelImpl::register_all_()
    {
        register_module("feature_extractor", this->feature_extractor);
        register_module("classifier", this->classifier);
    }

}