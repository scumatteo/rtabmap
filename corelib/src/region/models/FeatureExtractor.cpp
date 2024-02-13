#include "rtabmap/core/region/models/FeatureExtractor.h"

namespace region
{
    FeatureExtractorImpl::FeatureExtractorImpl(const std::string &model_path) : model_path(model_path)

    {
        try
        {
            std::ifstream file(model_path, std::ios::binary);
            std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            torch::IValue ivalue = torch::pickle_load(data);
            auto dict = ivalue.toGenericDict();
            
            for (const auto &p : this->freezed_part->named_parameters())
            {
                p.value().set_data(dict.at(p.key()).toTensor());
                p.value().requires_grad_(false);
            }
            this->freezed_part->eval();

            for (const auto &p : this->trainable_part->named_parameters())
            {
                p.value().set_data(dict.at(p.key()).toTensor());
            }

            this->register_all_();
        }
        catch (const c10::Error &e)
        {
            std::cerr << e.what() << "\n";
            std::cerr << "Error loading the state_dict\n";
        }
    }

    torch::Tensor FeatureExtractorImpl::extract_freezed_features(const torch::Tensor &input)
    {
        {
            torch::NoGradGuard no_grad;
            return this->freezed_part->forward(input);
        }
    }

    torch::Tensor FeatureExtractorImpl::extract_features(const torch::Tensor &input)
    {
        return this->trainable_part->forward(input);
    }

    void FeatureExtractorImpl::train(bool on)
    {
        torch::nn::Cloneable<FeatureExtractorImpl>::train(on);
        this->freezed_part->eval();
    }

    void FeatureExtractorImpl::reset()
    {
        this->rebuild_all_();
        this->register_all_();
    }

    void FeatureExtractorImpl::rebuild_all_()
    {
        freezed_part = std::dynamic_pointer_cast<FreezedPartImpl>(freezed_part->clone());
        trainable_part = std::dynamic_pointer_cast<TrainablePartImpl>(trainable_part->clone());
    }

    void FeatureExtractorImpl::register_all_()
    {

        register_module("freezed_part", freezed_part);
        register_module("trainable_part", trainable_part);
    }

}