#include "rtabmap/core/region/models/Model.h"
#include "rtabmap/utilite/ULogger.h"

namespace rtabmap
{
    ModelImpl::ModelImpl() : feature_extractor(nullptr),
                             classifier(nullptr)
    {
    }

    ModelImpl::ModelImpl(const FeatureExtractor &feature_extractor,
                         const IncrementalLinear &classifier) : feature_extractor(feature_extractor),
                                                                classifier(classifier)
    {
        for (const auto &p : this->feature_extractor->named_parameters())
        {
            std::cout << p.key() << "\n";
        }

        this->register_all_();
    }

    ModelImpl::ModelImpl(const FeatureExtractor &feature_extractor,
                         const IncrementalLinear &classifier,
                         const std::string &model_path) : feature_extractor(feature_extractor),
                                                          classifier(classifier)
    {

        this->register_all_();

        if (!model_path.empty())
        {
            try
            {
                std::ifstream file(model_path, std::ios::binary);
                std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                file.close();
                torch::IValue ivalue = torch::pickle_load(data);
                auto dict = ivalue.toGenericDict();

                for (const auto &p : this->named_buffers())
                {
                    // std::cout << p.key() << "\n";
                    if (dict.contains(p.key()))
                    {
                        p.value().set_data(dict.at(p.key()).toTensor());
                    }
                }

                for (const auto &p : this->named_parameters())
                {
                    // std::cout << p.key() << "\n";
                    if (dict.contains(p.key()))
                    {
                        p.value().set_data(dict.at(p.key()).toTensor());
                    }
                }

                // this->feature_extractor->freezed_part->eval();

                // for (const auto &p : this->feature_extractor->trainable_part->named_parameters())
                // {
                //     std::cout << p.key() << "\n";
                //     if(dict.contains(p.key())){
                //         p.value().set_data(dict.at(p.key()).toTensor());
                //     }                }

                // for (const auto &p : this->classifier->linear->named_parameters())
                // {
                //     std::cout << p.key() << "\n";
                //     if(dict.contains(p.key())){
                //         p.value().set_data(dict.at(p.key()).toTensor());
                //     }

                // }
            }

            catch (const c10::Error &e)
            {
                ULOGGER_DEBUG("Error loading the state_dict: %s\n", e.what());
                std::cerr << e.what() << "\n";
                std::cerr << "Error loading the state_dict\n";
            }
        }
        // this->register_all_();
    }

    void ModelImpl::set_freezed_part()
    {
        for (const auto &p : this->feature_extractor->freezed_part->named_parameters())
        {
            std::cout << p.key() << "\n";
            p.value().requires_grad_(false);
        }
        this->feature_extractor->freezed_part->eval();
    }

    // ModelImpl::ModelImpl(const std::string &model_path,
    //                      size_t initial_out_features) : feature_extractor(FeatureExtractor(model_path)),

    // {
    //     this->register_all_();
    // }

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
        // this->rebuild_all_();
        this->register_all_();
    }

    void ModelImpl::rebuild_all_()
    {
        this->feature_extractor = std::dynamic_pointer_cast<FeatureExtractorImpl>(this->feature_extractor->clone());
        this->classifier = std::dynamic_pointer_cast<IncrementalLinearImpl>(this->classifier->clone());
    }

    void ModelImpl::register_all_()
    {
        register_module("feature_extractor", this->feature_extractor);
        register_module("classifier", this->classifier);
    }

}