#include "rtabmap/core/region/models/TrainablePart.h"

namespace region
{
    TrainablePartImpl::TrainablePartImpl() : layer4(make_layer(512, 64, 1, 2)),
                                             avgpool(torch::nn::AdaptiveAvgPool2dOptions({1, 1}))

    {
        this->register_all_();
    }

    torch::Tensor TrainablePartImpl::forward(const torch::Tensor &input)
    {
        return torch::zeros({1}); //->model_.forward(input).toTensor();
    }

    void TrainablePartImpl::reset()
    {
        this->rebuild_all_();
        this->register_all_();
    }

    void TrainablePartImpl::rebuild_all_()
    {
        layer4 = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(layer4->clone());
        avgpool = std::dynamic_pointer_cast<torch::nn::AdaptiveAvgPool2dImpl>(avgpool->clone());
    }

    void TrainablePartImpl::register_all_()
    {
        register_module("layer4", layer4);
        register_module("avgpool", avgpool);
    }

}