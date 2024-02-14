#include "rtabmap/core/region/models/FreezedPart.h"

namespace rtabmap
{
    FreezedPartImpl::FreezedPartImpl() : conv1(conv_options(3, 64, 7, 2, 3)),
                                         bn1(64),
                                         relu(torch::nn::ReLUOptions().inplace(true)),
                                         maxpool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
                                         layer1(make_layer(64, 64, 1)),
                                         layer2(make_layer(128, 64, 1, 2)),
                                         layer3(make_layer(256, 64, 1, 2))

    {
        this->register_all_();

        // std::cout << "CONSTRUCTOR\n";
        // for (const auto &p : this->named_parameters())
        // {
        //     std::cout << p.key() << "\n";
        //     p.value().requires_grad_(false);
        // }
    }

    torch::Tensor FreezedPartImpl::forward(const torch::Tensor &input)
    {
        torch::Tensor x = this->conv1->forward(input);
        x = this->bn1->forward(x);
        x = this->relu->forward(x);
        x = this->maxpool->forward(x);
        x = this->layer1->forward(x);
        x = this->layer2->forward(x);
        x = this->layer3->forward(x);
        return x;

    }

    void FreezedPartImpl::reset()
    {
        // this->rebuild_all_();
        this->register_all_();

        // std::cout << "RESET\n";

        // for (const auto &p : this->named_parameters())
        // {
        //     std::cout << p.key() << "\n";
        //     // p.value().set_requires_grad(false);
        //     p.value().requires_grad_(false);
        // }
    }

    void FreezedPartImpl::rebuild_all_()
    {
        conv1 = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(conv1->clone());
        bn1 = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(bn1->clone());
        relu = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(relu->clone());
        maxpool = std::dynamic_pointer_cast<torch::nn::MaxPool2dImpl>(maxpool->clone());
        layer1 = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(layer1->clone());
        layer2 = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(layer2->clone());
        layer3 = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(layer3->clone());
    }

    void FreezedPartImpl::register_all_()
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
    }

}