#pragma once

#ifndef BASE_BUFFER_H
#define BASE_BUFFER_H

#include "libtorch_porting/region_learner/datasets/latent_dataset.hpp"

namespace region_learner
{

    class BaseBuffer
    {
    public:
        BaseBuffer(size_t max_size);

        inline const std::shared_ptr<LatentDataset> &buffer() const { return this->buffer_; }
        void buffer(const std::shared_ptr<LatentDataset> &new_buffer) { this->buffer_ = new_buffer; }

        virtual void update(const std::shared_ptr<LatentDataset> &new_data) = 0;
        virtual void resize(size_t new_size) = 0;


    protected:
        size_t max_size_;
        std::shared_ptr<LatentDataset> buffer_;
    };
}

#endif