#pragma once

#ifndef TRAIN_THREAD_H
#define TRAIN_THREAD_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
#include "rtabmap/core/Memory.h"

#include "rtabmap/core/Parameters.h"
#include "rtabmap/core/region/models/Model.h"
#include "rtabmap/core/region/utils.h"
#include "rtabmap/core/region/storage_policy/ClassBalancedBuffer.h"
#include "rtabmap/core/region/losses/CustomLoss.h"

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

namespace rtabmap
{
    class Memory;

    class RTABMAP_CORE_EXPORT TrainThread
    {
    public:
        TrainThread(Memory* memory,
                    const Model &model,
                    const ParametersMap &parameters,
                    int64_t target_width,
                    int64_t target_height,
                    const std::string &checkpointPath,
                    torch::DeviceType device = torch::kCPU);

        Model model(); 
        bool is_training();
        bool last_training_end();
        // const Model on_training_end();

        void run(const std::unordered_map<int, std::pair<cv::Mat, int>> &experience, const std::unordered_map<int, std::pair<int, int>> &signatures_moved);
        void train(std::unordered_map<int, std::pair<cv::Mat, int>> experience, std::unordered_map<int, std::pair<int, int>> signatures_moved);

    private:

        void parseParameters(const ParametersMap &parameters);

        void make_optimizer();
        void make_loss_fn();
        void make_replay_memory();
        torch::Tensor compute_weights(const torch::Tensor &samples_per_class);

        float loop(const auto &dataloader,
                   // TODo accuracy,
                   size_t num_classes,
                   bool train = true);

        boost::mutex _mutex;
        Model _model;
        bool _is_training;
        bool _training_end;
        std::string _checkpointPath;
        // double _learning_rate;
        // int _weighting_method;
        // int _loss_type;
        // torch::nn::CrossEntropyLoss _loss_fn;
        // int _optimizer_type;
        // std::shared_ptr<torch::optim::Optimizer> _optimizer;
        // std::vector<size_t> ids_;
        // std::vector<cv::Mat> images_;
        // std::vector<size_t> labels_;
        // ReservoirSamplingBuffer _replay_memory;
        // float _beta;
        // size_t _feature_batch_size;
        // size_t _experience_batch_size;
        // size_t _replay_memory_batch_size;
        // size_t _epochs;
        // int64_t _image_width;
        // int64_t _image_height;
        // torch::Device _device;

        double _learning_rate;
        unsigned int _epochs;
        int _optimizer_type;
        std::shared_ptr<torch::optim::Optimizer> _optimizer;
        int _weighting_method;
        double _beta;
        float _gamma;
        int _loss_type;
        std::shared_ptr<CustomLossImpl> _loss_fn;
        std::shared_ptr<Buffer> _replay_memory;
    
        int64_t _target_width;
        int64_t _target_height;
        unsigned int _batch_size;
        unsigned int _feature_batch_size;

        unsigned int _replay_memory_type;
        unsigned int _replay_memory_size;
        unsigned int _replay_memory_batch_size;
        torch::DeviceType _device;

        Memory *_memory;
    };

}

#endif