#include "rtabmap/core/region/utils.h"

namespace region
{

    torch::Tensor image_to_tensor(const cv::Mat &image, const int64_t &image_width, const int64_t &image_height)
    {
        cv::Mat img;
        cv::cvtColor(image, img, cv::COLOR_BGR2RGB); //TODO check
        cv::resize(img, img, cv::Size(image_width, image_height));
        img.convertTo(img, CV_32FC3, 1 / 255.0);
        torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, c10::kFloat);
        tensor = tensor.permute({0, 3, 1, 2});
        return tensor.clone();
    }

    torch::Tensor compute_simple_weights(const torch::Tensor &samples_per_class)
    {
        return 1.0 / samples_per_class;
    }

    torch::Tensor compute_effective_weights(const torch::Tensor &samples_per_class, float beta)
    {
        // torch::Tensor samples_per_class = std::get<2>(at::_unique2(labels, true, false, true));
        torch::Tensor effective_sum = 1.0 - torch::pow(beta, samples_per_class);
        torch::Tensor weights = (1.0 - beta) / effective_sum;
        return weights / torch::sum(weights) * samples_per_class.size(0);
    }
}