#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

namespace lsd_slam
{

void downscaleImageHalf(const float *in, float *out, size_t w, size_t h);

bool imwriteFloat(const std::string &filename, const cv::Mat &imToSave);
cv::Mat imreadFloat(const std::string &filename);

}
