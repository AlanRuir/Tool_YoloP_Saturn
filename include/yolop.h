#ifndef __YOLOP_H__
#define __YOLOP_H__

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

struct DetectResult
{
    cv::Rect box;      // 边界框
    int      class_id; // 类别索引
    float    score;    // 置信度得分
};

struct YoloPResult
{
    std::vector<DetectResult> detections;    // 目标检测结果
    cv::Mat                   drivable_area; // 驾驶区域分割掩码
    cv::Mat                   lane_lines;    // 车道线分割掩码
};

class YoloP
{
public:
    YoloP();
    ~YoloP();
    void init(const std::string& engine_path, float conf_threshold, float nms_threshold);
    void detect(cv::Mat& frame, YoloPResult& result);

private:
    cudaStream_t                                 stream_;     // CUDA 流
    nvinfer1::ICudaEngine*                       engine_;     // TensorRT 引擎
    std::unique_ptr<nvinfer1::IExecutionContext> context_;    // 执行上下文
    void*                                        buffers_[4]; // 输入和三个输出缓冲区
    std::vector<float>                           det_output_; // 检测输出
    std::vector<float>                           da_output_;  // 驾驶区域输出
    std::vector<float>                           ll_output_;  // 车道线输出
    int                                          input_height_;
    int                                          input_width_;
    int                                          det_output_size_;
    int                                          da_output_size_;
    int                                          ll_output_size_;
    float                                        conf_threshold_;
    float                                        nms_threshold_;

    void preprocess(cv::Mat& frame, float* input_buffer);
    void postprocess_detection(const std::vector<float>& det_output, std::vector<DetectResult>& detections, float x_factor, float y_factor);
    void postprocess_segmentation(const std::vector<float>& output, cv::Mat& mask, int height, int width, int output_height, int output_width);
};

#endif // __YOLOP_H__