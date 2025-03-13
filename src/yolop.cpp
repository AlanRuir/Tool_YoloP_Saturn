#include "yolop.h"
#include "simple_logger.hpp"
#include <iostream>

YoloP::YoloP()
    : stream_(nullptr)
    , engine_(nullptr)
{
    buffers_[0] = nullptr; // 输入
    buffers_[1] = nullptr; // 检测输出
    buffers_[2] = nullptr; // 驾驶区域输出
    buffers_[3] = nullptr; // 车道线输出
}

YoloP::~YoloP()
{
    if (engine_)
    {
        delete engine_;
    }
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    for (int i = 0; i < 4; ++i)
    {
        if (buffers_[i])
        {
            cudaFree(buffers_[i]);
        }
    }
    std::cout << "YoloP destructor" << std::endl;
}

void YoloP::init(const std::string& engine_path, float conf_threshold, float nms_threshold)
{
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
        exit(-1);
    }
    int size = file.tellg();
    file.seekg(0, file.beg);
    char* trt_stream = new char[size];
    file.read(trt_stream, size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(global_logger));
    engine_  = runtime->deserializeCudaEngine(trt_stream, size);
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    delete[] trt_stream;

    // 打印张量信息以调试
    std::cout << "Engine Tensor Information:" << std::endl;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char*    name    = engine_->getIOTensorName(i);
        nvinfer1::Dims dims    = engine_->getTensorShape(name);
        std::string    io_type = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output";
        std::cout << io_type << " Tensor: " << name << ", Shape: [";
        for (int d = 0; d < dims.nbDims; ++d)
        {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    // 获取输入维度
    auto input_dims = engine_->getTensorShape("images");
    if (input_dims.nbDims == -1)
    {
        std::cerr << "无法找到输入张量 'images'" << std::endl;
        exit(-1);
    }
    input_height_ = input_dims.d[2];
    input_width_  = input_dims.d[3];
    std::cout << "Input height: " << input_height_ << ", width: " << input_width_ << std::endl;
    context_->setInputShape("images", nvinfer1::Dims4{1, 3, input_height_, input_width_});

    // 获取输出维度
    auto det_dims    = engine_->getTensorShape("det_out");
    auto seg_dims    = engine_->getTensorShape("drive_area_seg");
    auto lane_dims   = engine_->getTensorShape("lane_line_seg");
    det_output_size_ = 1;
    da_output_size_  = 1;
    ll_output_size_  = 1;
    for (int d = 0; d < det_dims.nbDims; ++d)
        det_output_size_ *= det_dims.d[d];
    for (int d = 0; d < seg_dims.nbDims; ++d)
        da_output_size_ *= seg_dims.d[d];
    for (int d = 0; d < lane_dims.nbDims; ++d)
        ll_output_size_ *= lane_dims.d[d];

    // 分配 CUDA 内存
    cudaError_t err;
    err = cudaMalloc(&buffers_[0], input_height_ * input_width_ * 3 * sizeof(float));
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc input failed: " << cudaGetErrorString(err) << std::endl;
    err = cudaMalloc(&buffers_[1], det_output_size_ * sizeof(float));
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc det_out failed: " << cudaGetErrorString(err) << std::endl;
    err = cudaMalloc(&buffers_[2], da_output_size_ * sizeof(float));
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc drive_area_seg failed: " << cudaGetErrorString(err) << std::endl;
    err = cudaMalloc(&buffers_[3], ll_output_size_ * sizeof(float));
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc lane_line_seg failed: " << cudaGetErrorString(err) << std::endl;

    det_output_.resize(det_output_size_);
    da_output_.resize(da_output_size_);
    ll_output_.resize(ll_output_size_);

    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;

    conf_threshold_ = conf_threshold;
    nms_threshold_  = nms_threshold;
}

void YoloP::preprocess(cv::Mat& frame, float* input_buffer)
{
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), true, false);
    cudaMemcpyAsync(input_buffer, blob.ptr<float>(), input_height_ * input_width_ * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
}

void YoloP::postprocess_detection(const std::vector<float>& det_output, std::vector<DetectResult>& detections, float x_factor, float y_factor)
{
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;

    // det_out 形状为 [1, 25200, 6]，假设为 [x, y, w, h, conf, class_id]
    int     num_boxes = det_output_size_ / 6;
    cv::Mat det_mat(num_boxes, 6, CV_32F, (void*)det_output.data());

    for (int i = 0; i < det_mat.rows; ++i)
    {
        float conf = det_mat.at<float>(i, 4); // 置信度
        if (conf > conf_threshold_)
        {
            float cx       = det_mat.at<float>(i, 0);                   // 中心 x
            float cy       = det_mat.at<float>(i, 1);                   // 中心 y
            float w        = det_mat.at<float>(i, 2);                   // 宽度
            float h        = det_mat.at<float>(i, 3);                   // 高度
            int   class_id = static_cast<int>(det_mat.at<float>(i, 5)); // class_id

            int x      = static_cast<int>((cx - w / 2) * x_factor);
            int y      = static_cast<int>((cy - h / 2) * y_factor);
            int width  = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(x, y, width, height));
            scores.push_back(conf);
            class_ids.push_back(class_id); // class_id
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);
    for (int idx : indices)
    {
        DetectResult dr;
        dr.box      = boxes[idx];
        dr.score    = scores[idx];
        dr.class_id = class_ids[idx];
        detections.push_back(dr);
    }
}

void YoloP::postprocess_segmentation(const std::vector<float>& output, cv::Mat& mask, int height, int width, int output_height, int output_width)
{
    // 输出为 [1, 2, 640, 640]，两通道（背景/前景概率）
    cv::Mat raw_mask(2, output_height * output_width, CV_32F, (void*)output.data());
    cv::Mat prob_map(output_height, output_width, CV_32F);
    for (int i = 0; i < output_height * output_width; ++i)
    {
        float bg                                               = raw_mask.at<float>(0, i); // 背景概率
        float fg                                               = raw_mask.at<float>(1, i); // 前景概率
        prob_map.at<float>(i / output_width, i % output_width) = fg > bg ? 1.0 : 0.0;
    }
    // 调整掩码大小以匹配输入帧
    cv::resize(prob_map, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    mask.convertTo(mask, CV_8UC1, 255); // 转换为 0-255 的二值掩码
}

void YoloP::detect(cv::Mat& frame, YoloPResult& result)
{
    int64_t start = cv::getTickCount();

    float x_factor = frame.cols / static_cast<float>(input_width_);
    float y_factor = frame.rows / static_cast<float>(input_height_);

    preprocess(frame, static_cast<float*>(buffers_[0]));

    context_->setTensorAddress("images", buffers_[0]);
    context_->setTensorAddress("det_out", buffers_[1]);
    context_->setTensorAddress("drive_area_seg", buffers_[2]);
    context_->setTensorAddress("lane_line_seg", buffers_[3]);

    if (!context_->executeV2(buffers_))
    {
        std::cerr << "TensorRT inference failed" << std::endl;
        exit(-1);
    }

    cudaMemcpyAsync(det_output_.data(), buffers_[1], det_output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(da_output_.data(), buffers_[2], da_output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(ll_output_.data(), buffers_[3], ll_output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 处理检测结果
    postprocess_detection(det_output_, result.detections, x_factor, y_factor);

    // 处理分割结果，传递原始帧的 height 和 width
    int seg_height = 640; // 输出分辨率
    int seg_width  = 640;
    postprocess_segmentation(da_output_, result.drivable_area, frame.rows, frame.cols, seg_height, seg_width);
    postprocess_segmentation(ll_output_, result.lane_lines, frame.rows, frame.cols, seg_height, seg_width);

    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
}