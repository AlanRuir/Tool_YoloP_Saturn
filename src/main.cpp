#include <iostream>
#include <vector>
#include "yolop.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "用法: " << argv[0] << " <视频文件路径>" << std::endl;
        return -1;
    }

    std::shared_ptr<YoloP> detector = std::make_shared<YoloP>();
    detector->init("../models/yolop.engine", 0.25f, 0.45f);

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频文件: " << argv[1] << std::endl;
        return -1;
    }

    int             frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int             frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double          fps          = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("output_yolop.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat     frame;
    YoloPResult result;
    int         frame_count = 0;

    while (cap.read(frame))
    {
        frame_count++;
        std::cout << "处理第 " << frame_count << " 帧\r" << std::flush;

        detector->detect(frame, result);

        // 可视化检测结果
        for (const auto& dr : result.detections)
        {
            cv::rectangle(frame, dr.box, cv::Scalar(0, 255, 0), 2);
        }

        // 可视化驾驶区域（绿色半透明）
        cv::Mat da_colored(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        da_colored.setTo(cv::Scalar(0, 255, 0), result.drivable_area);
        cv::addWeighted(frame, 0.8, da_colored, 0.5, 0.0, frame);

        // 可视化车道线（红色）
        cv::Mat ll_colored(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        ll_colored.setTo(cv::Scalar(0, 0, 255), result.lane_lines);
        cv::addWeighted(frame, 0.8, ll_colored, 0.5, 0.0, frame);

        cv::imshow("YOLOP Result", frame);
        writer.write(frame);

        if (cv::waitKey(1) == 'q')
            break;

        result.detections.clear();
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "\n处理完成，结果保存至 output_yolop.mp4" << std::endl;
    return 0;
}