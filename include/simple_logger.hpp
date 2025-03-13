#ifndef __COMMON_SIMPLE_LOGGER__
#define __COMMON_SIMPLE_LOGGER__

#include <NvInfer.h>
#include <iostream>

class SimpleLogger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "[ERROR] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "[WARNING] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "[INFO] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "[FATAL] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kVERBOSE:
            std::cerr << "[VERBOSE] " << msg << std::endl;
            break;
        default:
            break;
        }
    }
};

static SimpleLogger global_logger;

#endif // __COMMON_SIMPLE_LOGGER__