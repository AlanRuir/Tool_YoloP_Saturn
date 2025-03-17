# Tool_YoloP_Saturn

# 介绍

本项目描述了Tool_YoloP_Saturn的功能、下载编译及运行效果。

YoloP是由华中科技大学视觉团队设计的一个用于自动驾驶感知的多任务网络，该网络可以联合处理自动驾驶中的目标检测、可驾驶区域分割和车道检测三个关键任务。

本项目在此模型基础上增加了基于TensorRT10.x的适配，将YoloP提供的标准onnx转换为TensorRT的engine，并且做了engine的读取、推理和后处理，使得此工具的使用者可以方便的将YoloP的功能集成在他自己的项目中。

# 项目的下载与编译

执行如下命令下载项目

```sh
git@github.com:AlanRuir/Tool_YoloP_Saturn.git
```

执行如下命令编译项目

```sh
mkdir build
cd build/
cmake ..
make
```

# 项目的运行

执行如下命令运行项目

```sh
./YoloP_Detector ../videos/input.mp4
```

![检测前](doc/images/input.gif)



![检测后](doc/images/output.gif)

进程会将../videos/input.mp4输入到模型中，并将检测结果绘制在图像上并以一个新的窗口显示，同时会将结果保存成一个mp4文件。