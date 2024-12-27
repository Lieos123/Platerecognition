#pragma once
#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>

using namespace cv;
using namespace std;

// 1. elbp函数：圆形算子LBP特征提取
// src: 输入的图像（单通道灰度图像）
// dst: 输出的LBP特征图像（单通道灰度图像）
// radius: LBP算子的半径
// neighbors: LBP算子的邻域像素数量（通常为8个邻域）
void elbp(Mat& src, Mat& dst, int radius, int neighbors);

// 2. getUniformPatternLBPFeature函数：计算均匀模式（Uniform Pattern）的LBP特征
// src: 输入的图像（单通道灰度图像）
// dst: 输出的均匀模式LBP特征图像（单通道灰度图像）
// radius: LBP算子的半径
// neighbors: LBP算子的邻域像素数量
void getUniformPatternLBPFeature(Mat src, Mat dst, int radius, int neighbors);

// 3. getHopTimes函数：计算数字n的二进制表达中1的跳跃次数
// n: 需要计算的整数
// 返回值：跳跃次数，即数字n的二进制表示中从0到1或从1到0的切换次数
int getHopTimes(int n);

// 4. getLBPH函数：计算LBP特征的直方图（LBPH）
// src: 输入的图像（单通道灰度图像）
// numPatterns: LBP算子模式的数量，通常是2^neighbors
// grid_x: 将图像按列划分为grid_x块
// grid_y: 将图像按行划分为grid_y块
// normed: 是否对直方图进行归一化处理（true表示归一化，false表示不归一化）
// 返回值：包含LBP特征的直方图（每个block的LBP直方图合并为一个大向量）
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed);

// 5. getLocalRegionLBPH函数：计算图像局部区域的LBPH直方图
// src: 输入的图像块（单通道灰度图像）
// minValue: 直方图的最小值
// maxValue: 直方图的最大值
// normed: 是否对直方图进行归一化处理（true表示归一化，false表示不归一化）
// 返回值：该局部区域的LBPH直方图
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed);


void extractLBP(Mat& img, Mat& lbpImg);

void extractLBP2(Mat& img, Mat& lbpImg);

void extractLBPs(const vector<Mat>& images, vector<Mat>& lbpImages);


