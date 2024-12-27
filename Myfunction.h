#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

const int FRAME_SKIP = 10; // 跳帧数量
const double THRESHOLD = 20.0; // 画面差异阈值

const string PROVINCE_DIR = "province/";  // 省份汉字库路径
const string NUM_DIR = "num/";            // 数字字母库路径

// 提取视频中的关键帧，并根据阈值提取关键帧存储在 keyFrames 中
std::vector<cv::Mat> extractKeyFrames(const std::string& videoPath);

// 预处理
cv::Mat preprocessImage(const cv::Mat& frame);

// 判断轮廓是否符合规定
bool verifySizes(const RotatedRect& rotatedRect);

// 找轮廓
void findAndDrawContours(const Mat& preprocessedImage, Mat& src, vector<Mat>& Firsttarget);

// 调整车牌图像
void tortuosity(const Mat& src, vector<RotatedRect>& rects, vector<Mat>& dst_plates);

// 计算安全矩形
void safeRect(const Mat& src, const RotatedRect& rotatedRect, Rect2f& safe_rect);

// 旋转矩形
void rotation(const Mat& src, Mat& dst, const Size& rect_size, const Point2f& center, double angle);

// 去柳丁
bool clearRivet(Mat& plate);

// 规范字符轮廓
bool verifyCharSize(Mat src);

// 寻找城市字符
int getCityIndex(vector<Rect> rects);

// 获取汉字字符矩形
void getChineseRect(Rect cityRect, Rect& chineseRect);

// 字符分割
void character_segmentation(Mat plate, vector<Mat>& plateCharMats);

// 字符匹配
string character_matching(vector<Mat>& plateCharMats);
