#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

const int FRAME_SKIP = 10; // ��֡����
const double THRESHOLD = 20.0; // ���������ֵ

const string PROVINCE_DIR = "province/";  // ʡ�ݺ��ֿ�·��
const string NUM_DIR = "num/";            // ������ĸ��·��

// ��ȡ��Ƶ�еĹؼ�֡����������ֵ��ȡ�ؼ�֡�洢�� keyFrames ��
std::vector<cv::Mat> extractKeyFrames(const std::string& videoPath);

// Ԥ����
cv::Mat preprocessImage(const cv::Mat& frame);

// �ж������Ƿ���Ϲ涨
bool verifySizes(const RotatedRect& rotatedRect);

// ������
void findAndDrawContours(const Mat& preprocessedImage, Mat& src, vector<Mat>& Firsttarget);

// ��������ͼ��
void tortuosity(const Mat& src, vector<RotatedRect>& rects, vector<Mat>& dst_plates);

// ���㰲ȫ����
void safeRect(const Mat& src, const RotatedRect& rotatedRect, Rect2f& safe_rect);

// ��ת����
void rotation(const Mat& src, Mat& dst, const Size& rect_size, const Point2f& center, double angle);

// ȥ����
bool clearRivet(Mat& plate);

// �淶�ַ�����
bool verifyCharSize(Mat src);

// Ѱ�ҳ����ַ�
int getCityIndex(vector<Rect> rects);

// ��ȡ�����ַ�����
void getChineseRect(Rect cityRect, Rect& chineseRect);

// �ַ��ָ�
void character_segmentation(Mat plate, vector<Mat>& plateCharMats);

// �ַ�ƥ��
string character_matching(vector<Mat>& plateCharMats);
