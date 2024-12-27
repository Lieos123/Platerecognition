#pragma once
#pragma once


#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>

using namespace cv;
using namespace std;

// 1. elbp������Բ������LBP������ȡ
// src: �����ͼ�񣨵�ͨ���Ҷ�ͼ��
// dst: �����LBP����ͼ�񣨵�ͨ���Ҷ�ͼ��
// radius: LBP���ӵİ뾶
// neighbors: LBP���ӵ���������������ͨ��Ϊ8������
void elbp(Mat& src, Mat& dst, int radius, int neighbors);

// 2. getUniformPatternLBPFeature�������������ģʽ��Uniform Pattern����LBP����
// src: �����ͼ�񣨵�ͨ���Ҷ�ͼ��
// dst: ����ľ���ģʽLBP����ͼ�񣨵�ͨ���Ҷ�ͼ��
// radius: LBP���ӵİ뾶
// neighbors: LBP���ӵ�������������
void getUniformPatternLBPFeature(Mat src, Mat dst, int radius, int neighbors);

// 3. getHopTimes��������������n�Ķ����Ʊ����1����Ծ����
// n: ��Ҫ���������
// ����ֵ����Ծ������������n�Ķ����Ʊ�ʾ�д�0��1���1��0���л�����
int getHopTimes(int n);

// 4. getLBPH����������LBP������ֱ��ͼ��LBPH��
// src: �����ͼ�񣨵�ͨ���Ҷ�ͼ��
// numPatterns: LBP����ģʽ��������ͨ����2^neighbors
// grid_x: ��ͼ���л���Ϊgrid_x��
// grid_y: ��ͼ���л���Ϊgrid_y��
// normed: �Ƿ��ֱ��ͼ���й�һ������true��ʾ��һ����false��ʾ����һ����
// ����ֵ������LBP������ֱ��ͼ��ÿ��block��LBPֱ��ͼ�ϲ�Ϊһ����������
Mat getLBPH(Mat src, int numPatterns, int grid_x, int grid_y, bool normed);

// 5. getLocalRegionLBPH����������ͼ��ֲ������LBPHֱ��ͼ
// src: �����ͼ��飨��ͨ���Ҷ�ͼ��
// minValue: ֱ��ͼ����Сֵ
// maxValue: ֱ��ͼ�����ֵ
// normed: �Ƿ��ֱ��ͼ���й�һ������true��ʾ��һ����false��ʾ����һ����
// ����ֵ���þֲ������LBPHֱ��ͼ
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed);


void extractLBP(Mat& img, Mat& lbpImg);

void extractLBP2(Mat& img, Mat& lbpImg);

void extractLBPs(const vector<Mat>& images, vector<Mat>& lbpImages);


