#include "Myfunction.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include <chrono>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;


// 函数：从视频中提取关键帧
std::vector<cv::Mat> extractKeyFrames(const std::string& videoPath) {
    cv::VideoCapture video(videoPath);

    if (!video.isOpened()) {
        std::cerr << "错误：无法打开视频文件" << std::endl;
        return {};
    }

    std::vector<cv::Mat> keyFrames;
    cv::Mat prevKeyFrame;
    int frameCount = 0;

    while (true) {
        cv::Mat frame;
        video >> frame;

        if (frame.empty()) {
            break; // 读取到文件末尾
        }

        if (frameCount == 0) {
            // 第一帧为关键帧
            keyFrames.push_back(frame.clone());
            prevKeyFrame = frame.clone();
        }
        else if (frameCount % FRAME_SKIP == 0) {
            // 每隔 FRAME_SKIP 帧计算差异
            cv::Mat diff;
            cv::absdiff(prevKeyFrame, frame, diff); // 计算两帧之间的绝对差值
            cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY); // 转为灰度图像
            double diffSum = cv::sum(diff)[0] / (diff.rows * diff.cols); // 计算差异均值

            if (diffSum > THRESHOLD) {
                keyFrames.push_back(frame.clone());
                prevKeyFrame = frame.clone(); // 更新上一关键帧
            }
        }

        frameCount++;
    }

    return keyFrames;
}

cv::Mat sharpenImage(const cv::Mat& blurredImg) {
    // 创建一个简单的锐化卷积核
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);

    // 使用卷积核进行锐化
    cv::Mat sharpenedImg;
    cv::filter2D(blurredImg, sharpenedImg, -1, kernel);
    return sharpenedImg;
}

cv::Mat preprocessImage(const cv::Mat& frame) {
    // 如果输入图像为空，则返回空矩阵
    if (frame.empty()) {
        std::cerr << "空图像！" << std::endl;
        return cv::Mat();
    }

    // 1. 高斯模糊：减少噪声
    cv::Mat blurredImg;
    cv::GaussianBlur(frame, blurredImg, cv::Size(5, 5), 0);
    //cv::imshow("高斯模糊", blurredImg);  // 显示高斯模糊后的图像

    // 2. 锐化：增强细节
    cv::Mat sharpenedImg = sharpenImage(blurredImg);
    //cv::imshow("锐化图像", sharpenedImg);  // 显示锐化后的图像

    // 3. 灰度化：将图像转换为灰度图像
    cv::Mat grayImg;
    cv::cvtColor(sharpenedImg, grayImg, cv::COLOR_BGR2GRAY);
    //cv::imshow("灰度图像", grayImg);  // 显示灰度图像

    // 4. Sobel 导数运算：检测图像边缘
    cv::Mat gradX, gradY, grad;
    cv::Sobel(grayImg, gradX, CV_32F, 1, 0, 3);  // Sobel X方向
    cv::Sobel(grayImg, gradY, CV_32F, 0, 1, 3);  // Sobel Y方向
    cv::magnitude(gradX, gradY, grad);  // 计算梯度的幅值
    grad.convertTo(grad, CV_8U);  // 转换为8位图像
    //cv::imshow("Sobel导数运算", grad);  // 显示Sobel运算后的图像

    // 5. 二值化：将图像转为黑白图像
    cv::Mat binaryImg;
    double thresholdVal = 100;  // 设置固定阈值
    cv::threshold(grad, binaryImg, thresholdVal, 255, cv::THRESH_BINARY);
    //cv::imshow("二值化图像", binaryImg);  // 显示二值化后的图像

    // 6. 形态学闭操作：消除小区域，连接空隙
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));  // 获取矩形结构元素
    cv::Mat closedImg;
    cv::morphologyEx(binaryImg, closedImg, cv::MORPH_CLOSE, element);
    //cv::imshow("形态学闭操作", closedImg);  // 显示闭操作后的图像

    // 7. 腐蚀操作：对图像中的白色区域进行腐蚀，减少腐蚀强度
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));  
    cv::Mat erodedImg;
    int erosionIterations = 1;  // 设置较少的腐蚀迭代次数
    cv::erode(closedImg, erodedImg, element2, cv::Point(-1, -1), erosionIterations);  // 使用较少的迭代次数
    //cv::imshow("轻度腐蚀图像", erodedImg);  // 显示腐蚀后的图像

    cv::waitKey(0);  // 等待用户按键
    cv::destroyAllWindows();  // 关闭所有窗口

    return erodedImg;  // 返回腐蚀后的图像
}

// 判断轮廓尺寸是否符合规格
bool verifySizes(const RotatedRect& rotatedRect) {
    // 容错率
    float error = 0.3;

    // 理想宽高比（训练样本使用的车牌规格为 136x36，因此将其用作理想宽高比计算）
    float aspect = float(136) / float(36);

    // 利用容错率计算出最小宽高比与最大宽高比
    float aspectMin = (1 - error) * aspect;
    float aspectMax = (1 + error) * aspect;

    // 真实宽高比
    float realAspect = float(rotatedRect.size.width) / float(rotatedRect.size.height);
    if (realAspect < 1) {
        realAspect = float(rotatedRect.size.height) / float(rotatedRect.size.width);
    }

    // 真实面积
    float area = rotatedRect.size.width * rotatedRect.size.height;

    // 最小面积与最大面积，不符合的丢弃
    // 给个大概就行，可以随时调整，给大一点也没关系，这只是初步筛选
    int areaMin = 44 * aspect * 14;
    int areaMax = 440 * aspect * 140;

    // 判断面积与宽高比是否符合规格
    if ((area < areaMin || area > areaMax) || (realAspect < aspectMin || realAspect > aspectMax)) {
        return false;
    }

    return true; // 符合规格
}

//// 找轮廓并绘制外接矩形
//void findAndDrawContours(const Mat& preprocessedImage, Mat& src) {
//    // 确保图像不为空
//    if (preprocessedImage.empty()) {
//        cerr << "空图像！" << endl;
//        return;
//    }
//
//    // 存储轮廓
//    vector<vector<Point>> contours;
//
//    // 查找轮廓
//    findContours(preprocessedImage, // 输入二值图像（已经过预处理的关键帧）
//        contours,           // 存储轮廓的容器
//        RETR_EXTERNAL,      // 轮廓检索模式：只检测外部轮廓
//        CHAIN_APPROX_NONE); // 轮廓近似模式：不进行近似，保留所有轮廓点
//
//    // 遍历所有轮廓
//    RotatedRect rotatedRect;
//    vector<RotatedRect> vec_sobel_rects;  // 存储符合尺寸规格的轮廓
//    for (const vector<Point>& points : contours) {
//        // 获取每个轮廓的最小外接旋转矩形
//        rotatedRect = minAreaRect(points);
//
//        // 获取原始矩形的宽度和高度
//        float width = rotatedRect.size.width;
//        float height = rotatedRect.size.height;
//
//        // 增大矩形尺寸（按比例）
//        float scale = 1.3;  // 比例因子，可以根据需求调整
//        rotatedRect.size.width = width * scale;
//        rotatedRect.size.height = height * scale;
//
//        // 在原图上绘制该旋转矩形的外接矩形（矩形不进行旋转）
//        rectangle(src, rotatedRect.boundingRect(), Scalar(0, 0, 255), 2);
//    }
//
//    // 显示结果图像
//    //imshow("找轮廓", src);
//    
//    // 用绿色矩形画出符合尺寸规格的轮廓
//    for (const RotatedRect& rect : vec_sobel_rects) {
//        rectangle(src, rect.boundingRect(), Scalar(0, 255, 0));
//    }
//    imshow("尺寸判断", src);
//    
//    //waitKey(0);  // 等待用户按键
//    destroyAllWindows();  // 关闭所有窗口
//}

// 找轮廓并绘制外接矩形
void findAndDrawContours(const Mat& preprocessedImage, Mat& src, vector<Mat>& Firsttarget) {
    // 确保图像不为空
    if (preprocessedImage.empty()) {
        cerr << "空图像！" << endl;
        return;
    }

    // 存储轮廓
    vector<vector<Point>> contours;

    // 查找轮廓
    findContours(preprocessedImage, // 输入二值图像（已经过预处理的关键帧）
        contours,           // 存储轮廓的容器
        RETR_EXTERNAL,      // 轮廓检索模式：只检测外部轮廓
        CHAIN_APPROX_NONE); // 轮廓近似模式：不进行近似，保留所有轮廓点

    // 存储符合尺寸规格的矩形
    vector<RotatedRect> vec_sobel_rects;  // 存储符合尺寸规格的轮廓
    for (const vector<Point>& points : contours) {
        // 获取每个轮廓的最小外接旋转矩形
        RotatedRect rotatedRect = minAreaRect(points);

        // 获取原始矩形的宽度和高度
        float width = rotatedRect.size.width;
        float height = rotatedRect.size.height;

        // 增大矩形尺寸（按比例）
        float scale = 1.22;  // 比例因子，可以根据需求调整
        rotatedRect.size.width = width * scale;
        rotatedRect.size.height = height * scale;

        // 判断轮廓是否符合尺寸规格
        if (verifySizes(rotatedRect)) {
            vec_sobel_rects.push_back(rotatedRect);
        }

        // 在原图上绘制该旋转矩形的外接矩形（矩形不进行旋转）
        rectangle(src, rotatedRect.boundingRect(), Scalar(0, 0, 255), 2);
    }

    // 显示符合尺寸要求的轮廓
    for (const RotatedRect& rect : vec_sobel_rects) {
        //rectangle(src, rect.boundingRect(), Scalar(255, 0, 0), 2);
        //imshow("当前符合规格的矩形", src);
        //cout<< "当前符合规格的矩形宽：" << rect.size.width << endl;
        //cout<< "当前符合规格的矩形高：" << rect.size.height << endl;
        //cout<< "当前符合规格的矩形角度" << rect.angle << endl;
        rectangle(src, rect.boundingRect(), Scalar(0, 255, 0), 2);
    }
    
    //// 调用tortuosity进行矩形矫正
    vector<Mat> dst_plates;
    tortuosity(src, vec_sobel_rects, dst_plates);

    // 判断定位候选车牌是否为空
    if (dst_plates.empty()) {
        cout << "未找到符合要求的车牌候选图像！" << endl;
    }
    else {
        // 查看矫正后的候选车牌
        //for (Mat& m : dst_plates) {
        //    imshow("Sobel 定位候选车牌", m);
        //    // 通过输入任意按键查看每一个候选车牌
        //    waitKey();
        //}
    }

    //Firsttarget = dst_plates; // 将矫正后的候选车牌存储到 Firsttarget 中
    for (const Mat& plate : dst_plates) {
        // 将矫正后的候选车牌存储到 Firsttarget 中
        Firsttarget.push_back(plate);
    }


    //imshow("尺寸判断", src);
    destroyAllWindows();  // 关闭所有窗口
}


// 调整车牌图像
void tortuosity(const Mat& src, vector<RotatedRect>& rects, vector<Mat>& dst_plates) {
    // 遍历要处理的矩形
    for (const RotatedRect& rect : rects) {
        // 矩形角度
        float angle = rect.angle;
        float r = float(rect.size.width) / float(rect.size.height);

        //cout<< "宽：" << rect.size.width << endl;
        //cout<< "高：" << rect.size.height << endl;
        //cout<< "宽高比：" << r << endl;
        //cout<< "角度：" << angle << endl;
        //imshow("原图", src);
        // 如果宽高比小于 1，调整角度
        if (r < 1 && angle > 80) {
            angle = angle - 90;
        }

        // 1. 确保矩形在图像范围内
        Rect2f safe_rect;
        safeRect(src, rect, safe_rect);

        // 在原图上截取出矩形区域
        Mat src_rect = src(safe_rect);
        

        // 真正的候选车牌图
        Mat dst;

        // 2. 旋转矩形
        if (angle - 5 < 0 && angle + 5 > 0) {
            // 旋转角度在 -5° 到 5° 之间不进行旋转
            dst = src_rect.clone();
        }
        else {
            // 矩形相对于安全矩形的中心点
            Point2f ref_center = rect.center - safe_rect.tl();
            Mat rotated_mat;
            // 旋转矩形
            rotation(src_rect, rotated_mat, rect.size, ref_center, angle);
            dst = rotated_mat;
            //imshow("旋转后的", dst);
        }

        // 3. 调整大小
        Mat plate_mat;
        plate_mat.create(36, 136, CV_8UC3);  // 136x36 为车牌的标准尺寸
        resize(dst, plate_mat, plate_mat.size());

        // 将校正后的车牌图像加入结果列表
        dst_plates.push_back(plate_mat);
        dst.release();
    }
}

// 计算安全矩形
void safeRect(const Mat& src, const RotatedRect& rotatedRect, Rect2f& safe_rect) {
    // 获取旋转矩形的外接矩形
    Rect2f boundRect = rotatedRect.boundingRect2f();

    // 计算安全矩形的左上角和右下角坐标，确保不超出原图边界
    float t1_x = boundRect.x > 0 ? boundRect.x : 0;
    float t1_y = boundRect.y > 0 ? boundRect.y : 0;
    float br_x = boundRect.x + boundRect.width < src.cols ?
        boundRect.x + boundRect.width - 1 : src.cols - 1;
    float br_y = boundRect.y + boundRect.height < src.rows ?
        boundRect.y + boundRect.height - 1 : src.rows - 1;

    // 计算矩形的宽度和高度
    float width = br_x - t1_x;
    float height = br_y - t1_y;

    // 如果宽高合法，设置安全矩形
    if (width > 0 && height > 0) {
        safe_rect = Rect2f(t1_x, t1_y, width, height);
    }
}

// 旋转矩形
void rotation(const Mat& src, Mat& dst, const Size& rect_size, const Point2f& center, double angle) {
    // 计算旋转矩阵，以矩形中心为旋转中心，进行旋转，角度为指定的角度
    Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);

    // 计算旋转后的矩形的大小，确保其可以容纳整个旋转后的图像
    int max_side = sqrt(pow(src.cols, 2) + pow(src.rows, 2));

    // 进行仿射变换，得到旋转后的图像
    Mat rotated_mat;
    warpAffine(src, rotated_mat, rot_mat, Size(max_side, max_side), INTER_CUBIC);

    // 截取旋转后的矩形区域，保持车牌的宽高比
    getRectSubPix(rotated_mat, rect_size, center, dst);

    // 释放临时图像和旋转矩阵
    rotated_mat.release();
    rot_mat.release();
}

/**
* 通过一行的颜色跳变次数判断是否扫描到了铆钉，
* 一行最小跳变次数为 12，最大为 12 + 8 * 6 = 60。
* 如果该行是铆钉行，则将该行所有像素都涂成黑色（像素值为 0）
*/
bool clearRivet(Mat& plate)
{
    // 1.逐行扫描统计颜色跳变次数保存到集合中
    int minChangeCount = 12;
    vector<int> changeCounts;
    int changeCount;
    for (int i = 0; i < plate.rows; i++)
    {
        for (int j = 0; j < plate.cols - 1; j++)
        {
            int pixel_front = plate.at<char>(i, j);
            int pixel_back = plate.at<char>(i, j + 1);
            if (pixel_front != pixel_back)
            {
                changeCount++;
            }
        }
        changeCounts.push_back(changeCount);
        changeCount = 0;
    }

    // 2.计算字符高度，即满足像素跳变次数的行数
    int charHeight = 0;
    for (int i = 0; i < plate.rows; i++)
    {
        if (changeCounts[i] >= 12 && changeCounts[i] <= 60)
        {
            charHeight++;
        }
    }

    // 3.判断字符高度 & 面积占整个车牌的高度 & 面积的百分比，排除不符合条件的情况
    // 3.1 高度占比小于 0.4 则认为无法识别
    float heightPercent = float(charHeight) / plate.rows;
    if (heightPercent <= 0.4)
    {
        return false;
    }
    // 3.2 面积占比小于 0.15 或大于 0.5 则认为无法识别
    float plate_area = plate.rows * plate.cols;
    // countNonZero 返回非 0 像素点（即白色）个数，或者自己遍历找像素点为 255 的个数也可
    float areaPercent = countNonZero(plate) * 1.0 / plate_area;
    // 小于 0.15 就是蓝背景白字车牌确实达不到识别标准，大于 0.5 是因为
    // 黄背景黑子二值化会把背景转化为白色，由于前面的处理逻辑只能处理
    // 蓝背景车牌，所以黄色车牌的情况也直接认为不可识别
    if (areaPercent <= 0.15 || areaPercent >= 0.5)
    {
        return false;
    }

    // 4.将小于最小颜色跳变次数的行全部涂成黑色
    for (int i = 0; i < plate.rows; i++)
    {
        if (changeCounts[i] < minChangeCount)
        {
            for (int j = 0; j < plate.cols; j++)
            {
                plate.at<char>(i, j) = 0;
            }
        }
    }

    return true;
}

// 规范字符轮廓
bool verifyCharSize(Mat src)
{
    // 最理想情况 车牌字符的标准宽高比
    float aspect = 45.0f / 90.0f;
    // 当前获得矩形的真实宽高比
    float realAspect = (float)src.cols / (float)src.rows;
    // 最小的字符高
    float minHeight = 10.0f;
    // 最大的字符高
    float maxHeight = 35.0f;
    // 1、判断高符合范围  2、宽、高比符合范围
    // 最大宽、高比 最小宽高比
    float error = 0.2f;
    float maxAspect = aspect + aspect * error;//0.85
    float minAspect = 0.05f;

    int plate_area = src.cols * src.rows;
    float areaPercent = countNonZero(src) * 1.0 / plate_area;

    if (areaPercent <= 0.8 && realAspect >= minAspect && realAspect <= maxAspect
        && src.rows >= minHeight &&
        src.rows <= maxHeight) {
        return true;
    }
    return false;
}

// 寻找城市字符
int getCityIndex(vector<Rect> rects)
{
    int cityIndex = 0;
    for (int i = 0; i < rects.size(); i++)
    {
        Rect rect = rects[i];
        int midX = rect.x + rect.width / 2;
        // 如果字符水平方向中点坐标在整个车牌水平坐标的
        // 1/7 ~ 2/7 之间，就认为是目标索引。136 是我们
        // 训练车牌使用的素材的车牌宽度
        if (midX < 136 / 7 * 2 && midX > 136 / 7)
        {
            cityIndex = i;
            break;
        }
    }
    return cityIndex;
}

// 获取汉字字符矩形
void getChineseRect(Rect cityRect, Rect& chineseRect)
{
    // 把宽度稍微扩大一点以包含完整的汉字字符
    // 还有一层理解，就是汉字与城市字符之间的空隙也要计算进去
    float width = cityRect.width * 1.15;

    // 城市轮廓矩形的横坐标
    int x = cityRect.x;

    // 用城市矩形的横坐标减去汉字宽度得到汉字矩形的横坐标
    int newX = x - width;
    chineseRect.x = newX > 0 ? newX : 0;
    chineseRect.y = cityRect.y;
    chineseRect.width = width;
    chineseRect.height = cityRect.height;
}


// 字符分割
void character_segmentation(Mat plate, vector<Mat>& plateCharMats) {
    // 1. 图像预处理：灰度化
    Mat gray;
    cvtColor(plate, gray, COLOR_BGR2GRAY);  // 转为灰度图

    // 2. 二值化（非黑即白，对比更强烈）
    Mat shold;
    threshold(gray, shold, 0, 255, THRESH_OTSU + THRESH_BINARY); // 使用Otsu自适应阈值进行二值化

    // 3. 去除“柳丁”区域（如果去除失败，返回未识别车牌）
    if (!clearRivet(shold)) {
        cout <<  "未识别到车牌" << endl;
    }

    // 4. 提取轮廓
    vector<vector<Point>> contours;
    findContours(shold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找外部轮廓

    // 5. 过滤符合字符尺寸的轮廓
    vector<Rect> vec_ann_rects;
    Mat src_clone = plate.clone();  // 创建一份克隆图像来绘制矩形

    for (const auto& points : contours) {
        Rect rect = boundingRect(points); // 获取每个轮廓的外接矩形
        Mat rectMat = shold(rect);        // 截取该区域

        rectangle(src_clone, rect, Scalar(0, 255, 0), 2); // 用绿色矩形框标记

        // 如果尺寸符合要求，将矩形区域加入 vec_ann_rects
        if (verifyCharSize(rectMat)) {
            // 计算缩放后的矩形框
            Rect scaledRect = rect;
            float scaleFactor = 1.23;  // 比例因子，可以根据需求调整
            scaledRect.x = static_cast<int>(rect.x - (rect.width * (scaleFactor - 1) / 2));
            scaledRect.y = static_cast<int>(rect.y - (rect.height * (scaleFactor - 1) / 2));
            scaledRect.width = static_cast<int>(rect.width * scaleFactor);
            scaledRect.height = static_cast<int>(rect.height * scaleFactor);

            vec_ann_rects.push_back(scaledRect);

            // 在图像上绘制缩放后的矩形框
            //rectangle(src_clone, scaledRect, Scalar(0, 0, 255), 2); // 用红色矩形框标记字符区域

        }
    }

    // 2.2 对矩形轮廓从左至右排序
    sort(vec_ann_rects.begin(), vec_ann_rects.end(), [](const Rect& rect1, const Rect& rect2) {
        return rect1.x < rect2.x;
        });

    // 2.3 获取城市字符轮廓的索引
    int cityIndex = getCityIndex(vec_ann_rects);

    // 2.4 推导汉字字符的轮廓
    Rect chineseRect;
    getChineseRect(vec_ann_rects[cityIndex], chineseRect);

    // 保存字符图像
    //vector<Mat> plateCharMats; //为传入参数

    //plateCharMats.push_back(shold(chineseRect));
    // 处理汉字字符区域，并调整为2x2大小
    //Mat chineseChar = shold(chineseRect);
    //if (!chineseChar.empty()) {
    //    resize(chineseChar, chineseChar, Size(20, 20));  // 调整大小为2x2
    //    plateCharMats.push_back(chineseChar);
    //}

    // 处理汉字字符区域，并调整为2x2大小,从原图中获取
    //Mat chineseChar = plate(chineseRect);
    Mat chineseChar = shold(chineseRect);
    if (!chineseChar.empty()) {
        resize(chineseChar, chineseChar, Size(20, 20));  // 调整大小为2x2
        plateCharMats.push_back(chineseChar);
    }

    // 再获取汉字之后的 6 个字符并保存
    int count = 6;
    if (vec_ann_rects.size() < 6)
    {
        cout << "未识别到车牌606" << endl;
    }

    Mat plate2 = plate.clone();
    // 1. 去噪：使用双边滤波去除噪声，保留边缘
    Mat denoised;
    bilateralFilter(plate2, denoised, 9, 75, 75); // 参数可以调整
    // 2. 灰度化处理
    Mat gray2;
    if (denoised.channels() == 3) { // 如果是彩色图像
        cvtColor(denoised, gray2, COLOR_BGR2GRAY);
    }
    else {
        gray2 = denoised.clone(); // 已经是灰度图像，直接克隆
    }
    //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 定义腐蚀核
    //erode(gray2, plate2, kernel); // 腐蚀操作

    //// 4. 锐化操作
    Mat kernelSharpen = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 6, -1,
        0, -1, 0);  // 锐化核，增强中心像素

    filter2D(plate2, plate2, -1, kernelSharpen); // 锐化操作

    // 5. 二值化处理
    // 使用Otsu的自适应阈值法进行二值化
    threshold(gray2, plate2, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); // 定义腐蚀核
    erode(gray2, plate2, kernel); // 腐蚀操作

    filter2D(plate2, plate2, -1, kernelSharpen); // 锐化操作

    threshold(plate2, plate2, 0, 255, THRESH_BINARY | THRESH_OTSU);

    imshow("处理后的全图", plate2);

    for (int i = cityIndex; i < vec_ann_rects.size() && count; i++, count--)
    {
        //Mat charMat = shold(vec_ann_rects[i]);

        Mat charMat = plate2(vec_ann_rects[i]); // 从原图中获取
        
        if (!charMat.empty()) {
            // 白色腐蚀操作
            Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); // 定义腐蚀核
            //erode(charMat, charMat, kernel); // 腐蚀操作
            ////Canny(charMat, charMat, 50, 150); // Canny 边缘检测
            resize(charMat, charMat, Size(20, 20));  // 调整大小为2x2
            plateCharMats.push_back(charMat);
        }

    }


    //for (Mat& m : plateCharMats) {
    //    imshow("字符" , m);
    //    // 通过输入任意按键查看每一个字符
    //    waitKey();
    //}

    // 在图像上绘制缩放后的矩形框
    //rectangle(src_clone, chineseRect, Scalar(0, 0, 255), 2); // 用红色矩形框标记字符区域

    // 6. 显示图像
    //imshow("字符分割后", src_clone);  // 显示带有矩形框的图像


    // 等待用户按键
    //waitKey(0);
    return;
}



//// 通过模板匹配进行字符识别
//string character_matching(vector<Mat>& plateCharMats) {
//    string result = "";  // 存储识别出的字符
//
//    // 1. 构建标准字库（汉字库 + 数字字母库）
//
//    // 存储汉字模板
//    vector<Mat> provinceTemplateImages;
//    vector<string> provinceLabels;
//
//    // 记录加载汉字库的开始时间
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // 加载汉字库：遍历province文件夹，加载每个省份文件夹下的所有图片
//    for (const auto& entry : fs::directory_iterator(PROVINCE_DIR)) {
//        if (fs::is_directory(entry)) {
//            string provinceName = entry.path().filename().string();
//            for (const auto& file : fs::directory_iterator(entry)) {
//                if (file.path().extension() == ".jpg") {
//                    Mat provinceImg = imread(file.path().string(), IMREAD_GRAYSCALE);
//                    if (!provinceImg.empty()) {
//                        provinceTemplateImages.push_back(provinceImg);
//                        provinceLabels.push_back(provinceName);  // 使用文件夹名作为标签（省份简称）
//                    }
//                }
//            }
//        }
//    }
//
//    // 记录加载汉字库的结束时间并计算时间差
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "加载汉字库用时: " << duration.count() << " 毫秒" << std::endl;
//
//    // 存储数字和字母模板
//    vector<Mat> numAlphaTemplateImages;
//    vector<string> numAlphaLabels;
//
//    // 记录加载数字和字母库的开始时间
//    start = std::chrono::high_resolution_clock::now();
//
//    // 加载数字和字母库：遍历num文件夹，加载0-9, a-z的所有模板
//    for (char ch = '0'; ch <= '9'; ++ch) {
//        string filename = NUM_DIR + string(1, ch) + ".jpg"; // 确保字符被转换为字符串
//        Mat tempImg = imread(filename, IMREAD_GRAYSCALE);
//        if (!tempImg.empty()) {
//            numAlphaTemplateImages.push_back(tempImg);
//            numAlphaLabels.push_back(string(1, ch));  // 使用字符本身作为标签
//        }
//    }
//    for (char ch = 'a'; ch <= 'z'; ++ch) {
//        if (ch == 'i' || ch == 'o')
//        {
//            continue;
//        }
//        string filename = NUM_DIR + string(1, ch) + ".jpg"; // 确保字符被转换为字符串
//        Mat tempImg = imread(filename, IMREAD_GRAYSCALE);
//        if (!tempImg.empty()) {
//            numAlphaTemplateImages.push_back(tempImg);
//            numAlphaLabels.push_back(string(1, ch));  // 使用字符本身作为标签
//        }
//    }
//
//    // 记录加载数字和字母库的结束时间并计算时间差
//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "加载数字字母库用时: " << duration.count() << " 毫秒" << std::endl;
//
//    // 2. 对每个分割出来的字符进行模板匹配
//    bool isFirstChar = true;  // 标记是否为第一张字符图像，进行汉字匹配
//
//    cout << "plateCharMats size: " << plateCharMats.size() << endl;
//    if (plateCharMats.empty()) {
//        cout << "plateCharMats is empty!" << endl;
//    }
//
//    for (const auto& charMat : plateCharMats) {
//        if (charMat.empty()) {
//            cout << "Found empty character mat!" << endl;
//            continue;  // 跳过空的字符图像
//        }
//
//        cout << "Original type: " << charMat.type() << endl;
//
//        // 确保输入图像是灰度图，如果不是则转换
//        if (charMat.channels() > 1) {
//            cvtColor(charMat, charMat, COLOR_BGR2GRAY);  // 转为灰度图
//        }
//
//        // 检查图像的类型
//        if (charMat.type() != CV_32F) {
//            charMat.convertTo(charMat, CV_32F);  // 转换为32位浮点型
//        }
//        else {
//            cout << "charMat is already in CV_32F format" << endl;
//        }
//
//        double maxVal = -1;
//        int bestMatchIdx = -1;
//        string bestMatchLabel = "";
//
//        if (isFirstChar) {
//            // 3. 对第一张字符图像进行汉字匹配
//            for (size_t i = 0; i < provinceTemplateImages.size(); ++i) {
//                if (provinceTemplateImages[i].empty()) {
//                    cout << "Found empty template in provinceTemplateImages!" << endl;
//                    continue;  // 跳过空的模板
//                }
//
//                // 确保模板是灰度图并转换为合适的类型
//                if (provinceTemplateImages[i].channels() > 1) {
//                    cvtColor(provinceTemplateImages[i], provinceTemplateImages[i], COLOR_BGR2GRAY);  // 转为灰度图
//                }
//                provinceTemplateImages[i].convertTo(provinceTemplateImages[i], CV_32F);  // 转换为32位浮点型
//
//                // 调用模板匹配
//                Mat resultImg;
//                matchTemplate(charMat, provinceTemplateImages[i], resultImg, TM_CCOEFF_NORMED);
//                double minVal, maxValTemp;
//                Point minLoc, maxLoc;
//                minMaxLoc(resultImg, &minVal, &maxValTemp, &minLoc, &maxLoc);
//
//                // 设置一个阈值，避免匹配度过低的情况
//                double threshold = 0.8;  // 你可以根据实际情况调整阈值
//                if (maxValTemp > threshold && maxValTemp > maxVal) {
//                    maxVal = maxValTemp;
//                    bestMatchIdx = (int)i;
//                    bestMatchLabel = provinceLabels[bestMatchIdx];
//                    cout << "汉字匹配: " << bestMatchLabel << endl;
//                }
//            }
//            isFirstChar = false;  // 已经处理过第一张字符图像，不再进行汉字匹配
//        }
//
//        if (bestMatchLabel.empty()) {
//            // 4. 对剩下的字符进行数字和字母匹配
//            for (size_t i = 0; i < numAlphaTemplateImages.size(); ++i) {
//                if (numAlphaTemplateImages[i].empty()) {
//                    cout << "Found empty template in numAlphaTemplateImages!" << endl;
//                    continue;  // 跳过空的模板
//                }
//
//                // 确保模板是灰度图并转换为合适的类型
//                if (numAlphaTemplateImages[i].channels() > 1) {
//                    cvtColor(numAlphaTemplateImages[i], numAlphaTemplateImages[i], COLOR_BGR2GRAY);  // 转为灰度图
//                }
//                numAlphaTemplateImages[i].convertTo(numAlphaTemplateImages[i], CV_32F);  // 转换为32位浮点型
//
//                // 调用模板匹配
//                Mat resultImg;
//                matchTemplate(charMat, numAlphaTemplateImages[i], resultImg, TM_CCOEFF_NORMED);
//                double minVal, maxValTemp;
//                Point minLoc, maxLoc;
//                minMaxLoc(resultImg, &minVal, &maxValTemp, &minLoc, &maxLoc);
//
//                // 设置一个阈值，避免匹配度过低的情况
//                double threshold = 0.8;  // 你可以根据实际情况调整阈值
//                if (maxValTemp > threshold && maxValTemp > maxVal) {
//                    maxVal = maxValTemp;
//                    bestMatchIdx = (int)i;
//                    bestMatchLabel = numAlphaLabels[bestMatchIdx];
//                }
//            }
//        }
//
//        // 5. 将匹配到的字符标签添加到结果中
//        if (!bestMatchLabel.empty()) {
//            result += bestMatchLabel;
//        }
//        else {
//            result += "?";  // 如果没有匹配到任何字符，用"?"表示
//        }
//    }
//
//    return result;
//}
