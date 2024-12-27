#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/ml.hpp>
#include <opencv2/core/ocl.hpp>
#include <omp.h>  // OpenMP库用于并行化

#include <thread>  // 引入线程库

#include "Myfunction.h"
#include "CLbp.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

#define     imgRows      20      // 图像行数：图像的高度
#define     imgCols      20     // 图像列数：图像的宽度
#define     lbpNum       256     // LBP（局部二值模式）值的种类数

// 省份简称
const char* provinces[31] = {
    "藏", "川", "鄂", "甘", "赣", "贵", "桂", "黑", "沪", "吉",
    "冀", "津", "晋", "京", "辽", "鲁", "蒙", "闽", "宁", "青",
    "琼", "陕", "苏", "皖", "湘", "新", "渝", "豫", "粤", "云",
     "浙"
};

char CHARS[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };



//int main()
//{
//    // 开启OpenCL加速
//    cv::ocl::setUseOpenCL(true);
//    // 加载 SVM 模型
//    string modelPath = "model/svm_small.xml";
//    Ptr<SVM> svm = SVM::load(modelPath);
//    if (svm.empty()) {
//        cout << "无法加载模型！" << endl;
//        return -1;
//    }
//
//    // 提取关键帧
//    std::vector<cv::Mat> keyFrames = extractKeyFrames("resource/carvideo.avi");  // 视频文件路径
//
//    if (keyFrames.empty()) {
//        std::cout << "未提取到关键帧" << std::endl;
//        return 0;
//    }
//
//    vector<Mat> Firsttarget;
//
//    for (size_t i = 0; i < keyFrames.size(); ++i) {
//        cv::Mat processedFrame = preprocessImage(keyFrames[i]);
//
//        // 只处理非空的预处理图像
//        if (!processedFrame.empty()) {
//            // 创建一个新的 Mat 用于存储符合要求的轮廓
//            Mat src = keyFrames[i].clone(); // 用于绘制轮廓的原始图像
//
//            // 查找并绘制轮廓
//            vector<vector<Point>> contours;
//            findAndDrawContours(processedFrame, src, Firsttarget);
//
//
//
//            //for (Mat& m : Firsttarget) {
//            //    imshow("Sobel 定位候选车牌", m);
//            //    // 通过输入任意按键查看每一个候选车牌
//            //    waitKey();
//            //}
//        }
//
//        // 按下 ESC 键退出循环
//        //if (cv::waitKey(0) == 27) {  // 0 表示按键持续等待
//        //    break;
//        //}
//    }
//
//    // 销毁所有窗口
//    cv::destroyAllWindows();
//
//    return 0;
//}

// 加载并预测单张图像是否为车牌
bool predictLicensePlate(const SVM& svm, const Mat& feature) {
    // 使用 SVM 进行预测
    float prediction = svm.predict(feature);
    // 如果预测结果为正类（车牌），返回 true
    return (prediction == 1);
}

int main() {

    // 计时器：获取当前时间戳
    double startTime = cv::getTickCount();

    // 开启 OpenCL 加速
    cv::ocl::setUseOpenCL(true);

    // 在主线程中定义 SVM 指针 防止线程中的局部svm不被识别
    Ptr<SVM> svm_Plate = nullptr; // 车牌识别模型
    Ptr<SVM> svm_Num_english = nullptr; // 数字和英文识别模型
    Ptr<SVM> svm_Ch_city = nullptr; // 中文城市简称识别模型

    // 计时：加载SVM模型
    double loadModelStartTime = cv::getTickCount();
    thread svmThread([&]() {
        // 加载 SVM 模型 （车牌识别）
        string modelPath_Plate = "model/svm_small.xml"; // 轻量化模型(车牌识别)
        string modelPath_Num_english = "model/svm_num_special_4.xml"; // 数字和英文识别模型
        string modelPath_Ch_city = "model/svm_ch_city2.xml"; // 中文城市简称识别模型

        svm_Plate = SVM::load(modelPath_Plate);
        svm_Num_english = SVM::load(modelPath_Num_english);
        svm_Ch_city = SVM::load(modelPath_Ch_city);
        if (svm_Plate.empty()) {
            std::cout << "无法加载模型（车牌识别）！" << endl;
            return;
        }
        if (svm_Num_english.empty()) {
            cout << "无法加载数字和英文识别模型 ！" << endl;
            return;
        }
        if (svm_Ch_city.empty()) {
            cout << "无法加载中文城市简称识别模型 ！" << endl;
            return;
        }
        cout << "SVM模型加载完成" << endl;
        });



    // 提取关键帧
    double extractFramesStartTime = cv::getTickCount();
    std::vector<cv::Mat> keyFrames = extractKeyFrames("resource/carvideo.avi");  // 视频文件路径
    if (keyFrames.empty()) {
        std::cout << "未提取到关键帧" << std::endl;
        return 0;
    }
    double extractFramesEndTime = cv::getTickCount();
    cout << "提取关键帧时间: " << (extractFramesEndTime - extractFramesStartTime) / cv::getTickFrequency() << "秒" << endl;

    svmThread.join();

    // 通过输入任意按键查看每一个字符图像
    std::cout << "模型加载完成，请按任意键开始识别";
    std::cin.get();

    // 用于存储可能是车牌的候选图像
    vector<Mat> Firsttarget;

    // 处理每一帧视频图像
    double processFramesStartTime = cv::getTickCount();
    for (size_t i = 0; i < keyFrames.size(); ++i) {
        cv::Mat processedFrame = preprocessImage(keyFrames[i]);

        // 只处理非空的预处理图像
        if (!processedFrame.empty()) {
            // 创建一个新的 Mat 用于存储符合要求的轮廓
            Mat src = keyFrames[i].clone(); // 用于绘制轮廓的原始图像

            // 查找并绘制轮廓
            findAndDrawContours(processedFrame, src, Firsttarget);
        }
    }
    double processFramesEndTime = cv::getTickCount();
    cout << "处理每帧图像时间: " << (processFramesEndTime - processFramesStartTime) / cv::getTickFrequency() << "秒" << endl;

    // 如果没有提取到候选车牌，则退出
    if (Firsttarget.empty()) {
        cout << "没有检测到车牌候选框" << endl;
        return 0;
    }

    // 批量处理候选车牌图像，选择最可能是车牌的一个
    Mat bestPlate;
    float bestConfidence = 0;  // 记录最好的置信度（可以根据SVM的得分来衡量）
    double plateProcessingStartTime = cv::getTickCount();
    for (size_t i = 0; i < Firsttarget.size(); ++i) {
        Mat candidatePlate = Firsttarget[i];  // 当前候选车牌图像

        // 提取 LBP 特征（根据你的需求，可能需要修改特征提取方式）
        Mat lbpImg;
        extractLBP2({ candidatePlate }, { lbpImg });  // 假设已经定义了 extractLBP 函数

        // 提取该图像的 LBP 特征
        Mat feature = getLBPH(lbpImg, 256, 17, 4, false);  // 假设你有getLBPH函数
        // 使用 SVM 预测是否为车牌
        bool isLicensePlate = predictLicensePlate(*svm_Plate, feature);

        // 如果预测为车牌并且比当前最好的置信度更高，则更新最好的车牌
        if (isLicensePlate) {
            float confidence = svm_Plate->predict(feature);  // 获取SVM的得分作为置信度
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestPlate = candidatePlate;
            }
        }
    }
    double plateProcessingEndTime = cv::getTickCount();
    cout << "车牌候选框处理时间: " << (plateProcessingEndTime - plateProcessingStartTime) / cv::getTickFrequency() << "秒" << endl;

    // 显示最可能是车牌的图像
    if (!bestPlate.empty()) {
        imshow("目标车牌", bestPlate);
        cv::imwrite("目标车牌.jpg", bestPlate);
        // 输出总用时
        double totalTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();
        cout << "总用时：" << totalTime << "秒" << endl;
    }
    else {
        cout << "未找到车牌" << endl;
    }

    // 加载目标车牌图片
    //std::string imagePath = "目标车牌.jpg"; // 替换为实际图片路径
    //Mat bestPlate = imread(imagePath, IMREAD_COLOR); // 加载彩色图片

    vector<Mat> plateCharMats; // 存放分割后的字符图像
    // 进行字符分割
    double charSegmentationStartTime = cv::getTickCount();
    character_segmentation(bestPlate, plateCharMats);
    double charSegmentationEndTime = cv::getTickCount();
    cout << "字符分割时间: " << (charSegmentationEndTime - charSegmentationStartTime) / cv::getTickFrequency() << "秒" << endl;



    int index = 0; // 用于生成唯一的文件名
    cout << "共有" << plateCharMats.size() << "个字符" << endl;
    for (Mat& m : plateCharMats) {

        // 构造唯一的文件名，例如 char_0.jpg, char_1.jpg, ...
        std::string filename = "char_12_" + std::to_string(index++) + ".jpg";
        // 将图像保存到文件
        cv::imwrite(filename, m);
        std::cout << "字符图像已保存为文件: " << filename << std::endl;

        // 显示字符图像
        //imshow("字符", m);

        // 通过输入任意按键查看每一个字符图像
        waitKey();
    }

    Mat testMat = Mat::zeros(plateCharMats.size(), lbpNum * 4 * 17, CV_32FC1);

    string platechar = "";
    // 对每个字符图像进行识别
    double charRecognitionStartTime = cv::getTickCount();
    for (size_t i = 0; i < plateCharMats.size(); ++i) {
        Mat img = plateCharMats[i];  // 当前的字符图像
        
        Mat lbpImg = Mat(imgRows, imgCols, CV_8UC1, Scalar(0));
        // 转换为灰度图
        //cvtColor(img, img, COLOR_BGR2GRAY);

        // 调整图像大小
        Mat shrink;
        resize(img, shrink, Size(imgCols + 2 * 1, imgRows + 2 * 1), 0, 0, INTER_LINEAR);

        // 提取 LBP 特征
        elbp(shrink, lbpImg, 1, 8);  // 使用圆形算子提取 LBP 特征

        // 获取 LBP 特征直方图
        Mat m = getLBPH(lbpImg, lbpNum, 17, 4, false);
        m.row(0).copyTo(testMat.row(i));

        Mat feature;
        testMat.row(i).copyTo(feature);  // 提取当前测试图像的特征

        if (i == 0)
        {
            //imshow("中文字符", feature);
            int result = svm_Ch_city->predict(feature);  // 预测该字符的类别
            // 使用provinces数组获取对应的字符
            platechar += provinces[result];  // 将预测的字符添加到结果中
        }
        else
        {
            //imshow("数字和英文字符", feature);
            int result = svm_Num_english->predict(feature);  // 预测该字符的类别
            if (result < 10)
                platechar += CHARS[result];
            else
                platechar += static_cast<char>('A' + (result - 10));
        }
    }
    double charRecognitionEndTime = cv::getTickCount();
    cout << "字符识别时间: " << (charRecognitionEndTime - charRecognitionStartTime) / cv::getTickFrequency() << "秒" << endl;
    cout << "车牌号为：" << platechar << endl;

    waitKey(0);
    // 销毁所有窗口
    //cv::destroyAllWindows();
    return 0;
}
