#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/ml.hpp>
#include <opencv2/core/ocl.hpp>
#include <omp.h>  // OpenMP�����ڲ��л�

#include <thread>  // �����߳̿�

#include "Myfunction.h"
#include "CLbp.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

#define     imgRows      20      // ͼ��������ͼ��ĸ߶�
#define     imgCols      20     // ͼ��������ͼ��Ŀ��
#define     lbpNum       256     // LBP���ֲ���ֵģʽ��ֵ��������

// ʡ�ݼ��
const char* provinces[31] = {
    "��", "��", "��", "��", "��", "��", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "³", "��", "��", "��", "��",
    "��", "��", "��", "��", "��", "��", "��", "ԥ", "��", "��",
     "��"
};

char CHARS[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };



//int main()
//{
//    // ����OpenCL����
//    cv::ocl::setUseOpenCL(true);
//    // ���� SVM ģ��
//    string modelPath = "model/svm_small.xml";
//    Ptr<SVM> svm = SVM::load(modelPath);
//    if (svm.empty()) {
//        cout << "�޷�����ģ�ͣ�" << endl;
//        return -1;
//    }
//
//    // ��ȡ�ؼ�֡
//    std::vector<cv::Mat> keyFrames = extractKeyFrames("resource/carvideo.avi");  // ��Ƶ�ļ�·��
//
//    if (keyFrames.empty()) {
//        std::cout << "δ��ȡ���ؼ�֡" << std::endl;
//        return 0;
//    }
//
//    vector<Mat> Firsttarget;
//
//    for (size_t i = 0; i < keyFrames.size(); ++i) {
//        cv::Mat processedFrame = preprocessImage(keyFrames[i]);
//
//        // ֻ����ǿյ�Ԥ����ͼ��
//        if (!processedFrame.empty()) {
//            // ����һ���µ� Mat ���ڴ洢����Ҫ�������
//            Mat src = keyFrames[i].clone(); // ���ڻ���������ԭʼͼ��
//
//            // ���Ҳ���������
//            vector<vector<Point>> contours;
//            findAndDrawContours(processedFrame, src, Firsttarget);
//
//
//
//            //for (Mat& m : Firsttarget) {
//            //    imshow("Sobel ��λ��ѡ����", m);
//            //    // ͨ���������ⰴ���鿴ÿһ����ѡ����
//            //    waitKey();
//            //}
//        }
//
//        // ���� ESC ���˳�ѭ��
//        //if (cv::waitKey(0) == 27) {  // 0 ��ʾ���������ȴ�
//        //    break;
//        //}
//    }
//
//    // �������д���
//    cv::destroyAllWindows();
//
//    return 0;
//}

// ���ز�Ԥ�ⵥ��ͼ���Ƿ�Ϊ����
bool predictLicensePlate(const SVM& svm, const Mat& feature) {
    // ʹ�� SVM ����Ԥ��
    float prediction = svm.predict(feature);
    // ���Ԥ����Ϊ���ࣨ���ƣ������� true
    return (prediction == 1);
}

int main() {

    // ��ʱ������ȡ��ǰʱ���
    double startTime = cv::getTickCount();

    // ���� OpenCL ����
    cv::ocl::setUseOpenCL(true);

    // �����߳��ж��� SVM ָ�� ��ֹ�߳��еľֲ�svm����ʶ��
    Ptr<SVM> svm_Plate = nullptr; // ����ʶ��ģ��
    Ptr<SVM> svm_Num_english = nullptr; // ���ֺ�Ӣ��ʶ��ģ��
    Ptr<SVM> svm_Ch_city = nullptr; // ���ĳ��м��ʶ��ģ��

    // ��ʱ������SVMģ��
    double loadModelStartTime = cv::getTickCount();
    thread svmThread([&]() {
        // ���� SVM ģ�� ������ʶ��
        string modelPath_Plate = "model/svm_small.xml"; // ������ģ��(����ʶ��)
        string modelPath_Num_english = "model/svm_num_special_4.xml"; // ���ֺ�Ӣ��ʶ��ģ��
        string modelPath_Ch_city = "model/svm_ch_city2.xml"; // ���ĳ��м��ʶ��ģ��

        svm_Plate = SVM::load(modelPath_Plate);
        svm_Num_english = SVM::load(modelPath_Num_english);
        svm_Ch_city = SVM::load(modelPath_Ch_city);
        if (svm_Plate.empty()) {
            std::cout << "�޷�����ģ�ͣ�����ʶ�𣩣�" << endl;
            return;
        }
        if (svm_Num_english.empty()) {
            cout << "�޷��������ֺ�Ӣ��ʶ��ģ�� ��" << endl;
            return;
        }
        if (svm_Ch_city.empty()) {
            cout << "�޷��������ĳ��м��ʶ��ģ�� ��" << endl;
            return;
        }
        cout << "SVMģ�ͼ������" << endl;
        });



    // ��ȡ�ؼ�֡
    double extractFramesStartTime = cv::getTickCount();
    std::vector<cv::Mat> keyFrames = extractKeyFrames("resource/carvideo.avi");  // ��Ƶ�ļ�·��
    if (keyFrames.empty()) {
        std::cout << "δ��ȡ���ؼ�֡" << std::endl;
        return 0;
    }
    double extractFramesEndTime = cv::getTickCount();
    cout << "��ȡ�ؼ�֡ʱ��: " << (extractFramesEndTime - extractFramesStartTime) / cv::getTickFrequency() << "��" << endl;

    svmThread.join();

    // ͨ���������ⰴ���鿴ÿһ���ַ�ͼ��
    std::cout << "ģ�ͼ�����ɣ��밴�������ʼʶ��";
    std::cin.get();

    // ���ڴ洢�����ǳ��Ƶĺ�ѡͼ��
    vector<Mat> Firsttarget;

    // ����ÿһ֡��Ƶͼ��
    double processFramesStartTime = cv::getTickCount();
    for (size_t i = 0; i < keyFrames.size(); ++i) {
        cv::Mat processedFrame = preprocessImage(keyFrames[i]);

        // ֻ����ǿյ�Ԥ����ͼ��
        if (!processedFrame.empty()) {
            // ����һ���µ� Mat ���ڴ洢����Ҫ�������
            Mat src = keyFrames[i].clone(); // ���ڻ���������ԭʼͼ��

            // ���Ҳ���������
            findAndDrawContours(processedFrame, src, Firsttarget);
        }
    }
    double processFramesEndTime = cv::getTickCount();
    cout << "����ÿ֡ͼ��ʱ��: " << (processFramesEndTime - processFramesStartTime) / cv::getTickFrequency() << "��" << endl;

    // ���û����ȡ����ѡ���ƣ����˳�
    if (Firsttarget.empty()) {
        cout << "û�м�⵽���ƺ�ѡ��" << endl;
        return 0;
    }

    // ���������ѡ����ͼ��ѡ��������ǳ��Ƶ�һ��
    Mat bestPlate;
    float bestConfidence = 0;  // ��¼��õ����Ŷȣ����Ը���SVM�ĵ÷���������
    double plateProcessingStartTime = cv::getTickCount();
    for (size_t i = 0; i < Firsttarget.size(); ++i) {
        Mat candidatePlate = Firsttarget[i];  // ��ǰ��ѡ����ͼ��

        // ��ȡ LBP ����������������󣬿�����Ҫ�޸�������ȡ��ʽ��
        Mat lbpImg;
        extractLBP2({ candidatePlate }, { lbpImg });  // �����Ѿ������� extractLBP ����

        // ��ȡ��ͼ��� LBP ����
        Mat feature = getLBPH(lbpImg, 256, 17, 4, false);  // ��������getLBPH����
        // ʹ�� SVM Ԥ���Ƿ�Ϊ����
        bool isLicensePlate = predictLicensePlate(*svm_Plate, feature);

        // ���Ԥ��Ϊ���Ʋ��ұȵ�ǰ��õ����Ŷȸ��ߣ��������õĳ���
        if (isLicensePlate) {
            float confidence = svm_Plate->predict(feature);  // ��ȡSVM�ĵ÷���Ϊ���Ŷ�
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestPlate = candidatePlate;
            }
        }
    }
    double plateProcessingEndTime = cv::getTickCount();
    cout << "���ƺ�ѡ����ʱ��: " << (plateProcessingEndTime - plateProcessingStartTime) / cv::getTickFrequency() << "��" << endl;

    // ��ʾ������ǳ��Ƶ�ͼ��
    if (!bestPlate.empty()) {
        imshow("Ŀ�공��", bestPlate);
        cv::imwrite("Ŀ�공��.jpg", bestPlate);
        // �������ʱ
        double totalTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();
        cout << "����ʱ��" << totalTime << "��" << endl;
    }
    else {
        cout << "δ�ҵ�����" << endl;
    }

    // ����Ŀ�공��ͼƬ
    //std::string imagePath = "Ŀ�공��.jpg"; // �滻Ϊʵ��ͼƬ·��
    //Mat bestPlate = imread(imagePath, IMREAD_COLOR); // ���ز�ɫͼƬ

    vector<Mat> plateCharMats; // ��ŷָ����ַ�ͼ��
    // �����ַ��ָ�
    double charSegmentationStartTime = cv::getTickCount();
    character_segmentation(bestPlate, plateCharMats);
    double charSegmentationEndTime = cv::getTickCount();
    cout << "�ַ��ָ�ʱ��: " << (charSegmentationEndTime - charSegmentationStartTime) / cv::getTickFrequency() << "��" << endl;



    int index = 0; // ��������Ψһ���ļ���
    cout << "����" << plateCharMats.size() << "���ַ�" << endl;
    for (Mat& m : plateCharMats) {

        // ����Ψһ���ļ��������� char_0.jpg, char_1.jpg, ...
        std::string filename = "char_12_" + std::to_string(index++) + ".jpg";
        // ��ͼ�񱣴浽�ļ�
        cv::imwrite(filename, m);
        std::cout << "�ַ�ͼ���ѱ���Ϊ�ļ�: " << filename << std::endl;

        // ��ʾ�ַ�ͼ��
        //imshow("�ַ�", m);

        // ͨ���������ⰴ���鿴ÿһ���ַ�ͼ��
        waitKey();
    }

    Mat testMat = Mat::zeros(plateCharMats.size(), lbpNum * 4 * 17, CV_32FC1);

    string platechar = "";
    // ��ÿ���ַ�ͼ�����ʶ��
    double charRecognitionStartTime = cv::getTickCount();
    for (size_t i = 0; i < plateCharMats.size(); ++i) {
        Mat img = plateCharMats[i];  // ��ǰ���ַ�ͼ��
        
        Mat lbpImg = Mat(imgRows, imgCols, CV_8UC1, Scalar(0));
        // ת��Ϊ�Ҷ�ͼ
        //cvtColor(img, img, COLOR_BGR2GRAY);

        // ����ͼ���С
        Mat shrink;
        resize(img, shrink, Size(imgCols + 2 * 1, imgRows + 2 * 1), 0, 0, INTER_LINEAR);

        // ��ȡ LBP ����
        elbp(shrink, lbpImg, 1, 8);  // ʹ��Բ��������ȡ LBP ����

        // ��ȡ LBP ����ֱ��ͼ
        Mat m = getLBPH(lbpImg, lbpNum, 17, 4, false);
        m.row(0).copyTo(testMat.row(i));

        Mat feature;
        testMat.row(i).copyTo(feature);  // ��ȡ��ǰ����ͼ�������

        if (i == 0)
        {
            //imshow("�����ַ�", feature);
            int result = svm_Ch_city->predict(feature);  // Ԥ����ַ������
            // ʹ��provinces�����ȡ��Ӧ���ַ�
            platechar += provinces[result];  // ��Ԥ����ַ���ӵ������
        }
        else
        {
            //imshow("���ֺ�Ӣ���ַ�", feature);
            int result = svm_Num_english->predict(feature);  // Ԥ����ַ������
            if (result < 10)
                platechar += CHARS[result];
            else
                platechar += static_cast<char>('A' + (result - 10));
        }
    }
    double charRecognitionEndTime = cv::getTickCount();
    cout << "�ַ�ʶ��ʱ��: " << (charRecognitionEndTime - charRecognitionStartTime) / cv::getTickFrequency() << "��" << endl;
    cout << "���ƺ�Ϊ��" << platechar << endl;

    waitKey(0);
    // �������д���
    //cv::destroyAllWindows();
    return 0;
}
