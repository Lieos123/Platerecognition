#include "Myfunction.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include <chrono>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;


// ����������Ƶ����ȡ�ؼ�֡
std::vector<cv::Mat> extractKeyFrames(const std::string& videoPath) {
    cv::VideoCapture video(videoPath);

    if (!video.isOpened()) {
        std::cerr << "�����޷�����Ƶ�ļ�" << std::endl;
        return {};
    }

    std::vector<cv::Mat> keyFrames;
    cv::Mat prevKeyFrame;
    int frameCount = 0;

    while (true) {
        cv::Mat frame;
        video >> frame;

        if (frame.empty()) {
            break; // ��ȡ���ļ�ĩβ
        }

        if (frameCount == 0) {
            // ��һ֡Ϊ�ؼ�֡
            keyFrames.push_back(frame.clone());
            prevKeyFrame = frame.clone();
        }
        else if (frameCount % FRAME_SKIP == 0) {
            // ÿ�� FRAME_SKIP ֡�������
            cv::Mat diff;
            cv::absdiff(prevKeyFrame, frame, diff); // ������֮֡��ľ��Բ�ֵ
            cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY); // תΪ�Ҷ�ͼ��
            double diffSum = cv::sum(diff)[0] / (diff.rows * diff.cols); // ��������ֵ

            if (diffSum > THRESHOLD) {
                keyFrames.push_back(frame.clone());
                prevKeyFrame = frame.clone(); // ������һ�ؼ�֡
            }
        }

        frameCount++;
    }

    return keyFrames;
}

cv::Mat sharpenImage(const cv::Mat& blurredImg) {
    // ����һ���򵥵��񻯾����
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);

    // ʹ�þ���˽�����
    cv::Mat sharpenedImg;
    cv::filter2D(blurredImg, sharpenedImg, -1, kernel);
    return sharpenedImg;
}

cv::Mat preprocessImage(const cv::Mat& frame) {
    // �������ͼ��Ϊ�գ��򷵻ؿվ���
    if (frame.empty()) {
        std::cerr << "��ͼ��" << std::endl;
        return cv::Mat();
    }

    // 1. ��˹ģ������������
    cv::Mat blurredImg;
    cv::GaussianBlur(frame, blurredImg, cv::Size(5, 5), 0);
    //cv::imshow("��˹ģ��", blurredImg);  // ��ʾ��˹ģ�����ͼ��

    // 2. �񻯣���ǿϸ��
    cv::Mat sharpenedImg = sharpenImage(blurredImg);
    //cv::imshow("��ͼ��", sharpenedImg);  // ��ʾ�񻯺��ͼ��

    // 3. �ҶȻ�����ͼ��ת��Ϊ�Ҷ�ͼ��
    cv::Mat grayImg;
    cv::cvtColor(sharpenedImg, grayImg, cv::COLOR_BGR2GRAY);
    //cv::imshow("�Ҷ�ͼ��", grayImg);  // ��ʾ�Ҷ�ͼ��

    // 4. Sobel �������㣺���ͼ���Ե
    cv::Mat gradX, gradY, grad;
    cv::Sobel(grayImg, gradX, CV_32F, 1, 0, 3);  // Sobel X����
    cv::Sobel(grayImg, gradY, CV_32F, 0, 1, 3);  // Sobel Y����
    cv::magnitude(gradX, gradY, grad);  // �����ݶȵķ�ֵ
    grad.convertTo(grad, CV_8U);  // ת��Ϊ8λͼ��
    //cv::imshow("Sobel��������", grad);  // ��ʾSobel������ͼ��

    // 5. ��ֵ������ͼ��תΪ�ڰ�ͼ��
    cv::Mat binaryImg;
    double thresholdVal = 100;  // ���ù̶���ֵ
    cv::threshold(grad, binaryImg, thresholdVal, 255, cv::THRESH_BINARY);
    //cv::imshow("��ֵ��ͼ��", binaryImg);  // ��ʾ��ֵ�����ͼ��

    // 6. ��̬ѧ�ղ���������С�������ӿ�϶
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));  // ��ȡ���νṹԪ��
    cv::Mat closedImg;
    cv::morphologyEx(binaryImg, closedImg, cv::MORPH_CLOSE, element);
    //cv::imshow("��̬ѧ�ղ���", closedImg);  // ��ʾ�ղ������ͼ��

    // 7. ��ʴ��������ͼ���еİ�ɫ������и�ʴ�����ٸ�ʴǿ��
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));  
    cv::Mat erodedImg;
    int erosionIterations = 1;  // ���ý��ٵĸ�ʴ��������
    cv::erode(closedImg, erodedImg, element2, cv::Point(-1, -1), erosionIterations);  // ʹ�ý��ٵĵ�������
    //cv::imshow("��ȸ�ʴͼ��", erodedImg);  // ��ʾ��ʴ���ͼ��

    cv::waitKey(0);  // �ȴ��û�����
    cv::destroyAllWindows();  // �ر����д���

    return erodedImg;  // ���ظ�ʴ���ͼ��
}

// �ж������ߴ��Ƿ���Ϲ��
bool verifySizes(const RotatedRect& rotatedRect) {
    // �ݴ���
    float error = 0.3;

    // �����߱ȣ�ѵ������ʹ�õĳ��ƹ��Ϊ 136x36����˽������������߱ȼ��㣩
    float aspect = float(136) / float(36);

    // �����ݴ��ʼ������С��߱�������߱�
    float aspectMin = (1 - error) * aspect;
    float aspectMax = (1 + error) * aspect;

    // ��ʵ��߱�
    float realAspect = float(rotatedRect.size.width) / float(rotatedRect.size.height);
    if (realAspect < 1) {
        realAspect = float(rotatedRect.size.height) / float(rotatedRect.size.width);
    }

    // ��ʵ���
    float area = rotatedRect.size.width * rotatedRect.size.height;

    // ��С������������������ϵĶ���
    // ������ž��У�������ʱ����������һ��Ҳû��ϵ����ֻ�ǳ���ɸѡ
    int areaMin = 44 * aspect * 14;
    int areaMax = 440 * aspect * 140;

    // �ж�������߱��Ƿ���Ϲ��
    if ((area < areaMin || area > areaMax) || (realAspect < aspectMin || realAspect > aspectMax)) {
        return false;
    }

    return true; // ���Ϲ��
}

//// ��������������Ӿ���
//void findAndDrawContours(const Mat& preprocessedImage, Mat& src) {
//    // ȷ��ͼ��Ϊ��
//    if (preprocessedImage.empty()) {
//        cerr << "��ͼ��" << endl;
//        return;
//    }
//
//    // �洢����
//    vector<vector<Point>> contours;
//
//    // ��������
//    findContours(preprocessedImage, // �����ֵͼ���Ѿ���Ԥ����Ĺؼ�֡��
//        contours,           // �洢����������
//        RETR_EXTERNAL,      // ��������ģʽ��ֻ����ⲿ����
//        CHAIN_APPROX_NONE); // ��������ģʽ�������н��ƣ���������������
//
//    // ������������
//    RotatedRect rotatedRect;
//    vector<RotatedRect> vec_sobel_rects;  // �洢���ϳߴ��������
//    for (const vector<Point>& points : contours) {
//        // ��ȡÿ����������С�����ת����
//        rotatedRect = minAreaRect(points);
//
//        // ��ȡԭʼ���εĿ�Ⱥ͸߶�
//        float width = rotatedRect.size.width;
//        float height = rotatedRect.size.height;
//
//        // ������γߴ磨��������
//        float scale = 1.3;  // �������ӣ����Ը����������
//        rotatedRect.size.width = width * scale;
//        rotatedRect.size.height = height * scale;
//
//        // ��ԭͼ�ϻ��Ƹ���ת���ε���Ӿ��Σ����β�������ת��
//        rectangle(src, rotatedRect.boundingRect(), Scalar(0, 0, 255), 2);
//    }
//
//    // ��ʾ���ͼ��
//    //imshow("������", src);
//    
//    // ����ɫ���λ������ϳߴ��������
//    for (const RotatedRect& rect : vec_sobel_rects) {
//        rectangle(src, rect.boundingRect(), Scalar(0, 255, 0));
//    }
//    imshow("�ߴ��ж�", src);
//    
//    //waitKey(0);  // �ȴ��û�����
//    destroyAllWindows();  // �ر����д���
//}

// ��������������Ӿ���
void findAndDrawContours(const Mat& preprocessedImage, Mat& src, vector<Mat>& Firsttarget) {
    // ȷ��ͼ��Ϊ��
    if (preprocessedImage.empty()) {
        cerr << "��ͼ��" << endl;
        return;
    }

    // �洢����
    vector<vector<Point>> contours;

    // ��������
    findContours(preprocessedImage, // �����ֵͼ���Ѿ���Ԥ����Ĺؼ�֡��
        contours,           // �洢����������
        RETR_EXTERNAL,      // ��������ģʽ��ֻ����ⲿ����
        CHAIN_APPROX_NONE); // ��������ģʽ�������н��ƣ���������������

    // �洢���ϳߴ���ľ���
    vector<RotatedRect> vec_sobel_rects;  // �洢���ϳߴ��������
    for (const vector<Point>& points : contours) {
        // ��ȡÿ����������С�����ת����
        RotatedRect rotatedRect = minAreaRect(points);

        // ��ȡԭʼ���εĿ�Ⱥ͸߶�
        float width = rotatedRect.size.width;
        float height = rotatedRect.size.height;

        // ������γߴ磨��������
        float scale = 1.22;  // �������ӣ����Ը����������
        rotatedRect.size.width = width * scale;
        rotatedRect.size.height = height * scale;

        // �ж������Ƿ���ϳߴ���
        if (verifySizes(rotatedRect)) {
            vec_sobel_rects.push_back(rotatedRect);
        }

        // ��ԭͼ�ϻ��Ƹ���ת���ε���Ӿ��Σ����β�������ת��
        rectangle(src, rotatedRect.boundingRect(), Scalar(0, 0, 255), 2);
    }

    // ��ʾ���ϳߴ�Ҫ�������
    for (const RotatedRect& rect : vec_sobel_rects) {
        //rectangle(src, rect.boundingRect(), Scalar(255, 0, 0), 2);
        //imshow("��ǰ���Ϲ��ľ���", src);
        //cout<< "��ǰ���Ϲ��ľ��ο�" << rect.size.width << endl;
        //cout<< "��ǰ���Ϲ��ľ��θߣ�" << rect.size.height << endl;
        //cout<< "��ǰ���Ϲ��ľ��νǶ�" << rect.angle << endl;
        rectangle(src, rect.boundingRect(), Scalar(0, 255, 0), 2);
    }
    
    //// ����tortuosity���о��ν���
    vector<Mat> dst_plates;
    tortuosity(src, vec_sobel_rects, dst_plates);

    // �ж϶�λ��ѡ�����Ƿ�Ϊ��
    if (dst_plates.empty()) {
        cout << "δ�ҵ�����Ҫ��ĳ��ƺ�ѡͼ��" << endl;
    }
    else {
        // �鿴������ĺ�ѡ����
        //for (Mat& m : dst_plates) {
        //    imshow("Sobel ��λ��ѡ����", m);
        //    // ͨ���������ⰴ���鿴ÿһ����ѡ����
        //    waitKey();
        //}
    }

    //Firsttarget = dst_plates; // ��������ĺ�ѡ���ƴ洢�� Firsttarget ��
    for (const Mat& plate : dst_plates) {
        // ��������ĺ�ѡ���ƴ洢�� Firsttarget ��
        Firsttarget.push_back(plate);
    }


    //imshow("�ߴ��ж�", src);
    destroyAllWindows();  // �ر����д���
}


// ��������ͼ��
void tortuosity(const Mat& src, vector<RotatedRect>& rects, vector<Mat>& dst_plates) {
    // ����Ҫ����ľ���
    for (const RotatedRect& rect : rects) {
        // ���νǶ�
        float angle = rect.angle;
        float r = float(rect.size.width) / float(rect.size.height);

        //cout<< "��" << rect.size.width << endl;
        //cout<< "�ߣ�" << rect.size.height << endl;
        //cout<< "��߱ȣ�" << r << endl;
        //cout<< "�Ƕȣ�" << angle << endl;
        //imshow("ԭͼ", src);
        // �����߱�С�� 1�������Ƕ�
        if (r < 1 && angle > 80) {
            angle = angle - 90;
        }

        // 1. ȷ��������ͼ��Χ��
        Rect2f safe_rect;
        safeRect(src, rect, safe_rect);

        // ��ԭͼ�Ͻ�ȡ����������
        Mat src_rect = src(safe_rect);
        

        // �����ĺ�ѡ����ͼ
        Mat dst;

        // 2. ��ת����
        if (angle - 5 < 0 && angle + 5 > 0) {
            // ��ת�Ƕ��� -5�� �� 5�� ֮�䲻������ת
            dst = src_rect.clone();
        }
        else {
            // ��������ڰ�ȫ���ε����ĵ�
            Point2f ref_center = rect.center - safe_rect.tl();
            Mat rotated_mat;
            // ��ת����
            rotation(src_rect, rotated_mat, rect.size, ref_center, angle);
            dst = rotated_mat;
            //imshow("��ת���", dst);
        }

        // 3. ������С
        Mat plate_mat;
        plate_mat.create(36, 136, CV_8UC3);  // 136x36 Ϊ���Ƶı�׼�ߴ�
        resize(dst, plate_mat, plate_mat.size());

        // ��У����ĳ���ͼ��������б�
        dst_plates.push_back(plate_mat);
        dst.release();
    }
}

// ���㰲ȫ����
void safeRect(const Mat& src, const RotatedRect& rotatedRect, Rect2f& safe_rect) {
    // ��ȡ��ת���ε���Ӿ���
    Rect2f boundRect = rotatedRect.boundingRect2f();

    // ���㰲ȫ���ε����ϽǺ����½����꣬ȷ��������ԭͼ�߽�
    float t1_x = boundRect.x > 0 ? boundRect.x : 0;
    float t1_y = boundRect.y > 0 ? boundRect.y : 0;
    float br_x = boundRect.x + boundRect.width < src.cols ?
        boundRect.x + boundRect.width - 1 : src.cols - 1;
    float br_y = boundRect.y + boundRect.height < src.rows ?
        boundRect.y + boundRect.height - 1 : src.rows - 1;

    // ������εĿ�Ⱥ͸߶�
    float width = br_x - t1_x;
    float height = br_y - t1_y;

    // �����ߺϷ������ð�ȫ����
    if (width > 0 && height > 0) {
        safe_rect = Rect2f(t1_x, t1_y, width, height);
    }
}

// ��ת����
void rotation(const Mat& src, Mat& dst, const Size& rect_size, const Point2f& center, double angle) {
    // ������ת�����Ծ�������Ϊ��ת���ģ�������ת���Ƕ�Ϊָ���ĽǶ�
    Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);

    // ������ת��ľ��εĴ�С��ȷ�����������������ת���ͼ��
    int max_side = sqrt(pow(src.cols, 2) + pow(src.rows, 2));

    // ���з���任���õ���ת���ͼ��
    Mat rotated_mat;
    warpAffine(src, rotated_mat, rot_mat, Size(max_side, max_side), INTER_CUBIC);

    // ��ȡ��ת��ľ������򣬱��ֳ��ƵĿ�߱�
    getRectSubPix(rotated_mat, rect_size, center, dst);

    // �ͷ���ʱͼ�����ת����
    rotated_mat.release();
    rot_mat.release();
}

/**
* ͨ��һ�е���ɫ��������ж��Ƿ�ɨ�赽��í����
* һ����С�������Ϊ 12�����Ϊ 12 + 8 * 6 = 60��
* ���������í���У��򽫸����������ض�Ϳ�ɺ�ɫ������ֵΪ 0��
*/
bool clearRivet(Mat& plate)
{
    // 1.����ɨ��ͳ����ɫ����������浽������
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

    // 2.�����ַ��߶ȣ������������������������
    int charHeight = 0;
    for (int i = 0; i < plate.rows; i++)
    {
        if (changeCounts[i] >= 12 && changeCounts[i] <= 60)
        {
            charHeight++;
        }
    }

    // 3.�ж��ַ��߶� & ���ռ�������Ƶĸ߶� & ����İٷֱȣ��ų����������������
    // 3.1 �߶�ռ��С�� 0.4 ����Ϊ�޷�ʶ��
    float heightPercent = float(charHeight) / plate.rows;
    if (heightPercent <= 0.4)
    {
        return false;
    }
    // 3.2 ���ռ��С�� 0.15 ����� 0.5 ����Ϊ�޷�ʶ��
    float plate_area = plate.rows * plate.cols;
    // countNonZero ���ط� 0 ���ص㣨����ɫ�������������Լ����������ص�Ϊ 255 �ĸ���Ҳ��
    float areaPercent = countNonZero(plate) * 1.0 / plate_area;
    // С�� 0.15 �������������ֳ���ȷʵ�ﲻ��ʶ���׼������ 0.5 ����Ϊ
    // �Ʊ������Ӷ�ֵ����ѱ���ת��Ϊ��ɫ������ǰ��Ĵ����߼�ֻ�ܴ���
    // ���������ƣ����Ի�ɫ���Ƶ����Ҳֱ����Ϊ����ʶ��
    if (areaPercent <= 0.15 || areaPercent >= 0.5)
    {
        return false;
    }

    // 4.��С����С��ɫ�����������ȫ��Ϳ�ɺ�ɫ
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

// �淶�ַ�����
bool verifyCharSize(Mat src)
{
    // ��������� �����ַ��ı�׼��߱�
    float aspect = 45.0f / 90.0f;
    // ��ǰ��þ��ε���ʵ��߱�
    float realAspect = (float)src.cols / (float)src.rows;
    // ��С���ַ���
    float minHeight = 10.0f;
    // �����ַ���
    float maxHeight = 35.0f;
    // 1���жϸ߷��Ϸ�Χ  2�����߱ȷ��Ϸ�Χ
    // �����߱� ��С��߱�
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

// Ѱ�ҳ����ַ�
int getCityIndex(vector<Rect> rects)
{
    int cityIndex = 0;
    for (int i = 0; i < rects.size(); i++)
    {
        Rect rect = rects[i];
        int midX = rect.x + rect.width / 2;
        // ����ַ�ˮƽ�����е���������������ˮƽ�����
        // 1/7 ~ 2/7 ֮�䣬����Ϊ��Ŀ��������136 ������
        // ѵ������ʹ�õ��زĵĳ��ƿ��
        if (midX < 136 / 7 * 2 && midX > 136 / 7)
        {
            cityIndex = i;
            break;
        }
    }
    return cityIndex;
}

// ��ȡ�����ַ�����
void getChineseRect(Rect cityRect, Rect& chineseRect)
{
    // �ѿ����΢����һ���԰��������ĺ����ַ�
    // ����һ����⣬���Ǻ���������ַ�֮��Ŀ�϶ҲҪ�����ȥ
    float width = cityRect.width * 1.15;

    // �����������εĺ�����
    int x = cityRect.x;

    // �ó��о��εĺ������ȥ���ֿ�ȵõ����־��εĺ�����
    int newX = x - width;
    chineseRect.x = newX > 0 ? newX : 0;
    chineseRect.y = cityRect.y;
    chineseRect.width = width;
    chineseRect.height = cityRect.height;
}


// �ַ��ָ�
void character_segmentation(Mat plate, vector<Mat>& plateCharMats) {
    // 1. ͼ��Ԥ�����ҶȻ�
    Mat gray;
    cvtColor(plate, gray, COLOR_BGR2GRAY);  // תΪ�Ҷ�ͼ

    // 2. ��ֵ�����Ǻڼ��ף��Աȸ�ǿ�ң�
    Mat shold;
    threshold(gray, shold, 0, 255, THRESH_OTSU + THRESH_BINARY); // ʹ��Otsu����Ӧ��ֵ���ж�ֵ��

    // 3. ȥ�����������������ȥ��ʧ�ܣ�����δʶ���ƣ�
    if (!clearRivet(shold)) {
        cout <<  "δʶ�𵽳���" << endl;
    }

    // 4. ��ȡ����
    vector<vector<Point>> contours;
    findContours(shold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); // �����ⲿ����

    // 5. ���˷����ַ��ߴ������
    vector<Rect> vec_ann_rects;
    Mat src_clone = plate.clone();  // ����һ�ݿ�¡ͼ�������ƾ���

    for (const auto& points : contours) {
        Rect rect = boundingRect(points); // ��ȡÿ����������Ӿ���
        Mat rectMat = shold(rect);        // ��ȡ������

        rectangle(src_clone, rect, Scalar(0, 255, 0), 2); // ����ɫ���ο���

        // ����ߴ����Ҫ�󣬽������������ vec_ann_rects
        if (verifyCharSize(rectMat)) {
            // �������ź�ľ��ο�
            Rect scaledRect = rect;
            float scaleFactor = 1.23;  // �������ӣ����Ը����������
            scaledRect.x = static_cast<int>(rect.x - (rect.width * (scaleFactor - 1) / 2));
            scaledRect.y = static_cast<int>(rect.y - (rect.height * (scaleFactor - 1) / 2));
            scaledRect.width = static_cast<int>(rect.width * scaleFactor);
            scaledRect.height = static_cast<int>(rect.height * scaleFactor);

            vec_ann_rects.push_back(scaledRect);

            // ��ͼ���ϻ������ź�ľ��ο�
            //rectangle(src_clone, scaledRect, Scalar(0, 0, 255), 2); // �ú�ɫ���ο����ַ�����

        }
    }

    // 2.2 �Ծ�������������������
    sort(vec_ann_rects.begin(), vec_ann_rects.end(), [](const Rect& rect1, const Rect& rect2) {
        return rect1.x < rect2.x;
        });

    // 2.3 ��ȡ�����ַ�����������
    int cityIndex = getCityIndex(vec_ann_rects);

    // 2.4 �Ƶ������ַ�������
    Rect chineseRect;
    getChineseRect(vec_ann_rects[cityIndex], chineseRect);

    // �����ַ�ͼ��
    //vector<Mat> plateCharMats; //Ϊ�������

    //plateCharMats.push_back(shold(chineseRect));
    // �������ַ����򣬲�����Ϊ2x2��С
    //Mat chineseChar = shold(chineseRect);
    //if (!chineseChar.empty()) {
    //    resize(chineseChar, chineseChar, Size(20, 20));  // ������СΪ2x2
    //    plateCharMats.push_back(chineseChar);
    //}

    // �������ַ����򣬲�����Ϊ2x2��С,��ԭͼ�л�ȡ
    //Mat chineseChar = plate(chineseRect);
    Mat chineseChar = shold(chineseRect);
    if (!chineseChar.empty()) {
        resize(chineseChar, chineseChar, Size(20, 20));  // ������СΪ2x2
        plateCharMats.push_back(chineseChar);
    }

    // �ٻ�ȡ����֮��� 6 ���ַ�������
    int count = 6;
    if (vec_ann_rects.size() < 6)
    {
        cout << "δʶ�𵽳���606" << endl;
    }

    Mat plate2 = plate.clone();
    // 1. ȥ�룺ʹ��˫���˲�ȥ��������������Ե
    Mat denoised;
    bilateralFilter(plate2, denoised, 9, 75, 75); // �������Ե���
    // 2. �ҶȻ�����
    Mat gray2;
    if (denoised.channels() == 3) { // ����ǲ�ɫͼ��
        cvtColor(denoised, gray2, COLOR_BGR2GRAY);
    }
    else {
        gray2 = denoised.clone(); // �Ѿ��ǻҶ�ͼ��ֱ�ӿ�¡
    }
    //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // ���帯ʴ��
    //erode(gray2, plate2, kernel); // ��ʴ����

    //// 4. �񻯲���
    Mat kernelSharpen = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 6, -1,
        0, -1, 0);  // �񻯺ˣ���ǿ��������

    filter2D(plate2, plate2, -1, kernelSharpen); // �񻯲���

    // 5. ��ֵ������
    // ʹ��Otsu������Ӧ��ֵ�����ж�ֵ��
    threshold(gray2, plate2, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); // ���帯ʴ��
    erode(gray2, plate2, kernel); // ��ʴ����

    filter2D(plate2, plate2, -1, kernelSharpen); // �񻯲���

    threshold(plate2, plate2, 0, 255, THRESH_BINARY | THRESH_OTSU);

    imshow("������ȫͼ", plate2);

    for (int i = cityIndex; i < vec_ann_rects.size() && count; i++, count--)
    {
        //Mat charMat = shold(vec_ann_rects[i]);

        Mat charMat = plate2(vec_ann_rects[i]); // ��ԭͼ�л�ȡ
        
        if (!charMat.empty()) {
            // ��ɫ��ʴ����
            Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); // ���帯ʴ��
            //erode(charMat, charMat, kernel); // ��ʴ����
            ////Canny(charMat, charMat, 50, 150); // Canny ��Ե���
            resize(charMat, charMat, Size(20, 20));  // ������СΪ2x2
            plateCharMats.push_back(charMat);
        }

    }


    //for (Mat& m : plateCharMats) {
    //    imshow("�ַ�" , m);
    //    // ͨ���������ⰴ���鿴ÿһ���ַ�
    //    waitKey();
    //}

    // ��ͼ���ϻ������ź�ľ��ο�
    //rectangle(src_clone, chineseRect, Scalar(0, 0, 255), 2); // �ú�ɫ���ο����ַ�����

    // 6. ��ʾͼ��
    //imshow("�ַ��ָ��", src_clone);  // ��ʾ���о��ο��ͼ��


    // �ȴ��û�����
    //waitKey(0);
    return;
}



//// ͨ��ģ��ƥ������ַ�ʶ��
//string character_matching(vector<Mat>& plateCharMats) {
//    string result = "";  // �洢ʶ������ַ�
//
//    // 1. ������׼�ֿ⣨���ֿ� + ������ĸ�⣩
//
//    // �洢����ģ��
//    vector<Mat> provinceTemplateImages;
//    vector<string> provinceLabels;
//
//    // ��¼���غ��ֿ�Ŀ�ʼʱ��
//    auto start = std::chrono::high_resolution_clock::now();
//
//    // ���غ��ֿ⣺����province�ļ��У�����ÿ��ʡ���ļ����µ�����ͼƬ
//    for (const auto& entry : fs::directory_iterator(PROVINCE_DIR)) {
//        if (fs::is_directory(entry)) {
//            string provinceName = entry.path().filename().string();
//            for (const auto& file : fs::directory_iterator(entry)) {
//                if (file.path().extension() == ".jpg") {
//                    Mat provinceImg = imread(file.path().string(), IMREAD_GRAYSCALE);
//                    if (!provinceImg.empty()) {
//                        provinceTemplateImages.push_back(provinceImg);
//                        provinceLabels.push_back(provinceName);  // ʹ���ļ�������Ϊ��ǩ��ʡ�ݼ�ƣ�
//                    }
//                }
//            }
//        }
//    }
//
//    // ��¼���غ��ֿ�Ľ���ʱ�䲢����ʱ���
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "���غ��ֿ���ʱ: " << duration.count() << " ����" << std::endl;
//
//    // �洢���ֺ���ĸģ��
//    vector<Mat> numAlphaTemplateImages;
//    vector<string> numAlphaLabels;
//
//    // ��¼�������ֺ���ĸ��Ŀ�ʼʱ��
//    start = std::chrono::high_resolution_clock::now();
//
//    // �������ֺ���ĸ�⣺����num�ļ��У�����0-9, a-z������ģ��
//    for (char ch = '0'; ch <= '9'; ++ch) {
//        string filename = NUM_DIR + string(1, ch) + ".jpg"; // ȷ���ַ���ת��Ϊ�ַ���
//        Mat tempImg = imread(filename, IMREAD_GRAYSCALE);
//        if (!tempImg.empty()) {
//            numAlphaTemplateImages.push_back(tempImg);
//            numAlphaLabels.push_back(string(1, ch));  // ʹ���ַ�������Ϊ��ǩ
//        }
//    }
//    for (char ch = 'a'; ch <= 'z'; ++ch) {
//        if (ch == 'i' || ch == 'o')
//        {
//            continue;
//        }
//        string filename = NUM_DIR + string(1, ch) + ".jpg"; // ȷ���ַ���ת��Ϊ�ַ���
//        Mat tempImg = imread(filename, IMREAD_GRAYSCALE);
//        if (!tempImg.empty()) {
//            numAlphaTemplateImages.push_back(tempImg);
//            numAlphaLabels.push_back(string(1, ch));  // ʹ���ַ�������Ϊ��ǩ
//        }
//    }
//
//    // ��¼�������ֺ���ĸ��Ľ���ʱ�䲢����ʱ���
//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    std::cout << "����������ĸ����ʱ: " << duration.count() << " ����" << std::endl;
//
//    // 2. ��ÿ���ָ�������ַ�����ģ��ƥ��
//    bool isFirstChar = true;  // ����Ƿ�Ϊ��һ���ַ�ͼ�񣬽��к���ƥ��
//
//    cout << "plateCharMats size: " << plateCharMats.size() << endl;
//    if (plateCharMats.empty()) {
//        cout << "plateCharMats is empty!" << endl;
//    }
//
//    for (const auto& charMat : plateCharMats) {
//        if (charMat.empty()) {
//            cout << "Found empty character mat!" << endl;
//            continue;  // �����յ��ַ�ͼ��
//        }
//
//        cout << "Original type: " << charMat.type() << endl;
//
//        // ȷ������ͼ���ǻҶ�ͼ�����������ת��
//        if (charMat.channels() > 1) {
//            cvtColor(charMat, charMat, COLOR_BGR2GRAY);  // תΪ�Ҷ�ͼ
//        }
//
//        // ���ͼ�������
//        if (charMat.type() != CV_32F) {
//            charMat.convertTo(charMat, CV_32F);  // ת��Ϊ32λ������
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
//            // 3. �Ե�һ���ַ�ͼ����к���ƥ��
//            for (size_t i = 0; i < provinceTemplateImages.size(); ++i) {
//                if (provinceTemplateImages[i].empty()) {
//                    cout << "Found empty template in provinceTemplateImages!" << endl;
//                    continue;  // �����յ�ģ��
//                }
//
//                // ȷ��ģ���ǻҶ�ͼ��ת��Ϊ���ʵ�����
//                if (provinceTemplateImages[i].channels() > 1) {
//                    cvtColor(provinceTemplateImages[i], provinceTemplateImages[i], COLOR_BGR2GRAY);  // תΪ�Ҷ�ͼ
//                }
//                provinceTemplateImages[i].convertTo(provinceTemplateImages[i], CV_32F);  // ת��Ϊ32λ������
//
//                // ����ģ��ƥ��
//                Mat resultImg;
//                matchTemplate(charMat, provinceTemplateImages[i], resultImg, TM_CCOEFF_NORMED);
//                double minVal, maxValTemp;
//                Point minLoc, maxLoc;
//                minMaxLoc(resultImg, &minVal, &maxValTemp, &minLoc, &maxLoc);
//
//                // ����һ����ֵ������ƥ��ȹ��͵����
//                double threshold = 0.8;  // ����Ը���ʵ�����������ֵ
//                if (maxValTemp > threshold && maxValTemp > maxVal) {
//                    maxVal = maxValTemp;
//                    bestMatchIdx = (int)i;
//                    bestMatchLabel = provinceLabels[bestMatchIdx];
//                    cout << "����ƥ��: " << bestMatchLabel << endl;
//                }
//            }
//            isFirstChar = false;  // �Ѿ��������һ���ַ�ͼ�񣬲��ٽ��к���ƥ��
//        }
//
//        if (bestMatchLabel.empty()) {
//            // 4. ��ʣ�µ��ַ��������ֺ���ĸƥ��
//            for (size_t i = 0; i < numAlphaTemplateImages.size(); ++i) {
//                if (numAlphaTemplateImages[i].empty()) {
//                    cout << "Found empty template in numAlphaTemplateImages!" << endl;
//                    continue;  // �����յ�ģ��
//                }
//
//                // ȷ��ģ���ǻҶ�ͼ��ת��Ϊ���ʵ�����
//                if (numAlphaTemplateImages[i].channels() > 1) {
//                    cvtColor(numAlphaTemplateImages[i], numAlphaTemplateImages[i], COLOR_BGR2GRAY);  // תΪ�Ҷ�ͼ
//                }
//                numAlphaTemplateImages[i].convertTo(numAlphaTemplateImages[i], CV_32F);  // ת��Ϊ32λ������
//
//                // ����ģ��ƥ��
//                Mat resultImg;
//                matchTemplate(charMat, numAlphaTemplateImages[i], resultImg, TM_CCOEFF_NORMED);
//                double minVal, maxValTemp;
//                Point minLoc, maxLoc;
//                minMaxLoc(resultImg, &minVal, &maxValTemp, &minLoc, &maxLoc);
//
//                // ����һ����ֵ������ƥ��ȹ��͵����
//                double threshold = 0.8;  // ����Ը���ʵ�����������ֵ
//                if (maxValTemp > threshold && maxValTemp > maxVal) {
//                    maxVal = maxValTemp;
//                    bestMatchIdx = (int)i;
//                    bestMatchLabel = numAlphaLabels[bestMatchIdx];
//                }
//            }
//        }
//
//        // 5. ��ƥ�䵽���ַ���ǩ��ӵ������
//        if (!bestMatchLabel.empty()) {
//            result += bestMatchLabel;
//        }
//        else {
//            result += "?";  // ���û��ƥ�䵽�κ��ַ�����"?"��ʾ
//        }
//    }
//
//    return result;
//}
