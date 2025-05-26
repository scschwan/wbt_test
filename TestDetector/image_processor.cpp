// image_processor.cpp - camera.cpp ���� �ڵ� �̽�
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// C �������̽��� ���� �����
extern "C" {
#include "image_processor.h"
#include "simple_types.h"
#include "image_result.h"
}

using namespace cv;
using namespace std;

// OpenCV Mat�� void*�� �����ϴ� ����ü
struct ImageData {
    Mat original;
    Mat result;
    Mat hsv;
    Mat binary;
    int width;
    int height;
};

// �̹��� �ε� �Լ�
extern "C" int load_image_opencv(const char* filepath, void** image_data) {
    try {
        ImageData* data = new ImageData();

        // �̹��� �ε�
        data->original = imread(filepath, IMREAD_COLOR);
        if (data->original.empty()) {
            delete data;
            return 0;
        }

        data->width = data->original.cols;
        data->height = data->original.rows;
        data->result = data->original.clone();

        printf("    [DEBUG] Image loaded: %dx%d\n", data->width, data->height);

        *image_data = static_cast<void*>(data);
        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in load_image_opencv: " << e.what() << endl;
        return 0;
    }
}

// camera.cpp�� setColorRange �Լ� �״�� �̽�
void setColorRange(Modoo_cfg* modoo_cfg, Scalar& lowerColor, Scalar& upperColor, int hue, int saturation, int value) {
    int hueBuffer = modoo_cfg->HsvBufferH;
    int satBuffer = modoo_cfg->HsvBufferS;
    int valBuffer = modoo_cfg->HsvBufferV;
    lowerColor = Scalar(max(0, hue - hueBuffer), max(0, saturation - satBuffer), max(0, value - valBuffer));
    upperColor = Scalar(min(180, hue + hueBuffer), min(255, saturation + satBuffer), min(255, value + valBuffer));
    cout << "set Color Range Lower: " << lowerColor << endl;
    cout << "set Color Range Upper: " << upperColor << endl;
}

// camera.cpp�� detectColorRegion �Լ� �״�� �̽� (�ܼ�ȭ ����)
bool detectColorRegion(const Mat& inputImage, Modoo_cfg* modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& center, int& radius) {
    cout << "None double check Color : " << productNumColor << endl;

    // 1. BGR �̹����� HSV�� ��ȯ
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, COLOR_BGR2HSV);

    // 2. ���� ���� ����
    Scalar lowerTargetColor, upperTargetColor;
    setColorRange(modoo_cfg, lowerTargetColor, upperTargetColor, static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]));

    // 3. ����ũ ����
    Mat mask;
    inRange(hsvImage, lowerTargetColor, upperTargetColor, mask);

    // 5. ������ ã��
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = inputImage.clone(); // ���� �̹����� ����
    // ��� ������ ����� ���������� �׸���
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contourImage, contours, static_cast<int>(i), Scalar(0, 0, 255), 2); // ������, �β� 2
    }

    double areaMax = 0.0;
    bool found = false;
    // 6. ������ ������ ������ �̿��� ������ ���
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > areaMax) {
            areaMax = area;
        }
        if (area >= modoo_cfg->minMarkArea && area < modoo_cfg->maxMarkArea) { // �ּ� ���� ����
            // ���Ʈ�� ����Ͽ� �߽����� ������ ��ȯ
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
            cout << "���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

            // ���� �����Ǿ����� Ȯ��
            cout << "���� color : " << targetHsv << endl;

            found = true;
            break; // ù ��° ������ ������ ���� ����
        }
    }

    if (!found) {
        cout << "���� ����, �ִ� ���� ���� : " << areaMax << endl;
    }
    return found;
}

// camera.cpp�� detectBlackArea �Լ� �״�� �̽�
bool detectBlackArea(const Mat& inputImage, int buffer, Point& center, int& radius, Modoo_cfg modoo_cfg) {
    // BGR ä�� �и�
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R ������ �и���

    // B ä���� bgrChannels[0]�� �����
    cv::Mat blueChannel = bgrChannels[0];

    //cv::Mat dst;
    //blueChannel.convertTo(dst, -1, 2.0, -175); // dst = src * 2 - 175
     //2. ������ ó�� (Gaussian Blur)
    Mat medianImage;
    cv::medianBlur(blueChannel, medianImage, 7); // 5x5 Ŀ�� ���

    // 3. ��� �̹��� ����
    Mat invertedImage;
    bitwise_not(medianImage, invertedImage);

    int whitePixelCount = countPixelsAboveThreshold(invertedImage, 150);
    putText(invertedImage, "warea:" + std::to_string(whitePixelCount), Size(5, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    // 4. ���� thresholding
    Mat thresholdedImage;
    double threshValue = 180; // �Ӱ谪 ����
    threshold(invertedImage, thresholdedImage, threshValue, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresholdedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    bool isdetected = false;
    // 6. ������ ������ ������ �̿��� ������ ���
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "���� ũ�� : " + std::to_string(area) << endl;

        if (area >= modoo_cfg->blackMinMarkArea && area < modoo_cfg->blackMaxMarkArea) { // �ּ� ���� ����
            // ���Ʈ�� ����Ͽ� �߽����� ������ ��ȯ
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���

            // ���� ������ �� �簢�� ũ�� ����
            int squareSize = radius + 15; // ũ���� �簢���� ũ��

            //bool isCircleDetected = detectCircle(thresholdedImage, center, radius, squareSize);
            bool isCircleDetected = detectCirclesAndEllipses(thresholdedImage, center, radius, squareSize, *modoo_cfg);

            if (isCircleDetected) {
                std::cout << "Circle detected!" << std::endl;
               

                return true;
            }
            else {
                std::cout << "No circle detected." << std::endl;
            }
        }
    }

    return false;
}

// Ÿ���� ������ ũ�� ������ �����ϴ� �Լ�
bool isEllipseValid(const cv::RotatedRect& ellipse, float minAspectRatio, float maxAspectRatio, float minSize, float maxSize) {
    // �ʺ�� ������ ���� ���
    float aspectRatio = ellipse.size.width / ellipse.size.height;

    // Ÿ���� ũ�� Ȯ�� (Ÿ���� �ʺ�� ������ ������� ũ�� ����)
    float size = (ellipse.size.width + ellipse.size.height) / 2.0;
    cout << "ellipse size : " << to_string(size) << endl;
    // Ÿ���� �־��� ������ ũ�� ���� ���� �ִ��� Ȯ��
    return (aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio && size >= minSize && size <= maxSize);
}

cv::Mat edges; // ���� �̹����� �������� ����

bool detectCirclesAndEllipses(const cv::Mat& image, cv::Point& center, int radius, int squareSize, Modoo_cfg modoo_cfg) {
    bool isShow = false;
    cv::Canny(image, edges, 50, 150);

    // �� ���� (HoughCircles)
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 100, 70, 10, 8, 16);

    // ����� ���� BGR �̹��� ����
    cv::Mat result;
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);

    bool circleDetected = !circles.empty(); // HoughCircles���� ���� ����Ǿ����� Ȯ��

    // �� �ð�ȭ
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point circleCenter(static_cast<int>(circles[i][0]), static_cast<int>(circles[i][1]));
        int detectedRadius = static_cast<int>(circles[i][2]);

        // �� �׵θ� �׸���
        cv::circle(result, circleCenter, detectedRadius, cv::Scalar(0, 0, 255), 1);
        cout << "detected radius: " << detectedRadius << endl;
    }

    // ������ ���� (Contours)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool ellipseDetected = false;

    // �������� �̿��� Ÿ�� ����
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() >= 5) { // Ÿ���� ���߷��� �ּ� 5���� ����Ʈ �ʿ�
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);

            // Ÿ���� ���� �� ũ�� ����
            if (isEllipseValid(ellipse, 0.7, 1.2, modoo_cfg.BlackEllipseMinSize, modoo_cfg.BlackEllipseMaxSize)) {
                // Ÿ���� �߽ɰ� ������(�뷫���� ũ��)�� ����
                center = cv::Point(static_cast<int>(ellipse.center.x), static_cast<int>(ellipse.center.y));
                radius = static_cast<int>((ellipse.size.width + ellipse.size.height) / 4); // ��� ũ���� ������ ���������� ���

                // Ÿ�� �׸���
                cv::ellipse(result, ellipse, cv::Scalar(0, 255, 0), 2);
                ellipseDetected = true; // Ÿ���� �����Ǿ����� ǥ��

                cout << "Valid ellipse detected: center(" << ellipse.center.x << ", " << ellipse.center.y << "), "
                    << "size(" << ellipse.size.width << ", " << ellipse.size.height << "), "
                    << "aspect ratio: " << (ellipse.size.width / ellipse.size.height) << endl;
            }
            else {
                cout << "Invalid ellipse: center(" << ellipse.center.x << ", " << ellipse.center.y << "), "
                    << "size(" << ellipse.size.width << ", " << ellipse.size.height << "), "
                    << "aspect ratio: " << (ellipse.size.width / ellipse.size.height) << endl;
            }
        }
    }

    // ��� �̹��� ���
    if (isShow) {
        imshow("image", image);
        imshow("edges", edges);
        cv::imshow("Cropped Image with Detected Circles and Ellipses", result);
        cv::waitKey(0);
    }

    // ���̳� Ÿ���� �ϳ��� �����Ǹ� true ��ȯ
    return ellipseDetected;
}

int countPixelsAboveThreshold(const cv::Mat& inputImage, int threshold) {

    // �̹����� ����ִ��� Ȯ��
    if (inputImage.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return -1;
    }

    // BGR ä�� �и�
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R ������ �и���

    // B ä���� bgrChannels[0]�� �����
    cv::Mat blueChannel = bgrChannels[0];

    // �̹����� ���� ä��(�׷��̽�����)���� Ȯ��
    if (blueChannel.channels() != 1) {
        std::cerr << "Input image must be a grayscale image!" << std::endl;
        return -1;
    }

    // �ȼ� ���� �������� ī��Ʈ
    int count = 0;
    for (int y = 0; y < blueChannel.rows; y++) {
        for (int x = 0; x < blueChannel.cols; x++) {
            if (blueChannel.at<uchar>(y, x) > threshold) {
                count++;
            }
        }
    }

    return count;
}

// camera.cpp�� CaptureImage ������ �״�� �̽��� �Լ�
extern "C" int ProcessImageFromFile_Mode1(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    void* image_data = nullptr;

    try {
        // �̹��� �ε�
        if (!load_image_opencv(filepath, &image_data)) {
            printf("    [ERROR] Failed to load image\n");
            return 0;
        }

        ImageData* data = static_cast<ImageData*>(image_data);
        Mat& ori = data->original;
        Mat& resultImage = data->result;

        printf("    [DEBUG] Processing image: %dx%d\n", ori.cols, ori.rows);

        // camera.cpp�� �׷��̽����� ��ȯ �κ� �״��
        Mat grayImage;
        if (cfg->hsvEnable == 1) {
            printf("hsvEnable : True\n");
        }
        else if (cfg->hsvEnable == 0) {
            cvtColor(ori, grayImage, COLOR_BGR2GRAY);
            printf("hsvEnable : False\n");
        }
        else {
            printf("error-hsvEnable\n");
        }

        // camera.cpp�� ����ȭ �κ� �״��
        Mat binaryImage;
        threshold(grayImage, binaryImage, cfg->binaryValue, 255, THRESH_BINARY);

        Mat median;
        medianBlur(binaryImage, median, 21);

        // camera.cpp�� ������ ã�� �κ� �״��
        vector<vector<Point>> contours;
        findContours(median, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // �� contour�� ROI�� �����ϰ� ����
        size_t savedCount = 0;
        std::vector<cv::Mat> croppedImages;

        result->total_detections = 0;
        result->success_count = 0;

        for (size_t i = 0; i < contours.size(); ++i) {
            bool isStickerOn = false;
            bool isColorStickerOn = false;

            // contour�� ���δ� �ּ� �簢�� ���
            Rect boundingRect = cv::boundingRect(contours[i]);

            // contour�� ���� ���
            double area = contourArea(contours[i]);

            // ������ ���ڿ��� ��ȯ
            std::stringstream areaText;
            areaText << "Area: " << area;

            // ������ ���� ���� �ٸ� �������� �ؽ�Ʈ ǥ��
            Scalar textColor;

            if (area >= cfg->MinContourArea && area <= cfg->MaxContourArea) {
                result->total_detections++;

                textColor = Scalar(0, 255, 0);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "OK." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }

                // ���Ʈ�� ����Ͽ� contour�� �߽��� ���
                Moments m = moments(contours[i]);
                Point center(m.m10 / m.m00, m.m01 / m.m00);

                // �߽����� ���� �׸���
                circle(resultImage, center, 5, Scalar(0, 0, 255), -1); // ������ �� �׸���

                double resultX = cfg->origin_robot_y + (center.x - cfg->origin_vision_x) * cfg->res_x;
                double resultY = cfg->origin_robot_x + (center.y - cfg->origin_vision_y) * cfg->res_y;

                if (resultX < 0 || resultY < 0) {
                    putText(resultImage, "ordi minus", Size(boundingRect.tl().x, boundingRect.tl().y + 30), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    continue;
                }

                // ���� ���� �� ������ ��Ŀ ���� (camera.cpp�� ������ ����)
                int buffer = 10;
                Point cropCenter;
                int radius;
                Scalar OnStickerColor;
                cv::Point cvtCenter;
                int colorIndex = cfg->ProductNum - 1;

                // ���� ���� �õ�
                for (int j = 0; j < cfg->maxColorCount; j++) {
                    HsvColor hsvColor = cfg->hsvColors[j];
                    if (cfg->productNumColor[colorIndex] != hsvColor.id) {
                        continue;
                    }
                    cout << "HsvColor �˻� ���� ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                    cout << "�˻� ���� productNum : " << colorIndex + 1 << ", productNumColor : " << cfg->productNumColor[colorIndex] << endl;

                    // HSV ���� Scalar�� ��ȯ
                    Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                    int productNumColor = cfg->productNumColor[colorIndex];

                    // ���� ����
                    bool detected = detectColorRegion(ori, cfg, productNumColor, targetHsv, cropCenter, radius);

                    if (detected) {
                        cout << "�ش� ���� ��ƼĿ �����Ǿ����ϴ�." << endl;
                        cout << "�߽�: (" << cropCenter.x << ", " << cropCenter.y << "), ������: " << radius << endl;
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);

                        // HSV�� BGR�� ��ȯ
                        Mat hsvColor(1, 1, CV_8UC3, targetHsv);
                        Mat bgrColor;
                        cvtColor(hsvColor, bgrColor, COLOR_HSV2BGR);

                        isStickerOn = true;

                        // BGR ���� ����
                        Vec3b bgr = bgrColor.at<Vec3b>(0, 0);
                        OnStickerColor = Scalar(bgr[0], bgr[1], bgr[2]);

                        // ���� ��� ����
                        DetectionResult* det = &result->detections[result->success_count];
                        det->detected = 1;
                        det->center_x = cvtCenter.x;
                        det->center_y = cvtCenter.y;
                        det->radius = radius;
                        det->area = static_cast<int>(area);
                        strcpy_s(det->result_msg, sizeof(det->result_msg), "OK");
                        strcpy_s(det->color_info, sizeof(det->color_info), "Color");

                        result->success_count++;
                        break;
                    }
                }

                // ������ ��Ŀ ���� �õ�
                if (!isStickerOn && cfg->productNumColor[colorIndex] == cfg->BlackTagNum) {
                    bool blackDetected = detectBlackArea(ori, buffer, cropCenter, radius, cfg);
                    if (blackDetected) {
                        cout << "���� ��ƼĿ �����Ǿ����ϴ�. flag2" << endl;
                        cout << "�߽�: (" << cropCenter.x << ", " << cropCenter.y << "), ������: " << radius << endl;
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);
                        OnStickerColor = Scalar(255, 255, 255);
                        isStickerOn = true;

                        // ���� ��� ����
                        DetectionResult* det = &result->detections[result->success_count];
                        det->detected = 1;
                        det->center_x = cvtCenter.x;
                        det->center_y = cvtCenter.y;
                        det->radius = radius;
                        det->area = static_cast<int>(area);
                        strcpy_s(det->result_msg, sizeof(det->result_msg), "OK");
                        strcpy_s(det->color_info, sizeof(det->color_info), "Black");

                        result->success_count++;
                    }
                    else {
                        cout << "���� ��ƼĿ ���� �ȵ�" << endl;
                    }
                }

                if (isStickerOn) {
                    // ��ȯ�� BGR �������� �� �׸���
                    if (radius + 8 >= 0) {
                        circle(resultImage, cvtCenter, radius + 8, OnStickerColor, 3); // ������ �������� �� �׸���
                    }
                    else {
                        circle(resultImage, cvtCenter, 20, OnStickerColor, 3); // ������ �������� �� �׸���
                    }

                    // ��� �̹��� ǥ��
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 30);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker On", tmp, FONT_HERSHEY_SIMPLEX, 1.5, OnStickerColor, 3);
                    }
                }
                else {
                    cout << "�ش� ���� ������ �������� �ʾҽ��ϴ�." << endl;
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 35);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker NG", tmp, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    }
                    continue;
                }

                // ROI�� �ѷ��� �簢���� �̹����� �׸��� (�Ķ��� �簢��, �β� 2)
                rectangle(resultImage, boundingRect, textColor, 2);
            }
            else if (area >= cfg->MinContourArea / 3 && area <= cfg->MaxContourArea * 3) {
                textColor = Scalar(0, 0, 255);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "NG." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }
            }
        }

        // ��� �̹��� ����
        save_result_image_opencv(filepath, result->result_filename, result);

        free_image_data(image_data);
        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in ProcessImageFromFile_Mode1: " << e.what() << endl;
        if (image_data) free_image_data(image_data);
        return 0;
    }
}

// ��� 2�� �ϴ� �⺻ ���� ����
extern "C" int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    printf("    [INFO] Mode 2 processing not yet implemented with original camera.cpp logic\n");
    result->total_detections = 20;
    result->success_count = 0;
    return 1;
}

// ��� �̹��� ���� (���� ����)
extern "C" int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result) {
    try {
        // ���� �̹��� �ε�
        Mat originalImage = imread(input_path, IMREAD_COLOR);
        if (originalImage.empty()) {
            return 0;
        }

        Mat resultImage = originalImage.clone();

        if (result->mode == 1) {
            // ��� 1: ���� ���� ��� ǥ��
            for (int i = 0; i < result->success_count; i++) {
                DetectionResult* det = &result->detections[i];

                // �� �׸���
                Scalar color = (strcmp(det->result_msg, "OK") == 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                circle(resultImage, Point(det->center_x, det->center_y), det->radius + 8, color, 3);

                // �ؽ�Ʈ �� �߰�
                string label = string(det->result_msg) + " A:" + to_string(det->area);
                Point textPos(det->center_x + 10, det->center_y - 10);
                putText(resultImage, label, textPos, FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

                // �߽��� ǥ��
                circle(resultImage, Point(det->center_x, det->center_y), 3, Scalar(255, 255, 0), -1);
            }
        }

        // ó�� �ð� ǥ��
        string timeInfo = "Time: " + to_string(result->processing_time) + "s";
        putText(resultImage, timeInfo, Point(10, resultImage.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // �̹��� ����
        bool success = imwrite(output_path, resultImage);
        return success ? 1 : 0;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in save_result_image_opencv: " << e.what() << endl;
        return 0;
    }
}

// �޸� ����
extern "C" void free_image_data(void* image_data) {
    if (image_data) {
        ImageData* data = static_cast<ImageData*>(image_data);
        delete data;
    }
}

// ��Ÿ �Լ����� ������� ����
extern "C" int detect_color_region(void* image_data, Modoo_cfg* cfg, int product_color,
    int* center_x, int* center_y, int* radius, int* area) {
    return 0;
}
extern "C" int detect_black_area(void* image_data, Modoo_cfg* cfg,
    int* center_x, int* center_y, int* radius) {
    return 0;
}
extern "C" int convert_to_binary(void* image_data, int threshold) { return 0; }
extern "C" int find_contours(void* image_data, Modoo_cfg* cfg, DetectionResult results[], int* count) { return 0; }