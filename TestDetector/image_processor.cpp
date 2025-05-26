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

// camera.cpp�� countPixelsAboveThreshold �Լ� �״�� �̽�
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

// Ÿ���� ������ ũ�� ������ �����ϴ� �Լ� (camera.cpp���� �̽�)
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
std::vector<cv::Vec3f> circles; // ����� ���� �������� ����

bool detectCirclesAndEllipses(const cv::Mat& image, cv::Point& center, int radius, int squareSize, Modoo_cfg modoo_cfg) {
    bool isShow = false;
    cv::Canny(image, edges, 50, 150);

    // �� ���� (HoughCircles)
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

// camera.cpp�� detectColorRegion �Լ� �״�� �̽� (���� üũ ���� ����)
bool detectColorRegion(const Mat& inputImage, Modoo_cfg* modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& center, int& radius, Point& secondCenter, int& secondRadius) {
    //2025.03.10
    //��ȫ ��Ŀ ���� ó�� ����
    bool doublecheckYN = false;
    int firstYN = false;
    //tag �� �ݵ��� Ȯ�� �ʿ�

    if (productNumColor == 5) {
        cout << "double check Color : " << productNumColor << endl;
        doublecheckYN = true;
    }
    else
    {
        cout << "None double check Color : " << productNumColor << endl;
    }

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


            if (doublecheckYN) {
                if (firstYN) {
                    found = true;
                    secondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    secondRadius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                    cout << "�ι�° �� ���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                    // ���� �����Ǿ����� Ȯ��
                    cout << "�ι�° �� ���� color : " << targetHsv << endl;
                }
                else
                {
                    center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                    cout << "ù�� ° �� ���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                    // ���� �����Ǿ����� Ȯ��
                    cout << "ù�� ° �� ���� color : " << targetHsv << endl;

                    firstYN = true;
                }
            }
            else
            {
                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                cout << "���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                // ���� �����Ǿ����� Ȯ��
                cout << "���� color : " << targetHsv << endl;

                found = true;
                break; // ù ��° ������ ������ ���� ����
            }
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
    bool isShow = false;
    // 6. ������ ������ ������ �̿��� ������ ���
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "���� ũ�� : " + std::to_string(area) << endl;

        if (area >= modoo_cfg.blackMinMarkArea && area < modoo_cfg.blackMaxMarkArea) { // �ּ� ���� ����
            // ���Ʈ�� ����Ͽ� �߽����� ������ ��ȯ
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���

            // ���� ������ �� �簢�� ũ�� ����
            int squareSize = radius + 15; // ũ���� �簢���� ũ��

            bool isCircleDetected = detectCirclesAndEllipses(thresholdedImage, center, radius, squareSize, modoo_cfg);

            if (isCircleDetected) {
                std::cout << "Circle detected!" << std::endl;
                if (isShow) {
                    // �߰� ���� �� ��� ǥ��
                    imshow("Gray Image", blueChannel);
                    imshow("median Image", medianImage);
                    imshow("Inverted Image", invertedImage);
                    imshow("Thresholded Image", thresholdedImage);
                    waitKey(0); // Ű �Է� ���

                    destroyAllWindows(); // ��� â �ݱ�         
                }

                return true;
            }
            else {
                std::cout << "No circle detected." << std::endl;
            }
        }
    }

    if (isShow) {
        // �߰� ���� �� ��� ǥ��
        imshow("Gray Image", blueChannel);
        imshow("median Image", medianImage);
        imshow("Inverted Image", invertedImage);
        imshow("Thresholded Image", thresholdedImage);
        waitKey(0); // Ű �Է� ���

        destroyAllWindows(); // ��� â �ݱ�         
    }

    return false;
}

// camera.cpp�� detectColorRegion3rd �Լ� �״�� �̽�
bool detectColorRegion3rd(const Mat& inputImage, Modoo_cfg modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& center, int& radius, int& markAreaSum, Point& secondCenter, int& secondRadius) {
    //2025.03.10
    //��ȫ ��Ŀ ���� ó�� ����
    bool doublecheckYN = false;
    int firstYN = false;
    //tag �� �ݵ�� Ȯ�� �ʿ�
    if (productNumColor == 5) {
        cout << "double check Color : " << productNumColor << endl;
        doublecheckYN = true;
    }
    else
    {
        cout << "None double check Color : " << productNumColor << endl;
    }

    // 1. BGR �̹����� HSV�� ��ȯ
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, COLOR_BGR2HSV);

    // 2. ���� ���� ����
    Scalar lowerTargetColor, upperTargetColor;
    setColorRange(&modoo_cfg, lowerTargetColor, upperTargetColor, static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]));

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

    bool found = false;
    // 6. ������ ������ ������ �̿��� ������ ���
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area >= modoo_cfg.minMarkArea && area < modoo_cfg.maxMarkArea) { // �ּ� ���� ����
            // ���Ʈ�� ����Ͽ� �߽����� ������ ��ȯ
            Moments m = moments(contour);
            if (doublecheckYN) {
                if (firstYN) {
                    secondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    secondRadius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                    cout << "�ι�° ���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;
                    found = true;
                    // ���� �����Ǿ����� Ȯ��
                    cout << "�ι�° ���� color : " << targetHsv << endl;
                    break; // ù ��° ������ ������ ���� ����
                }
                else
                {
                    center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                    markAreaSum = area;
                    cout << "ù��° ���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;
                    // ���� �����Ǿ����� Ȯ��
                    cout << "ù��° ���� color : " << targetHsv << endl;
                    firstYN = true;
                }
            }
            else
            {
                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���
                markAreaSum = area;
                cout << "���� ũ�� : " + std::to_string(area) << "������ : " + std::to_string(radius) << "��ġ : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;
                found = true;
                // ���� �����Ǿ����� Ȯ��
                cout << "���� color : " << targetHsv << endl;
                break; // ù ��° ������ ������ ���� ����
            }
        }
    }

    if (!found) {
        markAreaSum = 0;
    }

    return found;
}

// camera.cpp�� detectBlackArea3rd �Լ� �״�� �̽�
bool detectBlackArea3rd(const Mat& inputImage, int buffer, Point& center, int& radius, Modoo_cfg modoo_cfg) {
    // BGR ä�� �и�
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R ������ �и���

    // B ä���� bgrChannels[0]�� �����
    cv::Mat blueChannel = bgrChannels[0];

    cv::Mat dst;
    blueChannel.convertTo(dst, -1, 2.0, -175); // dst = src * 2 - 175

    // 3. ��� �̹��� ����
    Mat invertedImage;
    bitwise_not(dst, invertedImage);

    // 4. ���� thresholding
    Mat thresholdedImage;
    double threshValue = 240; // �Ӱ谪 ����
    threshold(invertedImage, thresholdedImage, threshValue, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresholdedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (false) {
        // �߰� ���� �� ��� ǥ��
        imshow("Gray Image", blueChannel);
        imshow("gamma", dst);
        imshow("Inverted Image", invertedImage);
        imshow("Thresholded Image", thresholdedImage);
        waitKey(0); // Ű �Է� ���

        destroyAllWindows(); // ��� â �ݱ�
    }

    bool isdetected = false;
    // 6. ������ ������ ������ �̿��� ������ ���
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "���� ũ�� : " + std::to_string(area) << endl;

        if (area >= modoo_cfg.blackMinMarkArea && area < modoo_cfg.blackMaxMarkArea) { // �ּ� ���� ����
            // ���Ʈ�� ����Ͽ� �߽����� ������ ��ȯ
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // ������ �̿��� ������ ���

            // ���� ������ �� �簢�� ũ�� ����
            int squareSize = radius + 15; // ũ���� �簢���� ũ��

            bool isCircleDetected = detectCirclesAndEllipses(thresholdedImage, center, radius, squareSize, modoo_cfg);

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

// camera.cpp�� CaptureImage ������ �״�� �̽��� �Լ�
// camera.cpp�� CaptureImage ������ �״�� �̽��� �Լ� (���� ����)
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

        // �̹����� ����ִ��� Ȯ��
        if (ori.empty()) {
            printf("    [ERROR] Original image is empty\n");
            free_image_data(image_data);
            return 0;
        }

        int leftTop = 300;
        int rightBot = 1700;
        double totalArea = 0.0;
        int totalCount = 0;

        // camera.cpp�� �׷��̽����� ��ȯ �κ� ����
        Mat grayImage;
        if (cfg->hsvEnable == 1) {
            printf("hsvEnable : True\n");
            // HSV ����� ���� �ϴ� �׷��̽����Ϸ� ��ȯ (���� ������ ��ġ��Ű�� ����)
            //cvtColor(ori, grayImage, COLOR_BGR2GRAY);
            cvtColor(ori, grayImage, COLOR_RGB2GRAY);
        }
        else if (cfg->hsvEnable == 0) {
            //cvtColor(ori, grayImage, COLOR_BGR2GRAY);
            cvtColor(ori, grayImage, COLOR_RGB2GRAY);
            printf("hsvEnable : False\n");
        }
        else {
            printf("error-hsvEnable, using default grayscale conversion\n");
            cvtColor(ori, grayImage, COLOR_RGB2GRAY);
            //cvtColor(ori, grayImage, COLOR_BGR2GRAY);
        }

        // �׷��̽����� �̹����� ����� �����Ǿ����� Ȯ��
        if (grayImage.empty()) {
            printf("    [ERROR] Grayscale conversion failed\n");
            free_image_data(image_data);
            return 0;
        }

        printf("    [DEBUG] Grayscale image created: %dx%d\n", grayImage.cols, grayImage.rows);

        // camera.cpp�� ����ȭ �κ� �״��
        Mat binaryImage;
        threshold(grayImage, binaryImage, cfg->binaryValue, 255, THRESH_BINARY);

        // ����ȭ �̹��� Ȯ��
        if (binaryImage.empty()) {
            printf("    [ERROR] Binary threshold failed\n");
            free_image_data(image_data);
            return 0;
        }

        printf("    [DEBUG] Binary image created: %dx%d\n", binaryImage.cols, binaryImage.rows);

        Mat median;
        medianBlur(binaryImage, median, 21);

        //imshow("grayImage", grayImage);
        //imshow("binaryImage", binaryImage);
        //imshow("median", median);
        //cv::waitKey(0);

        // �̵�� �� ��� Ȯ��
        if (median.empty()) {
            printf("    [ERROR] Median blur failed\n");
            free_image_data(image_data);
            return 0;
        }

        printf("    [DEBUG] Median blur completed: %dx%d\n", median.cols, median.rows);

        // camera.cpp�� ������ ã�� �κ� �״��
        vector<vector<Point>> contours;
        findContours(median, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        printf("    [DEBUG] Found %zu contours\n", contours.size());

        result->total_detections = 0;
        result->success_count = 0;



        for (size_t i = 0; i < contours.size(); ++i) {
            bool isStickerOn = false;
            bool isColorStickerOn = false;

            

            // contour�� ���δ� �ּ� �簢�� ���
            Rect boundingRect = cv::boundingRect(contours[i]);

            // �ٿ�� �ڽ��� �̹��� ������ ����� �ʵ��� Ȯ��
            boundingRect &= Rect(0, 0, ori.cols, ori.rows);
            if (boundingRect.width <= 0 || boundingRect.height <= 0) {
                continue;
            }

            // contour�� ���� ���
            double area = contourArea(contours[i]);

            

            // ������ ���ڿ��� ��ȯ
            std::stringstream areaText;
            areaText << "Area: " << area;

            // ������ ���� ���� �ٸ� �������� �ؽ�Ʈ ǥ��
            Scalar textColor;
            totalArea = totalArea + area;

            if (area >= cfg->MinContourArea && area <= cfg->MaxContourArea) {
                printf("    [DEBUG] Found %zu contours area\n", area);
                
                totalCount++;
                result->total_detections++;

                textColor = Scalar(0, 255, 0);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "OK." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }

                // ���Ʈ�� ����Ͽ� contour�� �߽��� ���
                Moments m = moments(contours[i]);
                if (m.m00 == 0) continue; // ������ 0�� ��� �ǳʶٱ�

                Point center(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

                // �߽����� ���� �׸���
                circle(resultImage, center, 5, Scalar(0, 0, 255), -1); // ������ �� �׸���

                double resultX = cfg->origin_robot_y + (center.x - cfg->origin_vision_x) * cfg->res_x;
                double resultY = cfg->origin_robot_x + (center.y - cfg->origin_vision_y) * cfg->res_y;

                if (resultX < 0 || resultY < 0) {
                    putText(resultImage, "ordi minus", Point(boundingRect.tl().x, boundingRect.tl().y + 30), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    continue;
                }

                if (center.x < leftTop || center.x > rightBot || center.y < leftTop || center.y > rightBot) {
                    putText(resultImage, "ordi ng", Point(boundingRect.tl().x, boundingRect.tl().y + 30), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    continue;
                }

                // ���������κ��� ����ũ ����
                cv::Mat mask = cv::Mat::zeros(median.size(), CV_8UC1);
                cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

                // ������ ������ �������� �б� (erode ���)
                int erosionSize = 15; // �� ���� �����Ͽ� ũ�� ���� ����
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                    cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),
                    cv::Point(erosionSize, erosionSize));
                cv::erode(mask, mask, element);

                // ���� �÷� �̹������� ����ũ�� ����Ͽ� ũ��
                cv::Mat croppedImage;
                ori.copyTo(croppedImage, mask); // ����ũ�� ���� ���õ� ������ ����

                // ���������� ����� ������� �����ϱ�
                cv::Mat whiteCroppedImage(median.size(), ori.type(), cv::Scalar(255, 255, 255)); // ��� ��� ����
                croppedImage.copyTo(whiteCroppedImage, mask); // ����ũ�� ���� ���õ� ������ ����

                croppedImage = croppedImage(boundingRect);
                whiteCroppedImage = whiteCroppedImage(boundingRect);

                // ũ�ӵ� �̹����� ������� ������ Ȯ��
                if (croppedImage.empty() || whiteCroppedImage.empty()) {
                    printf("    [WARNING] Cropped image is empty for contour %zu\n", i);
                    continue;
                }

                // ���� ���� �� ������ ��Ŀ ���� (camera.cpp�� ������ ����)
                int buffer = 10;
                Point cropCenter;
                int radius;
                Point secondCropCenter;
                int secondRadius = 0;
                Scalar OnStickerColor;
                cv::Point cvtCenter;
                int colorIndex = cfg->ProductNum - 1;

                // colorIndex ���� Ȯ��
                if (colorIndex < 0 || colorIndex >= 20) {
                    colorIndex = 0; // �⺻������ ����
                }



                // ���� ��ǰ�� �ش��ϴ� ���� ID Ȯ��
                int target_color_id = cfg->productNumColor[colorIndex];
                printf("    [DEBUG] ProductNum: %d, colorIndex: %d, target_color_id: %d\n",
                    cfg->ProductNum, colorIndex, target_color_id);


                double angle = 0.0;
                bool color_detected = false;

                // ���� ���� �õ�
                for (int j = 0; j < cfg->maxColorCount; j++) {
                    HsvColor hsvColor = cfg->hsvColors[j];
                    


                    // �ٽ�: hsvColor.id�� target_color_id�� ��ġ�ϴ��� Ȯ��
                    if (hsvColor.id != target_color_id) {
                        printf("    [DEBUG] Skipping HsvColor[%d] id=%d (target=%d)\n",
                            j, hsvColor.id, target_color_id);
                        continue;
                    }

                    cout << "HsvColor �˻� ���� ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                    cout << "�˻� ���� productNum : " << colorIndex + 1 << ", productNumColor : " << cfg->productNumColor[colorIndex] << endl;

                    // ��ȿ�� HSV ������ Ȯ��
                    if (hsvColor.h <= 0 || hsvColor.h > 180 || hsvColor.s <= 0 || hsvColor.s > 255 || hsvColor.v <= 0 || hsvColor.v > 255) {
                        cout << "HsvColor �� �̻� : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << endl;
                        continue;  // HSV ���� ��ȿ���� ������ �������� �Ѿ
                    }

                    // HSV ���� Scalar�� ��ȯ
                    Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                    int productNumColor = cfg->productNumColor[colorIndex];

                    // ���� ����
                    bool detected = detectColorRegion(croppedImage, cfg, productNumColor, targetHsv, cropCenter, radius, secondCropCenter, secondRadius);

                    if (detected) {
                        cout << "�ش� ���� ��ƼĿ �����Ǿ����ϴ�." << endl;
                        cout << "�߽�: (" << cropCenter.x << ", " << cropCenter.y << "), ������: " << radius << endl;
                        printf("    [SUCCESS] Target color detected! HSV(%d,%d,%d) ID=%d\n",
                            hsvColor.h, hsvColor.s, hsvColor.v, hsvColor.id);
                        printf("    [SUCCESS] Center: (%d, %d), Radius: %d\n",
                            cropCenter.x, cropCenter.y, radius);
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);

                        // HSV�� BGR�� ��ȯ
                        Mat hsvColorMat(1, 1, CV_8UC3, targetHsv);  // 1x1 ũ���� Mat ���� �� HSV �� �Ҵ�
                        Mat bgrColorMat;
                        cvtColor(hsvColorMat, bgrColorMat, COLOR_HSV2BGR);  // HSV -> BGR ��ȯ

                        isStickerOn = true;
                        color_detected = true;


                        // BGR ���� ����
                        Vec3b bgr = bgrColorMat.at<Vec3b>(0, 0);
                        OnStickerColor = Scalar(bgr[0], bgr[1], bgr[2]);

                        break;
                    }
                    else
                    {
                        printf("    [DEBUG] Color detection failed for HSV(%d,%d,%d) ID=%d\n",
                            hsvColor.h, hsvColor.s, hsvColor.v, hsvColor.id);
                    }
                }

                // ��Ī�Ǵ� ������ ���� ��� �α� ���
                if (!color_detected) {
                    printf("    [INFO] No matching color found for target_color_id=%d\n", target_color_id);

                    // ��� ������ ��� ���� ID ��� (������)
                    printf("    [DEBUG] Available color IDs: ");
                    for (int j = 0; j < cfg->maxColorCount; j++) {
                        printf("%d ", cfg->hsvColors[j].id);
                    }
                    printf("\n");
                }

                // ������ ��Ŀ ���� �õ�
                if (!isStickerOn && cfg->productNumColor[colorIndex] == cfg->BlackTagNum) {
                    bool blackDetected = detectBlackArea(whiteCroppedImage, buffer, cropCenter, radius, *cfg);
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
                        cout << "���� ��ƼĿ �� ����" << endl;
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

                    //2025.03.10
                    //�ι�° ��Ŀ ǥ��
                    if (secondRadius != 0) {
                        Point adjustedSecondCenter = Point(secondCropCenter.x + boundingRect.tl().x, secondCropCenter.y + boundingRect.tl().y);
                        if (secondRadius + 8 >= 0) {
                            circle(resultImage, adjustedSecondCenter, secondRadius + 8, OnStickerColor, 3); // ������ �������� �� �׸���
                        }
                        else
                        {
                            circle(resultImage, adjustedSecondCenter, 20, OnStickerColor, 3); // ������ �������� �� �׸���
                        }
                    }

                    // ��� �̹��� ǥ��
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 30);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker On", tmp, FONT_HERSHEY_SIMPLEX, 1.5, OnStickerColor, 3);
                    }

                    // ���� ��� ����
                    DetectionResult* det = &result->detections[result->success_count];
                    det->detected = 1;
                    det->center_x = cvtCenter.x;
                    det->center_y = cvtCenter.y;
                    det->radius = radius;
                    det->area = static_cast<int>(area);
                    strcpy_s(det->result_msg, sizeof(det->result_msg), "OK");
                    strcpy_s(det->color_info, sizeof(det->color_info), isStickerOn ? "Color" : "Black");

                    result->success_count++;

                    // ��ǥ���� ���ڿ��� ��ȯ
                    std::ostringstream ossX;
                    ossX << std::fixed << std::setprecision(3) << std::setw(7) << std::setfill('0') << resultX;
                    std::string strResultX = ossX.str();

                    std::ostringstream ossY;
                    ossY << std::fixed << std::setprecision(3) << std::setw(7) << std::setfill('0') << resultY;
                    std::string strResultY = ossY.str();

                    // ��ǥ���� ���ڿ��� ��ȯ
                    std::string coordinates = "(" + strResultY + ", " + strResultX + ")";

                    // �߽��� ��ǥ�� �̹����� �ؽ�Ʈ�� ǥ��
                    putText(resultImage, coordinates, Point(center.x + 10, center.y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3);

                    // ���� ��� �� ǥ�� (camera.cpp�� calculateAngle �Լ� ����)
                    double dx = cvtCenter.x - center.x;
                    double dy = cvtCenter.y - center.y;
                    double angleRad = std::atan2(dy, dx);
                    double angleDeg = angleRad * (180.0 / CV_PI);
                    if (angleDeg < 0) {
                        angleDeg += 360.0;
                    }

                    std::ostringstream angleStream;
                    angleStream << std::fixed << std::setprecision(3) << std::setw(7) << std::setfill('0') << angleDeg;
                    std::string formattedAngle = angleStream.str();

                    putText(resultImage, formattedAngle, Point(center.x + 10, center.y + 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 0), 3);

                    isStickerOn = false;
                }
                else {
                    cout << "�ش� ���� ������ �������� �ʾҽ��ϴ�." << endl;
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 35);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker NG", tmp, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    }
                    continue;
                }

                // ROI�� �ѷ��� �簢���� �̹����� �׸���
                rectangle(resultImage, boundingRect, textColor, 2);
            }
            else if (area >= cfg->MinContourArea / 3 && area <= cfg->MaxContourArea * 3) {
                totalCount++;
                textColor = Scalar(0, 0, 255);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "NG." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }
            }
            else {
                printf("    [DEBUG] %zu contours area is not collect min : %zu , max : %zu \n", area , cfg->MinContourArea , cfg->MaxContourArea);
            }
        }

        // ���� �簢�� �׸��� (camera.cpp�� drawDashedRectangle ���� ����ȭ)
        rectangle(resultImage, Point(leftTop, leftTop), Point(rightBot, rightBot), Scalar(255, 0, 0), 2);

        // ��� �̹��� ����
        save_result_image_opencv(filepath, result->result_filename, result);

        free_image_data(image_data);

        printf("    [DEBUG] ProcessImageFromFile_Mode1 completed successfully\n");
        printf("    [DEBUG] Total detections: %d, Success count: %d\n", result->total_detections, result->success_count);

        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in ProcessImageFromFile_Mode1: " << e.what() << endl;
        if (image_data) free_image_data(image_data);
        return 0;
    }
}

// ��� 2 (3rd detection mode) - camera.cpp�� CaptureImage3rd ���� �̽�
extern "C" int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    void* image_data = nullptr;

    try {
        // �̹��� �ε�
        if (!load_image_opencv(filepath, &image_data)) {
            printf("    [ERROR] Failed to load image\n");
            return 0;
        }

        ImageData* data = static_cast<ImageData*>(image_data);
        Mat& ori = data->original;
        Mat& resultMat = data->result;

        printf("    [DEBUG] Processing Mode 2 (3rd detection): %dx%d\n", ori.cols, ori.rows);

        // camera.cpp�� CaptureImage3rd���� ����ϴ� ũ�� ���� ����
        int buffer = 25;
        int buffer_y = 45;
        int comSize = 270; // 10M 25mm�϶� 187
        int startPointX[2] = { 83, 1585 };
        int startPointY[2] = { 140 - buffer_y, 1600 - buffer_y };
        buffer_y = buffer_y + buffer_y;

        // ũ���� �������� ������ ���� (20���� Rect)
        vector<Rect> cropRegions = {
            Rect(startPointX[0] - buffer, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 1, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 2, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 3, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 4, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),

            Rect(startPointX[1] - buffer, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 1, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 2, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 3, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 4, startPointY[0] - buffer, comSize + buffer, comSize + buffer + buffer_y),

            Rect(startPointX[0] - buffer, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 1, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 2, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 3, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[0] - buffer + comSize * 4, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),

            Rect(startPointX[1] - buffer, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 1, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 2, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 3, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
            Rect(startPointX[1] - buffer + comSize * 4, startPointY[1] - buffer, comSize + buffer, comSize + buffer + buffer_y),
        };

        // ũ�ӵ� �̹����� ������ ����
        vector<Mat> croppedImages;
        string detectedResult[20];

        result->total_detections = 20;
        result->success_count = 0;

        // ũ�� ������ ���� �̹��� ũ�� �� ������ ��� �̹����� �׸���
        for (int i = 0; i < cropRegions.size(); ++i) {
            cfg->ProductNum = i;
            // �̹������� �ش� ������ ũ���Ͽ� Mat�� ����
            Mat cropped = ori(cropRegions[i]);

            // ���� ���� �� ��� Ȯ��
            int buffer = 5; // ���� ũ�� ����
            Point cropCenter;
            int radius;
            Point secondCropCenter;
            int secondRadius = 0;
            Scalar OnStickerColor;
            cv::Point cvtCenter;

            bool isColorMarkDetected = false;
            int MarkAreaSum = 0;
            int MarkAreaNow;
            // cfg->ProductNum 1~20 ��
            detectedResult[i] = "NG";

            // �� ����(i)�� ���� �ش��ϴ� target_color_id Ȯ��
            int target_color_id = cfg->productNumColor[i]; // i��° ������ ���� ID
            printf("    [DEBUG] Area[%d]: target_color_id=%d\n", i, target_color_id);

            for (int j = 0; j < cfg->maxColorCount; j++) { // �÷� ��ĵ

                HsvColor hsvColor = cfg->hsvColors[j];
                if (cfg->productNumColor[i] != hsvColor.id) {
                    continue;
                }
                cout << "HsvColor �˻� ���� ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                cout << "�˻� ���� productNum : " << i + 1 << ", productNumColor : " << cfg->productNumColor[i] << endl;

                // ��ȿ�� HSV ������ Ȯ�� (�� ���ÿ����� ��� h, s, v ���� 0�� �ƴϸ� ��ȿ�ϴٰ� ����)
                if (hsvColor.h <= 0 || hsvColor.h > 180 || hsvColor.s <= 0 || hsvColor.s > 255 || hsvColor.v <= 0 || hsvColor.v > 255) {
                    cout << "HsvColor �� �̻� : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << endl;
                    continue;  // HSV ���� ��ȿ���� ������ �������� �Ѿ
                }

                // HSV ���� Scalar�� ��ȯ
                Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                int productNumColor = cfg->productNumColor[i];

                // ���� ����
                bool detected = detectColorRegion3rd(cropped, *cfg, productNumColor, targetHsv, cropCenter, radius, MarkAreaNow, secondCropCenter, secondRadius);

                if (detected) {
                    cout << "�ش� ���� ��ƼĿ �����Ǿ����ϴ�." << endl;
                    cout << "�߽�: (" << cropCenter.x << ", " << cropCenter.y << "), ������: " << radius << endl;
                    MarkAreaSum = MarkAreaNow + MarkAreaSum;

                    // HSV�� BGR�� ��ȯ
                    Mat hsvMat(1, 1, CV_8UC3, targetHsv);  // 1x1 ũ���� Mat ���� �� HSV �� �Ҵ�
                    Mat bgrMat;
                    cvtColor(hsvMat, bgrMat, COLOR_HSV2BGR);  // HSV -> BGR ��ȯ

                    // BGR ���� ����
                    Vec3b bgr = bgrMat.at<Vec3b>(0, 0);
                    OnStickerColor = Scalar(bgr[0], bgr[1], bgr[2]);

                    // ������ �������� �� �׸���
                    circle(cropped, cropCenter, radius + 8, OnStickerColor, 3);

                    if (secondRadius != 0) {
                        circle(cropped, secondCropCenter, secondRadius + 8, OnStickerColor, 3);
                    }

                    detectedResult[i] = "OK";
                    isColorMarkDetected = true;
                    break;
                }
            }

            // �÷� ���� ���� �ջ� ��� �ڵ�
            if (MarkAreaSum >= cfg->minSUMMarkArea && MarkAreaSum < cfg->maxSUMMarkArea) {
                isColorMarkDetected = true;
                detectedResult[i] = "OK";
                cout << "min:" << cfg->minSUMMarkArea << ", max : " << cfg->maxSUMMarkArea << endl;
                cout << "***************** Mark Area Sum : " << MarkAreaSum << "*******************" << endl;
            }
            else {
                cout << "-------- Mark Area Sum : " << MarkAreaSum << "--------" << endl;
            }

            // �� ��Ŀ ���� �ڵ�
            if (cfg->productNumColor[i] == 7) { // 7�� ��
                bool isBlackMarkDetected = detectBlackArea3rd(cropped, buffer, cropCenter, radius, *cfg);
                if (isBlackMarkDetected) {
                    isColorMarkDetected = true;
                    detectedResult[i] = "OK";
                    cout << "Black Marker ���� �Ϸ� " << MarkAreaSum << "" << endl;
                    circle(cropped, cropCenter, radius + 8, Scalar(255, 255, 255), 3);
                }
                else {
                    detectedResult[i] = "BK.NG";
                    cout << "Black Marker ���� �ȵ� " << MarkAreaSum << "" << endl;
                }
            }

            // �˻� ��� OFF ����
            if (cfg->isDetect[i] == 0) {
                detectedResult[i] = "OFF";
                cout << "�˻� ���� ����" << endl;
            }

            // ���Ϳ� ũ�ӵ� �̹��� ����
            croppedImages.push_back(cropped);

            // ũ���� ������ ���� �̹����� �׸��� (�Ķ��� �簢��, �β� 2)
            rectangle(resultMat, cropRegions[i], Scalar(255, 0, 0), 2);

            // ���� ��� ����
            DetectionResult* det = &result->detections[i];
            det->detected = (detectedResult[i] == "OK") ? 1 : 0;
            det->center_x = cropCenter.x + cropRegions[i].x;
            det->center_y = cropCenter.y + cropRegions[i].y;
            det->radius = radius;
            det->area = MarkAreaSum;
            strcpy_s(det->result_msg, sizeof(det->result_msg), detectedResult[i].c_str());
            strcpy_s(det->color_info, sizeof(det->color_info), "Color");

            if (det->detected) {
                result->success_count++;
            }
        }

        // ��� �̹��� ����
        save_result_image_opencv(filepath, result->result_filename, result);

        free_image_data(image_data);
        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in ProcessImageFromFile_Mode2: " << e.what() << endl;
        if (image_data) free_image_data(image_data);
        return 0;
    }
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
        else if (result->mode == 2) {
            // ��� 2: 20�� ���� ���� ��� ǥ��
            for (int i = 0; i < 20; i++) {
                DetectionResult* det = &result->detections[i];

                Scalar color;
                if (strcmp(det->result_msg, "OK") == 0) {
                    color = Scalar(0, 255, 0); // �ʷϻ�
                }
                else if (strcmp(det->result_msg, "OFF") == 0) {
                    color = Scalar(0, 255, 255); // �����
                }
                else {
                    color = Scalar(0, 0, 255); // ������
                }

                // ���� ��ȣ�� ��� �ؽ�Ʈ ǥ��
                string label = string(det->result_msg) + to_string(i + 1);
                Point textPos(det->center_x - 20, det->center_y - 20);
                putText(resultImage, label, textPos, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

                if (det->detected) {
                    circle(resultImage, Point(det->center_x, det->center_y), det->radius + 8, color, 3);
                }
            }
        }

        // ó�� �ð� ǥ��
        string timeInfo = "Time: " + to_string(result->processing_time) + "s";
        putText(resultImage, timeInfo, Point(10, resultImage.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // ������ ǥ��
        string successInfo = "Success: " + to_string(result->success_count) + "/" + to_string(result->total_detections);
        putText(resultImage, successInfo, Point(10, resultImage.rows - 50),
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

// ��Ÿ �Լ����� ������� ���� (���� ����)
extern "C" int detect_color_region(void* image_data, Modoo_cfg* cfg, int product_color,
    int* center_x, int* center_y, int* radius, int* area) {
    return 0;
}

extern "C" int detect_black_area(void* image_data, Modoo_cfg* cfg,
    int* center_x, int* center_y, int* radius) {
    return 0;
}

extern "C" int convert_to_binary(void* image_data, int threshold) {
    return 0;
}

extern "C" int find_contours(void* image_data, Modoo_cfg* cfg, DetectionResult results[], int* count) {
    return 0;
}
