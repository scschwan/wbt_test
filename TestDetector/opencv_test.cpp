// opencv_test.cpp - OpenCV 연결 테스트
#include <opencv2/opencv.hpp>
#include <iostream>

extern "C" {
    int test_opencv_basic();
    int test_image_load(const char* filepath);
}

using namespace cv;
using namespace std;

// 기본 OpenCV 테스트
extern "C" int test_opencv_basic() {
    try {
        cout << "OpenCV version: " << CV_VERSION << endl;

        // 간단한 Mat 생성 테스트
        Mat testImage = Mat::zeros(100, 100, CV_8UC3);
        testImage.setTo(Scalar(0, 255, 0)); // 초록색으로 채우기

        cout << "Basic OpenCV Mat creation: SUCCESS" << endl;
        cout << "Image size: " << testImage.cols << "x" << testImage.rows << endl;
        cout << "Image channels: " << testImage.channels() << endl;

        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error: " << e.what() << endl;
        return 0;
    }
    catch (...) {
        cerr << "Unknown error in OpenCV test" << endl;
        return 0;
    }
}

// 이미지 로드 테스트
extern "C" int test_image_load(const char* filepath) {
    try {
        cout << "Attempting to load image: " << filepath << endl;

        Mat image = imread(filepath, IMREAD_COLOR);

        if (image.empty()) {
            cerr << "Failed to load image: " << filepath << endl;
            return 0;
        }

        cout << "Image loaded successfully!" << endl;
        cout << "Size: " << image.cols << "x" << image.rows << endl;
        cout << "Channels: " << image.channels() << endl;
        cout << "Type: " << image.type() << endl;

        // 간단한 처리 테스트
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        cout << "Color conversion test: SUCCESS" << endl;

        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in image load: " << e.what() << endl;
        return 0;
    }
    catch (...) {
        cerr << "Unknown error in image load test" << endl;
        return 0;
    }
}