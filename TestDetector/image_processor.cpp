// image_processor.cpp - camera.cpp 원본 코드 이식
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// C 인터페이스를 위한 헤더들
extern "C" {
#include "image_processor.h"
#include "simple_types.h"
#include "image_result.h"
}

using namespace cv;
using namespace std;

// OpenCV Mat을 void*로 래핑하는 구조체
struct ImageData {
    Mat original;
    Mat result;
    Mat hsv;
    Mat binary;
    int width;
    int height;
};

// 이미지 로드 함수
extern "C" int load_image_opencv(const char* filepath, void** image_data) {
    try {
        ImageData* data = new ImageData();

        // 이미지 로드
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

// camera.cpp의 setColorRange 함수 그대로 이식
void setColorRange(Modoo_cfg* modoo_cfg, Scalar& lowerColor, Scalar& upperColor, int hue, int saturation, int value) {
    int hueBuffer = modoo_cfg->HsvBufferH;
    int satBuffer = modoo_cfg->HsvBufferS;
    int valBuffer = modoo_cfg->HsvBufferV;
    lowerColor = Scalar(max(0, hue - hueBuffer), max(0, saturation - satBuffer), max(0, value - valBuffer));
    upperColor = Scalar(min(180, hue + hueBuffer), min(255, saturation + satBuffer), min(255, value + valBuffer));
    cout << "set Color Range Lower: " << lowerColor << endl;
    cout << "set Color Range Upper: " << upperColor << endl;
}

// camera.cpp의 detectColorRegion 함수 그대로 이식 (단순화 버전)
bool detectColorRegion(const Mat& inputImage, Modoo_cfg* modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& center, int& radius) {
    cout << "None double check Color : " << productNumColor << endl;

    // 1. BGR 이미지를 HSV로 변환
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, COLOR_BGR2HSV);

    // 2. 색상 범위 설정
    Scalar lowerTargetColor, upperTargetColor;
    setColorRange(modoo_cfg, lowerTargetColor, upperTargetColor, static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]));

    // 3. 마스크 생성
    Mat mask;
    inRange(hsvImage, lowerTargetColor, upperTargetColor, mask);

    // 5. 컨투어 찾기
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = inputImage.clone(); // 원본 이미지를 복사
    // 모든 감지된 컨투어를 빨간색으로 그리기
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contourImage, contours, static_cast<int>(i), Scalar(0, 0, 255), 2); // 빨간색, 두께 2
    }

    double areaMax = 0.0;
    bool found = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > areaMax) {
            areaMax = area;
        }
        if (area >= modoo_cfg->minMarkArea && area < modoo_cfg->maxMarkArea) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
            cout << "감지 크기 : " + std::to_string(area) << "반지름 : " + std::to_string(radius) << "위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

            // 원이 감지되었는지 확인
            cout << "감지 color : " << targetHsv << endl;

            found = true;
            break; // 첫 번째 감지된 원으로 제한 종료
        }
    }

    if (!found) {
        cout << "감지 실패, 최대 감지 면적 : " << areaMax << endl;
    }
    return found;
}

// camera.cpp의 detectBlackArea 함수 그대로 이식
bool detectBlackArea(const Mat& inputImage, int buffer, Point& center, int& radius, Modoo_cfg modoo_cfg) {
    // BGR 채널 분리
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R 순으로 분리됨

    // B 채널은 bgrChannels[0]에 저장됨
    cv::Mat blueChannel = bgrChannels[0];

    //cv::Mat dst;
    //blueChannel.convertTo(dst, -1, 2.0, -175); // dst = src * 2 - 175
     //2. 스무딩 처리 (Gaussian Blur)
    Mat medianImage;
    cv::medianBlur(blueChannel, medianImage, 7); // 5x5 커널 사용

    // 3. 흑백 이미지 반전
    Mat invertedImage;
    bitwise_not(medianImage, invertedImage);

    int whitePixelCount = countPixelsAboveThreshold(invertedImage, 150);
    putText(invertedImage, "warea:" + std::to_string(whitePixelCount), Size(5, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);

    // 4. 이진 thresholding
    Mat thresholdedImage;
    double threshValue = 180; // 임계값 설정
    threshold(invertedImage, thresholdedImage, threshValue, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresholdedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    bool isdetected = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "감지 크기 : " + std::to_string(area) << endl;

        if (area >= modoo_cfg->blackMinMarkArea && area < modoo_cfg->blackMaxMarkArea) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산

            // 점과 반지름 및 사각형 크기 설정
            int squareSize = radius + 15; // 크롭할 사각형의 크기

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

// 타원의 비율과 크기 범위를 설정하는 함수
bool isEllipseValid(const cv::RotatedRect& ellipse, float minAspectRatio, float maxAspectRatio, float minSize, float maxSize) {
    // 너비와 높이의 비율 계산
    float aspectRatio = ellipse.size.width / ellipse.size.height;

    // 타원의 크기 확인 (타원의 너비와 높이의 평균으로 크기 설정)
    float size = (ellipse.size.width + ellipse.size.height) / 2.0;
    cout << "ellipse size : " << to_string(size) << endl;
    // 타원이 주어진 비율과 크기 범위 내에 있는지 확인
    return (aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio && size >= minSize && size <= maxSize);
}

cv::Mat edges; // 엣지 이미지를 전역으로 선언

bool detectCirclesAndEllipses(const cv::Mat& image, cv::Point& center, int radius, int squareSize, Modoo_cfg modoo_cfg) {
    bool isShow = false;
    cv::Canny(image, edges, 50, 150);

    // 원 검출 (HoughCircles)
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 100, 70, 10, 8, 16);

    // 결과를 위한 BGR 이미지 생성
    cv::Mat result;
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);

    bool circleDetected = !circles.empty(); // HoughCircles에서 원이 검출되었는지 확인

    // 원 시각화
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point circleCenter(static_cast<int>(circles[i][0]), static_cast<int>(circles[i][1]));
        int detectedRadius = static_cast<int>(circles[i][2]);

        // 원 테두리 그리기
        cv::circle(result, circleCenter, detectedRadius, cv::Scalar(0, 0, 255), 1);
        cout << "detected radius: " << detectedRadius << endl;
    }

    // 윤곽선 검출 (Contours)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool ellipseDetected = false;

    // 윤곽선을 이용한 타원 감지
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() >= 5) { // 타원을 맞추려면 최소 5개의 포인트 필요
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);

            // 타원의 비율 및 크기 검증
            if (isEllipseValid(ellipse, 0.7, 1.2, modoo_cfg.BlackEllipseMinSize, modoo_cfg.BlackEllipseMaxSize)) {
                // 타원의 중심과 반지름(대략적인 크기)을 저장
                center = cv::Point(static_cast<int>(ellipse.center.x), static_cast<int>(ellipse.center.y));
                radius = static_cast<int>((ellipse.size.width + ellipse.size.height) / 4); // 평균 크기의 절반을 반지름으로 사용

                // 타원 그리기
                cv::ellipse(result, ellipse, cv::Scalar(0, 255, 0), 2);
                ellipseDetected = true; // 타원이 감지되었음을 표시

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

    // 결과 이미지 출력
    if (isShow) {
        imshow("image", image);
        imshow("edges", edges);
        cv::imshow("Cropped Image with Detected Circles and Ellipses", result);
        cv::waitKey(0);
    }

    // 원이나 타원이 하나라도 감지되면 true 반환
    return ellipseDetected;
}

int countPixelsAboveThreshold(const cv::Mat& inputImage, int threshold) {

    // 이미지가 비어있는지 확인
    if (inputImage.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return -1;
    }

    // BGR 채널 분리
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R 순으로 분리됨

    // B 채널은 bgrChannels[0]에 저장됨
    cv::Mat blueChannel = bgrChannels[0];

    // 이미지가 단일 채널(그레이스케일)인지 확인
    if (blueChannel.channels() != 1) {
        std::cerr << "Input image must be a grayscale image!" << std::endl;
        return -1;
    }

    // 픽셀 값을 기준으로 카운트
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

// camera.cpp의 CaptureImage 로직을 그대로 이식한 함수
extern "C" int ProcessImageFromFile_Mode1(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    void* image_data = nullptr;

    try {
        // 이미지 로드
        if (!load_image_opencv(filepath, &image_data)) {
            printf("    [ERROR] Failed to load image\n");
            return 0;
        }

        ImageData* data = static_cast<ImageData*>(image_data);
        Mat& ori = data->original;
        Mat& resultImage = data->result;

        printf("    [DEBUG] Processing image: %dx%d\n", ori.cols, ori.rows);

        // camera.cpp의 그레이스케일 변환 부분 그대로
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

        // camera.cpp의 이진화 부분 그대로
        Mat binaryImage;
        threshold(grayImage, binaryImage, cfg->binaryValue, 255, THRESH_BINARY);

        Mat median;
        medianBlur(binaryImage, median, 21);

        // camera.cpp의 컨투어 찾기 부분 그대로
        vector<vector<Point>> contours;
        findContours(median, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 각 contour를 ROI로 설정하고 저장
        size_t savedCount = 0;
        std::vector<cv::Mat> croppedImages;

        result->total_detections = 0;
        result->success_count = 0;

        for (size_t i = 0; i < contours.size(); ++i) {
            bool isStickerOn = false;
            bool isColorStickerOn = false;

            // contour를 감싸는 최소 사각형 얻기
            Rect boundingRect = cv::boundingRect(contours[i]);

            // contour의 면적 계산
            double area = contourArea(contours[i]);

            // 면적을 문자열로 변환
            std::stringstream areaText;
            areaText << "Area: " << area;

            // 면적에 따라 서로 다른 색상으로 텍스트 표시
            Scalar textColor;

            if (area >= cfg->MinContourArea && area <= cfg->MaxContourArea) {
                result->total_detections++;

                textColor = Scalar(0, 255, 0);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "OK." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }

                // 모멘트를 사용하여 contour의 중심점 계산
                Moments m = moments(contours[i]);
                Point center(m.m10 / m.m00, m.m01 / m.m00);

                // 중심점에 원을 그리기
                circle(resultImage, center, 5, Scalar(0, 0, 255), -1); // 빨간색 원 그리기

                double resultX = cfg->origin_robot_y + (center.x - cfg->origin_vision_x) * cfg->res_x;
                double resultY = cfg->origin_robot_x + (center.y - cfg->origin_vision_y) * cfg->res_y;

                if (resultX < 0 || resultY < 0) {
                    putText(resultImage, "ordi minus", Size(boundingRect.tl().x, boundingRect.tl().y + 30), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    continue;
                }

                // 색상 검출 및 검은색 마커 검출 (camera.cpp와 동일한 로직)
                int buffer = 10;
                Point cropCenter;
                int radius;
                Scalar OnStickerColor;
                cv::Point cvtCenter;
                int colorIndex = cfg->ProductNum - 1;

                // 색상 검출 시도
                for (int j = 0; j < cfg->maxColorCount; j++) {
                    HsvColor hsvColor = cfg->hsvColors[j];
                    if (cfg->productNumColor[colorIndex] != hsvColor.id) {
                        continue;
                    }
                    cout << "HsvColor 검사 시작 ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                    cout << "검사 시작 productNum : " << colorIndex + 1 << ", productNumColor : " << cfg->productNumColor[colorIndex] << endl;

                    // HSV 값을 Scalar로 변환
                    Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                    int productNumColor = cfg->productNumColor[colorIndex];

                    // 색상 감지
                    bool detected = detectColorRegion(ori, cfg, productNumColor, targetHsv, cropCenter, radius);

                    if (detected) {
                        cout << "해당 색상 스티커 감지되었습니다." << endl;
                        cout << "중심: (" << cropCenter.x << ", " << cropCenter.y << "), 반지름: " << radius << endl;
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);

                        // HSV를 BGR로 변환
                        Mat hsvColor(1, 1, CV_8UC3, targetHsv);
                        Mat bgrColor;
                        cvtColor(hsvColor, bgrColor, COLOR_HSV2BGR);

                        isStickerOn = true;

                        // BGR 색상 추출
                        Vec3b bgr = bgrColor.at<Vec3b>(0, 0);
                        OnStickerColor = Scalar(bgr[0], bgr[1], bgr[2]);

                        // 검출 결과 저장
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

                // 검은색 마커 검출 시도
                if (!isStickerOn && cfg->productNumColor[colorIndex] == cfg->BlackTagNum) {
                    bool blackDetected = detectBlackArea(ori, buffer, cropCenter, radius, cfg);
                    if (blackDetected) {
                        cout << "검은 스티커 감지되었습니다. flag2" << endl;
                        cout << "중심: (" << cropCenter.x << ", " << cropCenter.y << "), 반지름: " << radius << endl;
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);
                        OnStickerColor = Scalar(255, 255, 255);
                        isStickerOn = true;

                        // 검출 결과 저장
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
                        cout << "검은 스티커 감지 안됨" << endl;
                    }
                }

                if (isStickerOn) {
                    // 변환된 BGR 색상으로 원 그리기
                    if (radius + 8 >= 0) {
                        circle(resultImage, cvtCenter, radius + 8, OnStickerColor, 3); // 감지된 색상으로 원 그리기
                    }
                    else {
                        circle(resultImage, cvtCenter, 20, OnStickerColor, 3); // 감지된 색상으로 원 그리기
                    }

                    // 결과 이미지 표시
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 30);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker On", tmp, FONT_HERSHEY_SIMPLEX, 1.5, OnStickerColor, 3);
                    }
                }
                else {
                    cout << "해당 색상 영역이 감지되지 않았습니다." << endl;
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 35);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker NG", tmp, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    }
                    continue;
                }

                // ROI를 둘러싼 사각형을 이미지에 그리기 (파란색 사각형, 두께 2)
                rectangle(resultImage, boundingRect, textColor, 2);
            }
            else if (area >= cfg->MinContourArea / 3 && area <= cfg->MaxContourArea * 3) {
                textColor = Scalar(0, 0, 255);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "NG." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }
            }
        }

        // 결과 이미지 저장
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

// 모드 2는 일단 기본 구현 유지
extern "C" int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    printf("    [INFO] Mode 2 processing not yet implemented with original camera.cpp logic\n");
    result->total_detections = 20;
    result->success_count = 0;
    return 1;
}

// 결과 이미지 저장 (실제 구현)
extern "C" int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result) {
    try {
        // 원본 이미지 로드
        Mat originalImage = imread(input_path, IMREAD_COLOR);
        if (originalImage.empty()) {
            return 0;
        }

        Mat resultImage = originalImage.clone();

        if (result->mode == 1) {
            // 모드 1: 단일 검출 결과 표시
            for (int i = 0; i < result->success_count; i++) {
                DetectionResult* det = &result->detections[i];

                // 원 그리기
                Scalar color = (strcmp(det->result_msg, "OK") == 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                circle(resultImage, Point(det->center_x, det->center_y), det->radius + 8, color, 3);

                // 텍스트 라벨 추가
                string label = string(det->result_msg) + " A:" + to_string(det->area);
                Point textPos(det->center_x + 10, det->center_y - 10);
                putText(resultImage, label, textPos, FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

                // 중심점 표시
                circle(resultImage, Point(det->center_x, det->center_y), 3, Scalar(255, 255, 0), -1);
            }
        }

        // 처리 시간 표시
        string timeInfo = "Time: " + to_string(result->processing_time) + "s";
        putText(resultImage, timeInfo, Point(10, resultImage.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // 이미지 저장
        bool success = imwrite(output_path, resultImage);
        return success ? 1 : 0;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in save_result_image_opencv: " << e.what() << endl;
        return 0;
    }
}

// 메모리 해제
extern "C" void free_image_data(void* image_data) {
    if (image_data) {
        ImageData* data = static_cast<ImageData*>(image_data);
        delete data;
    }
}

// 기타 함수들은 사용하지 않음
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