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

//2025.05.28
//디버깅 체크용 이미지 생성
static bool debugImageCreate = false;
int totalCount = 0;
double totalArea = 0.0;

extern "C" int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result, const Mat& processedImage);

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

// camera.cpp의 countPixelsAboveThreshold 함수 그대로 이식
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

// 타원의 비율과 크기 범위를 설정하는 함수 (camera.cpp에서 이식)
bool isEllipseValid(const cv::RotatedRect& ellipse, float minAspectRatio, float maxAspectRatio, float minSize, float maxSize) {
    // 너비와 높이의 비율 계산
    float aspectRatio = ellipse.size.width / ellipse.size.height;

    // 타원의 크기 확인 (타원의 너비와 높이의 평균으로 크기 설정)
    float size = (ellipse.size.width + ellipse.size.height) / 2.0;
    cout << "ellipse size : " << to_string(size)  << "  aspectRatio :  " << to_string(aspectRatio) << endl;
    // 타원이 주어진 비율과 크기 범위 내에 있는지 확인
    return (aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio && size >= minSize && size <= maxSize);
}

cv::Mat edges; // 엣지 이미지를 전역으로 선언
std::vector<cv::Vec3f> circles; // 검출된 원을 전역으로 선언

bool detectCirclesAndEllipses(const cv::Mat& image, cv::Point& center, int radius, int squareSize, Modoo_cfg modoo_cfg) {
    bool isShow = false;
    cv::Canny(image, edges, 50, 150);

    // 원 검출 (HoughCircles)
    //cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 100, 70, 10, 8, 16);
    cv::HoughCircles(edges, circles,
        cv::HOUGH_GRADIENT,
        1,          // dp
        100,         // minDist
        70,         // param1 (Canny 상위 임계값)
        10,         // param2 (낮은 값으로 관대하게)
        13, 18);    // 반지름 범위

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
    if (circleDetected) {
        return true;
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
                cv::ellipse(result, ellipse, cv::Scalar(255, 255, 255), 2);
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
    //string fileName = "results/Ellipses_test_result_" + to_string(detectionIndex) + ".jpg";
    //imwrite("results/Ellipses_test_result.jpg", result);

    // 원이나 타원이 하나라도 감지되면 true 반환
    return ellipseDetected;
}
// detectColorRegion 함수 시작 부분에 추가 (함수 맨 위에)
static int detectionIndex = 0; // 전역적으로 검출 인덱스 관리
static bool firstYN = false;
// camera.cpp의 detectColorRegion 함수 그대로 이식 (이중 체크 로직 포함)
bool detectColorRegion(const Mat& inputImage, Modoo_cfg* modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& circleCenter, int& circleRadius, Point& secondCenter, int& secondRadius) {
    //2025.03.10
    //분홍 마커 예외 처리 로직
    bool doublecheckYN = false;
    //bool firstYN = false;
    //tag 값 반듯이 확인 필요

    //20525.05.27
    //파랑 마커 원 검출 추가 로직
    bool circleCheckYN = false;

    if (productNumColor == 5) {
        cout << "double check Color : " << productNumColor << endl;
        doublecheckYN = true;
    }
    else
    {
        //cout << "None double check Color : " << productNumColor << endl;
    }

    if (productNumColor == 4) {
        cout << "circle check Color : " << productNumColor << endl;
        circleCheckYN = true;
    }
    else
    {
        //cout << "None double check Color : " << productNumColor << endl;
    }


    // 1. BGR 이미지를 HSV로 변환
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, COLOR_BGR2HSV);

    // 2. 색상 범위 설정
    Scalar lowerTargetColor, upperTargetColor;
    setColorRange(modoo_cfg, lowerTargetColor, upperTargetColor, static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]));

    // 3. 마스크 생성
    Mat mask;
    inRange(hsvImage, lowerTargetColor, upperTargetColor, mask);

    //imshow("mask", mask);
    //waitKey(2000);

    // 5. 컨투어 찾기
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = inputImage.clone(); // 원본 이미지를 복사
    

    // 모든 감지된 컨투어를 빨간색으로 그리기
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(contourImage, contours, static_cast<int>(i), Scalar(0, 0, 255), 2); // 빨간색, 두께 2
    }

    //imshow("contourImage", contourImage);
    //waitKey(2000);


    double areaMax = 0.0;
    bool found = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산

    printf("    [DEBUG] Processing image modoo_cfg->minMarkArea: %d  modoo_cfg->maxMarkArea : %d\n", modoo_cfg->minMarkArea, modoo_cfg->maxMarkArea);
    printf("    [DEBUG] [detectColorRegion] found mark Area count: %d  \n", contours.size());

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > areaMax) {
            areaMax = area;
        }
        

        if (area >= modoo_cfg->minMarkArea && area < modoo_cfg->maxMarkArea) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            Moments m = moments(contour);


            //파랑 마커의 경우 원 감지 추가
            if (circleCheckYN) {


                if (m.m00 == 0) continue; // 면적이 0인 경우 건너뛰기

                // 기존 모멘트 기반 계산
                Point center(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                int radius = static_cast<int>(sqrt(area / CV_PI));

                // ========== HoughCircles 검증 로직 ==========
                // 크롭 영역 정의 (예상 반지름 18 + 여유공간 10 = 28)
                int cropSize = 28 * 2; // 지름 + 여유공간
                int halfCrop = cropSize / 2;

                // 크롭 영역이 이미지 범위를 벗어나지 않도록 클리핑
                int cropX = max(0, center.x - halfCrop);
                int cropY = max(0, center.y - halfCrop);
                int cropWidth = min(cropSize, inputImage.cols - cropX);
                int cropHeight = min(cropSize, inputImage.rows - cropY);

                // 유효한 크롭 영역인지 확인
                if (cropWidth <= 0 || cropHeight <= 0) continue;

                Rect cropRect(cropX, cropY, cropWidth, cropHeight);
                Mat croppedForValidation = inputImage(cropRect);

                // 그레이스케일 변환 (HoughCircles는 그레이스케일 필요)
                Mat grayForValidation;
                if (croppedForValidation.channels() == 3) {
                    cvtColor(croppedForValidation, grayForValidation, COLOR_BGR2GRAY);
                }
                else {
                    grayForValidation = croppedForValidation;
                }

                // HoughCircles로 원형성 검증
                vector<Vec3f> validationCircles;
                HoughCircles(grayForValidation, validationCircles,
                    HOUGH_GRADIENT,
                    1,          // dp
                    30,         // minDist
                    50,         // param1 (Canny 상위 임계값)
                    12,         // param2 (낮은 값으로 관대하게)
                    14, 18);    // 반지름 범위

                // 원이 검증되었는지 확인
                bool isValidCircle = !validationCircles.empty();

                if (!isValidCircle){
                    printf("    [DEBUG] Circle validation Failed - Area: %.0f, Radius: %d\n", area, radius);
                    continue;
                }

                printf("    [DEBUG] Circle validation SUCCESS - Area: %.0f, Radius: %d\n", area, radius);

                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                cout << "    [DEBUG] " << "감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(radius) << "  위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                // 원이 감지되었는지 확인
                cout << "    [DEBUG] " << "감지 color : " << targetHsv << endl;

                circleCenter = center;
                circleRadius = radius;

                if (debugImageCreate) {
                    // =================== 검출된 원 시각화 및 저장 ===================
                   // 시각화를 위한 이미지 생성 (contourImage 복사)
                    Mat detectionVisualization = contourImage.clone();

                    // 검출된 원을 초록색으로 그리기
                    circle(detectionVisualization, center, radius, Scalar(0, 255, 0), 2);
                    circle(detectionVisualization, center, 2, Scalar(0, 0, 255), -1); // 중심점

                    // HSV 정보와 면적 정보를 텍스트로 표시
                    string hsvText = "HSV(" + to_string(static_cast<int>(targetHsv[0])) + "," +
                        to_string(static_cast<int>(targetHsv[1])) + "," +
                        to_string(static_cast<int>(targetHsv[2])) + ")";
                    string areaText = "Area: " + to_string(static_cast<int>(area));
                    string radiusText = "Radius: " + to_string(radius);

                    putText(detectionVisualization, hsvText, Point(center.x + 10, center.y - 30),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                    putText(detectionVisualization, areaText, Point(center.x + 10, center.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                    putText(detectionVisualization, radiusText, Point(center.x + 10, center.y + 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

                    // 마스크를 오버레이로 표시 (반투명하게)
                    Mat maskOverlay;
                    cvtColor(mask, maskOverlay, COLOR_GRAY2BGR);
                    maskOverlay *= 0.3; // 투명도 조절
                    detectionVisualization += maskOverlay;

                    // 파일명 생성 및 저장
                    string detectionImagePath = "results/color_detection_" + to_string(detectionIndex) + ".jpg";
                    bool saveSuccess = imwrite(detectionImagePath, detectionVisualization);

                    if (saveSuccess) {
                        printf("    [DEBUG] Color detection image saved: %s (HSV: %d,%d,%d, Area: %.0f)\n",
                            detectionImagePath.c_str(),
                            static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]),
                            area);
                    }
                    else {
                        printf("    [ERROR] Failed to save color detection image: %s\n", detectionImagePath.c_str());
                    }

                    detectionIndex++; // 다음 저장을 위해 인덱스 증가
                    // =================== 검출된 원 시각화 및 저장 끝 ===================
                }
               

                found = true;
                break; // 첫 번째 감지된 원으로 제한 종료
            }
            else 
            {

                if (doublecheckYN) {
                    if (firstYN) {
                        // 두 번째 원의 중심점과 반지름 계산
                        Point tempSecondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                        int tempSecondRadius = static_cast<int>(sqrt(area / CV_PI));

                        // 첫 번째 원과 두 번째 원의 거리 계산
                        int distanceX = abs(circleCenter.x - tempSecondCenter.x);
                        int distanceY = abs(circleCenter.y - tempSecondCenter.y);
                        double euclideanDistance = sqrt(pow(distanceX, 2) + pow(distanceY, 2));

                        printf("    [DEBUG] Distance check - First: (%d,%d), Second: (%d,%d), Distance: %.1f\n",
                            circleCenter.x, circleCenter.y, tempSecondCenter.x, tempSecondCenter.y, euclideanDistance);

                        // 거리 기준으로 동일 객체 판단 (x,y 차이가 각각 10 미만)
                        //if (distanceX < 10 && distanceY < 10) {
                        if (euclideanDistance < 40) {
                            printf("    [DEBUG] Same object detected (distance too close), skipping duplicate\n");
                            continue; // 동일 객체로 판단하여 스킵
                        }

                        // 거리가 충분히 멀면 실제 두 번째 원으로 인정

                        found = true;
                        secondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                        secondRadius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                        cout << "두번째 원 감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(secondRadius) << "  위치 : " + std::to_string(secondCenter.x) << "," << std::to_string(secondCenter.y) << endl;

                        // 원이 감지되었는지 확인
                        cout << "두번째 원 감지 color : " << targetHsv << endl;
                        firstYN = false;
                        break;
                    }
                    else
                    {
                        circleCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                        circleRadius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                        cout << "첫번 째 원 감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(circleRadius) << "  위치 : " + std::to_string(circleCenter.x) << "," << std::to_string(circleCenter.y) << endl;

                        // 원이 감지되었는지 확인
                        cout << "첫번 째 원 감지 color : " << targetHsv << endl;

                        if (debugImageCreate) {
                            // =================== 검출된 원 시각화 및 저장 ===================
                        // 시각화를 위한 이미지 생성 (contourImage 복사)
                            Mat detectionVisualization = contourImage.clone();

                            // 검출된 원을 초록색으로 그리기
                            circle(detectionVisualization, circleCenter, circleRadius, Scalar(0, 255, 0), 2);
                            circle(detectionVisualization, circleCenter, 2, Scalar(0, 0, 255), -1); // 중심점

                            // HSV 정보와 면적 정보를 텍스트로 표시
                            string hsvText = "HSV(" + to_string(static_cast<int>(targetHsv[0])) + "," +
                                to_string(static_cast<int>(targetHsv[1])) + "," +
                                to_string(static_cast<int>(targetHsv[2])) + ")";
                            string areaText = "Area: " + to_string(static_cast<int>(area));
                            string radiusText = "Radius: " + to_string(circleRadius);

                            putText(detectionVisualization, hsvText, Point(circleCenter.x + 10, circleCenter.y - 30),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                            putText(detectionVisualization, areaText, Point(circleCenter.x + 10, circleCenter.y - 10),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                            putText(detectionVisualization, radiusText, Point(circleCenter.x + 10, circleCenter.y + 10),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

                            // 마스크를 오버레이로 표시 (반투명하게)
                            Mat maskOverlay;
                            cvtColor(mask, maskOverlay, COLOR_GRAY2BGR);
                            maskOverlay *= 0.3; // 투명도 조절
                            detectionVisualization += maskOverlay;

                            // 파일명 생성 및 저장
                            string detectionImagePath = "results/double_color_detection_" + to_string(detectionIndex) + ".jpg";
                            bool saveSuccess = imwrite(detectionImagePath, detectionVisualization);

                            if (saveSuccess) {
                                printf("    [DEBUG] Color detection image saved: %s (HSV: %d,%d,%d, Area: %.0f)\n",
                                    detectionImagePath.c_str(),
                                    static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]),
                                    area);
                            }
                            else {
                                printf("    [ERROR] Failed to save color detection image: %s\n", detectionImagePath.c_str());
                            }

                            detectionIndex++; // 다음 저장을 위해 인덱스 증가
                            // =================== 검출된 원 시각화 및 저장 끝 ==================
                        }
                        

                        firstYN = true;
                        continue;
                    }
                }
                else
                {
                    circleCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    circleRadius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                    cout << "    [DEBUG] " << "감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(circleRadius) << "  위치 : " + std::to_string(circleCenter.x) << "," << std::to_string(circleCenter.y) << endl;

                    // 원이 감지되었는지 확인
                    cout << "    [DEBUG] " << "감지 color : " << targetHsv << endl;

                    if (debugImageCreate) {
                        // =================== 검출된 원 시각화 및 저장 ===================
                        // 시각화를 위한 이미지 생성 (contourImage 복사)
                        Mat detectionVisualization = contourImage.clone();

                        // 검출된 원을 초록색으로 그리기
                        circle(detectionVisualization, circleCenter, circleRadius, Scalar(0, 255, 0), 2);
                        circle(detectionVisualization, circleCenter, 2, Scalar(0, 0, 255), -1); // 중심점

                        // HSV 정보와 면적 정보를 텍스트로 표시
                        string hsvText = "HSV(" + to_string(static_cast<int>(targetHsv[0])) + "," +
                            to_string(static_cast<int>(targetHsv[1])) + "," +
                            to_string(static_cast<int>(targetHsv[2])) + ")";
                        string areaText = "Area: " + to_string(static_cast<int>(area));
                        string radiusText = "Radius: " + to_string(circleRadius);

                        putText(detectionVisualization, hsvText, Point(circleCenter.x + 10, circleCenter.y - 30),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                        putText(detectionVisualization, areaText, Point(circleCenter.x + 10, circleCenter.y - 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                        putText(detectionVisualization, radiusText, Point(circleCenter.x + 10, circleCenter.y + 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

                        // 마스크를 오버레이로 표시 (반투명하게)
                        Mat maskOverlay;
                        cvtColor(mask, maskOverlay, COLOR_GRAY2BGR);
                        maskOverlay *= 0.3; // 투명도 조절
                        detectionVisualization += maskOverlay;

                        // 파일명 생성 및 저장
                        string detectionImagePath = "results/color_detection_non_circle_" + to_string(detectionIndex) + ".jpg";
                        bool saveSuccess = imwrite(detectionImagePath, detectionVisualization);

                        if (saveSuccess) {
                            printf("    [DEBUG] Color detection image saved: %s (HSV: %d,%d,%d, Area: %.0f)\n",
                                detectionImagePath.c_str(),
                                static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]),
                                area);
                        }
                        else {
                            printf("    [ERROR] Failed to save color detection image: %s\n", detectionImagePath.c_str());
                        }

                        detectionIndex++; // 다음 저장을 위해 인덱스 증가
                        // =================== 검출된 원 시각화 및 저장 끝 ===================
                    }

                    

                    found = true;
                    break; // 첫 번째 감지된 원으로 제한 종료
                }

            }
            // ========== HoughCircles 검증 로직 끝 ==========


            
        }
        else
        {
            printf("    [DEBUG] [detectColorRegion] non mark boundary: %f  \n", area);
            //cout << "감지 실패, 최대 감지 면적 : " << areaMax << endl;
            //Mat contourSmaill = inputImage.clone(); // 원본 이미지를 복사
            //drawContours(contourImage, contour, static_cast<int>(0), Scalar(0, 0, 255), 2); // 빨간색, 두께 2
            //imshow("contourImage", contourImage);
            //waitKey(2000);
        }
    }

    if (!found) {
        //cout << "감지 실패, 최대 감지 면적 : " << areaMax << endl;
        //Mat contourSmaill = inputImage.clone(); // 원본 이미지를 복사
        //drawContours(contourImage, contour, static_cast<int>(0), Scalar(0, 0, 255), 2); // 빨간색, 두께 2
        //imshow("contourImage", contourImage);
        //waitKey(2000);
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

    //2. 스무딩 처리 (Gaussian Blur)
    Mat medianImage;
    cv::medianBlur(blueChannel, medianImage, 7); // 5x5 커널 사용

    // 3. 흑백 이미지 반전
    Mat invertedImage;
    bitwise_not(medianImage, invertedImage);

    int whitePixelCount = countPixelsAboveThreshold(invertedImage, 150);
    putText(invertedImage, "warea:" + std::to_string(whitePixelCount), Size(5, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    cout << "    [DEBUG] detectBlackArea  " << "warea : " + to_string(whitePixelCount) << endl;
    // 4. 이진 thresholding
    Mat thresholdedImage;
    double threshValue = 180; // 임계값 설정
    threshold(invertedImage, thresholdedImage, threshValue, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresholdedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    bool isdetected = false;
    bool isShow = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "detectBlackArea   감지 크기 : " + std::to_string(area) << endl;

        if (area >= modoo_cfg.blackMinMarkArea && area < modoo_cfg.blackMaxMarkArea) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산

            // 점과 반지름 및 사각형 크기 설정
            int squareSize = radius + 15; // 크롭할 사각형의 크기

            bool isCircleDetected = detectCirclesAndEllipses(thresholdedImage, center, radius, squareSize, modoo_cfg);

            if (isCircleDetected) {
                std::cout << "Circle detected!" << std::endl;
                if (isShow) {
                    // 중간 과정 및 결과 표시
                    imshow("Gray Image", blueChannel);
                    imshow("median Image", medianImage);
                    imshow("Inverted Image", invertedImage);
                    imshow("Thresholded Image", thresholdedImage);
                    waitKey(0); // 키 입력 대기

                    destroyAllWindows(); // 모든 창 닫기         
                }

                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                cout << "    [DEBUG] detectBlackArea  " << "감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(radius) << "  위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                if (debugImageCreate) {
                    // =================== 검출된 원 시각화 및 저장 ===================
                    // 시각화를 위한 이미지 생성 (contourImage 복사)
                    Mat detectionVisualization = thresholdedImage.clone();

                    // 검출된 원을 초록색으로 그리기
                    circle(detectionVisualization, center, radius, Scalar(255, 255, 255), 2);
                    circle(detectionVisualization, center, 2, Scalar(255, 255, 255), -1); // 중심점


                    string areaText = "Area: " + to_string(static_cast<int>(area));
                    string radiusText = "Radius: " + to_string(radius);

                    cout << "    [DEBUG] detectBlackArea  " << "Area : " + areaText << "  Radius : " << radiusText << endl;
                    putText(detectionVisualization, areaText, Point(center.x + 10, center.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                    putText(detectionVisualization, radiusText, Point(center.x + 10, center.y + 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

                    // 마스크를 오버레이로 표시 (반투명하게)
                    cout << "    [DEBUG] detectBlackArea  " << "detectionIndex : " + detectionIndex << endl;
                    string  invertedImagePath = "results/invertedImage_" + to_string(detectionIndex) + ".jpg";


                    imwrite(invertedImagePath, invertedImage);
                    // 파일명 생성 및 저장
                    string detectionImagePath = "results/black_detection_" + to_string(detectionIndex) + ".jpg";
                    bool saveSuccess = imwrite(detectionImagePath, detectionVisualization);

                    if (saveSuccess) {
                        printf("    [DEBUG] Color detection image saved: %s ( Area: %.0f)\n",
                            detectionImagePath.c_str(),
                            area);
                    }
                    else {
                        printf("    [ERROR] Failed to save color detection image: %s\n", detectionImagePath.c_str());
                    }

                    detectionIndex++; // 다음 저장을 위해 인덱스 증가
                    // =================== 검출된 원 시각화 및 저장 끝 ===================

                }
                
                //2025.05.27
                //원 검출 별도 로직 추가
                if (radius < 14) {
                    cout << "    [DEBUG] detectBlackArea radius too small -> false  " << "Area : " << to_string(static_cast<int>(area)) << "  Radius : " << to_string(radius) << endl;
                    return false;
                }

                return true;
            }
            else {


                std::cout << "    [DEBUG] " << "No circle detected." << std::endl;

                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                cout << "    [DEBUG] " << "감지 크기 : " + std::to_string(area) << "  반지름 : " + std::to_string(radius) << "  위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;

                if (debugImageCreate) {
                    // =================== 검출된 원 시각화 및 저장 ===================
                   // 시각화를 위한 이미지 생성 (contourImage 복사)
                    Mat detectionVisualization = thresholdedImage.clone();

                    // 검출된 원을 초록색으로 그리기
                    circle(detectionVisualization, center, radius, Scalar(0, 255, 0), 2);
                    circle(detectionVisualization, center, 2, Scalar(0, 0, 255), -1); // 중심점


                    string areaText = "Area: " + to_string(static_cast<int>(area));
                    string radiusText = "Radius: " + to_string(radius);


                    putText(detectionVisualization, areaText, Point(center.x + 10, center.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                    putText(detectionVisualization, radiusText, Point(center.x + 10, center.y + 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

                    // 마스크를 오버레이로 표시 (반투명하게)

                    string  invertedImagePath = "results/non_invertedImage_" + to_string(detectionIndex) + ".jpg";


                    imwrite(invertedImagePath, invertedImage);
                    // 파일명 생성 및 저장
                    string detectionImagePath = "results/non_black_detection_" + to_string(detectionIndex) + ".jpg";
                    bool saveSuccess = imwrite(detectionImagePath, detectionVisualization);

                    if (saveSuccess) {
                        printf("    [DEBUG] Color detection image saved: %s ( Area: %.0f)\n",
                            detectionImagePath.c_str(),
                            area);
                    }
                    else {
                        printf("    [ERROR] Failed to save color detection image: %s\n", detectionImagePath.c_str());
                    }

                    detectionIndex++; // 다음 저장을 위해 인덱스 증가
                    // =================== 검출된 원 시각화 및 저장 끝 ===================
                }
               

            }
        }
    }

    if (isShow) {
        // 중간 과정 및 결과 표시
        imshow("Gray Image", blueChannel);
        imshow("median Image", medianImage);
        imshow("Inverted Image", invertedImage);
        imshow("Thresholded Image", thresholdedImage);
        waitKey(0); // 키 입력 대기

        destroyAllWindows(); // 모든 창 닫기         
    }

    return false;
}

// camera.cpp의 detectColorRegion3rd 함수 그대로 이식
bool detectColorRegion3rd(const Mat& inputImage, Modoo_cfg modoo_cfg, int productNumColor, const Scalar& targetHsv, Point& center, int& radius, int& markAreaSum, Point& secondCenter, int& secondRadius) {
    //2025.03.10
    //분홍 마커 예외 처리 로직
    bool doublecheckYN = false;
    //int firstYN = false;
    //tag 값 반듯시 확인 필요
    if (productNumColor == 5) {
        cout << "double check Color : " << productNumColor << endl;
        doublecheckYN = true;
    }
    else
    {
        cout << "None double check Color : " << productNumColor << endl;
    }

    // 1. BGR 이미지를 HSV로 변환
    Mat hsvImage;
    cvtColor(inputImage, hsvImage, COLOR_BGR2HSV);

    // 2. 색상 범위 설정
    Scalar lowerTargetColor, upperTargetColor;
    setColorRange(&modoo_cfg, lowerTargetColor, upperTargetColor, static_cast<int>(targetHsv[0]), static_cast<int>(targetHsv[1]), static_cast<int>(targetHsv[2]));

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

    bool found = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산
    for (const auto& contour : contours) {
        double area = contourArea(contour);

        printf("    [DEBUG] detectColorRegion3rd ->Processing image modoo_cfg->minMarkArea: %d  modoo_cfg->maxMarkArea : %d\n", modoo_cfg.minMarkArea, modoo_cfg.maxMarkArea);
        printf("    [DEBUG] [detectColorRegion3rd] found mark Area count: %d  \n", contours.size());


        if (area >= modoo_cfg.minMarkArea && area < modoo_cfg.maxMarkArea * 1.3) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            printf("    [DEBUG] [detectColorRegion3rd] found mark Area : %f  \n", area);
            Moments m = moments(contour);
            if (doublecheckYN) {
                if (firstYN) {
                    // 두 번째 원의 중심점과 반지름 계산
                    Point tempSecondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    int tempSecondRadius = static_cast<int>(sqrt(area / CV_PI));

                    // 첫 번째 원과 두 번째 원의 거리 계산
                    int distanceX = abs(center.x - tempSecondCenter.x);
                    int distanceY = abs(center.y - tempSecondCenter.y);
                    double euclideanDistance = sqrt(pow(distanceX, 2) + pow(distanceY, 2));

                    printf("    [DEBUG] Distance check - First: (%d,%d), Second: (%d,%d), Distance: %.1f\n",
                        center.x, center.y, tempSecondCenter.x, tempSecondCenter.y, euclideanDistance);

                    // 거리 기준으로 동일 객체 판단 (x,y 차이가 각각 10 미만)
                    if (distanceX < 10 && distanceY < 10) {
                        printf("    [DEBUG] Same object detected (distance too close), skipping duplicate\n");
                        continue; // 동일 객체로 판단하여 스킵
                    }

                    secondCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    secondRadius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                    cout << "두번째 감지 크기 : " + std::to_string(area) << "반지름 : " + std::to_string(secondRadius) << "위치 : " + std::to_string(secondCenter.x) << "," << std::to_string(secondCenter.y) << endl;
                    found = true;
                    // 원이 감지되었는지 확인
                    cout << "두번째 감지 color : " << targetHsv << endl;
                    firstYN = false;
                    break; // 첫 번째 감지된 원으로 제한 종료
                }
                else
                {
                    center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                    radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                    markAreaSum = area;
                    cout << "첫번째 감지 크기 : " + std::to_string(area) << "반지름 : " + std::to_string(radius) << "위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;
                    // 원이 감지되었는지 확인
                    cout << "첫번째 감지 color : " << targetHsv << endl;
                    firstYN = true;
                }
            }
            else
            {
                center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산
                markAreaSum = area;
                cout << "감지 크기 : " + std::to_string(area) << "반지름 : " + std::to_string(radius) << "위치 : " + std::to_string(center.x) << "," << std::to_string(center.y) << endl;
                found = true;
                // 원이 감지되었는지 확인
                cout << "감지 color : " << targetHsv << endl;
                break; // 첫 번째 감지된 원으로 제한 종료
            }
        }
        else
        {
            printf("    [DEBUG] [detectColorRegion3rd] not found mark Area : %f  \n", area);
        }
    }

    if (!found) {
        markAreaSum = 0;
    }

    return found;
}

// camera.cpp의 detectBlackArea3rd 함수 그대로 이식
bool detectBlackArea3rd(const Mat& inputImage, int buffer, Point& center, int& radius, Modoo_cfg modoo_cfg) {
    // BGR 채널 분리
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);  // B, G, R 순으로 분리됨

    // B 채널은 bgrChannels[0]에 저장됨
    cv::Mat blueChannel = bgrChannels[0];

    cv::Mat dst;
    blueChannel.convertTo(dst, -1, 2.0, -175); // dst = src * 2 - 175

    // 3. 흑백 이미지 반전
    Mat invertedImage;
    bitwise_not(dst, invertedImage);

    // 4. 이진 thresholding
    Mat thresholdedImage;
    double threshValue = 240; // 임계값 설정
    threshold(invertedImage, thresholdedImage, threshValue, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresholdedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (false) {
        // 중간 과정 및 결과 표시
        imshow("Gray Image", blueChannel);
        imshow("gamma", dst);
        imshow("Inverted Image", invertedImage);
        imshow("Thresholded Image", thresholdedImage);
        waitKey(0); // 키 입력 대기

        destroyAllWindows(); // 모든 창 닫기
    }

    bool isdetected = false;
    // 6. 감지된 영역의 면적을 이용해 반지름 계산
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        cout << "감지 크기 : " + std::to_string(area) << endl;

        if (area >= modoo_cfg.blackMinMarkArea && area < modoo_cfg.blackMaxMarkArea) { // 최소 면적 기준
            // 모멘트를 계산하여 중심점과 반지름 반환
            Moments m = moments(contour);
            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            radius = static_cast<int>(sqrt(area / CV_PI)); // 면적을 이용한 반지름 계산

            // 점과 반지름 및 사각형 크기 설정
            int squareSize = radius + 15; // 크롭할 사각형의 크기

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

// camera.cpp의 CaptureImage 로직을 그대로 이식한 함수
// camera.cpp의 CaptureImage 로직을 그대로 이식한 함수 (수정 버전)
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

        // 이미지가 비어있는지 확인
        if (ori.empty()) {
            printf("    [ERROR] Original image is empty\n");
            free_image_data(image_data);
            return 0;
        }

        int leftTop = 300;
        int rightBot = 1700;
        totalArea = 0.0;
        totalCount = 0;

        // camera.cpp의 그레이스케일 변환 부분 수정
        Mat grayImage;
        if (cfg->hsvEnable == 1) {
            printf("hsvEnable : True\n");
            // HSV 모드일 때도 일단 그레이스케일로 변환 (원본 로직과 일치시키기 위해)
            cvtColor(ori, grayImage, COLOR_BGR2GRAY);
            //cvtColor(ori, grayImage, COLOR_RGB2GRAY);
        }
        else if (cfg->hsvEnable == 0) {
            cvtColor(ori, grayImage, COLOR_BGR2GRAY);
            //cvtColor(ori, grayImage, COLOR_RGB2GRAY);
            printf("hsvEnable : False\n");
        }
        else {
            printf("error-hsvEnable, using default grayscale conversion\n");
            //cvtColor(ori, grayImage, COLOR_RGB2GRAY);
            cvtColor(ori, grayImage, COLOR_BGR2GRAY);
        }

        // 그레이스케일 이미지가 제대로 생성되었는지 확인
        if (grayImage.empty()) {
            printf("    [ERROR] Grayscale conversion failed\n");
            free_image_data(image_data);
            return 0;
        }

        printf("    [DEBUG] Grayscale image created: %dx%d\n", grayImage.cols, grayImage.rows);

        // camera.cpp의 이진화 부분 그대로
        Mat binaryImage;
        threshold(grayImage, binaryImage, cfg->binaryValue, 255, THRESH_BINARY);

        // 이진화 이미지 확인
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

        // 미디안 블러 결과 확인
        if (median.empty()) {
            printf("    [ERROR] Median blur failed\n");
            free_image_data(image_data);
            return 0;
        }

        printf("    [DEBUG] Median blur completed: %dx%d\n", median.cols, median.rows);

        // camera.cpp의 컨투어 찾기 부분 그대로
        vector<vector<Point>> contours;
        findContours(median, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        printf("    [DEBUG] Found %zu contours\n", contours.size());

        result->total_detections = 0;
        result->success_count = 0;
        if (debugImageCreate) {
            // =========================== 컨투어 시각화 디버깅 코드 추가 ===========================
        // 컨투어 시각화를 위한 Mat 생성 (원본 이미지 기반)
            Mat contourDebugImage = ori.clone();

            // 모든 컨투어를 그리고 area 값을 표시
            for (size_t i = 0; i < contours.size(); i++) {
                // 컨투어 경계선 그리기 (빨간색, 두께 2)
                drawContours(contourDebugImage, contours, static_cast<int>(i), Scalar(0, 0, 255), 2);

                // 컨투어의 면적 계산
                double area = contourArea(contours[i]);

                // 컨투어의 중심점 계산 (모멘트 사용)
                Moments m = moments(contours[i]);
                if (m.m00 != 0) {
                    Point center(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

                    // 중심점에 작은 원 그리기 (파란색)
                    circle(contourDebugImage, center, 3, Scalar(255, 0, 0), -1);

                    // 면적 값을 텍스트로 표시
                    string areaText = to_string(static_cast<int>(area));
                    putText(contourDebugImage, areaText, Point(center.x + 5, center.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

                    // 컨투어 번호도 표시
                    string indexText = "#" + to_string(i);
                    putText(contourDebugImage, indexText, Point(center.x + 5, center.y + 15),
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
                }
            }

            // 설정된 면적 범위를 이미지에 표시
            string areaRangeText = "Area Range: " + to_string(cfg->MinContourArea) + " ~ " + to_string(cfg->MaxContourArea);
            putText(contourDebugImage, areaRangeText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

            // 컨투어 개수 표시
            string contourCountText = "Total Contours: " + to_string(contours.size());
            putText(contourDebugImage, contourCountText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

            // 컨투어 디버그 이미지 저장
            // 원본 파일명에서 확장자 제거하고 _contours 접미사 추가
            string originalPath(filepath);
            size_t lastSlash = originalPath.find_last_of("/\\");
            size_t lastDot = originalPath.find_last_of(".");
            string fileName = originalPath.substr(lastSlash + 1, lastDot - lastSlash - 1);
            string debugImagePath = "results/" + fileName + "_contours.jpg";

            bool saveSuccess = imwrite(debugImagePath, contourDebugImage);
            if (saveSuccess) {
                printf("    [DEBUG] Contour debug image saved: %s\n", debugImagePath.c_str());
            }
            else {
                printf("    [ERROR] Failed to save contour debug image: %s\n", debugImagePath.c_str());
            }
            // =========================== 컨투어 시각화 디버깅 코드 끝 ===========================

        }
        

        for (size_t i = 0; i < contours.size(); ++i) {
            bool isStickerOn = false;
            bool isColorStickerOn = false;

            

            // contour를 감싸는 최소 사각형 얻기
            Rect boundingRect = cv::boundingRect(contours[i]);

            // 바운딩 박스가 이미지 범위를 벗어나지 않도록 확인
            boundingRect &= Rect(0, 0, ori.cols, ori.rows);
            if (boundingRect.width <= 0 || boundingRect.height <= 0) {
                continue;
            }

            // contour의 면적 계산
            double area = contourArea(contours[i]);


            // 면적을 문자열로 변환
            std::stringstream areaText;
            areaText << "Area: " << area;

            // 면적에 따라 서로 다른 색상으로 텍스트 표시
            Scalar textColor;
            totalArea = totalArea + area;

            if (area >= cfg->MinContourArea && area <= cfg->MaxContourArea) {
                printf("    [DEBUG] Found %f contours area\n", area);
                //printf("    [DEBUG] areaText.str() : %s\n", areaText.str());
                
                totalCount++;
                result->total_detections++;

                textColor = Scalar(0, 255, 0);
                if (cfg->debugMode == 1) {
                    putText(resultImage, "OK." + areaText.str(), boundingRect.tl(), FONT_HERSHEY_SIMPLEX, 1.5, textColor, 3);
                }

                // 모멘트를 사용하여 contour의 중심점 계산
                Moments m = moments(contours[i]);
                if (m.m00 == 0) continue; // 면적이 0인 경우 건너뛰기

                Point center(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

                // 중심점에 원을 그리기
                circle(resultImage, center, 5, Scalar(0, 0, 255), -1); // 빨간색 원 그리기

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

                // 윤곽선으로부터 마스크 생성
                cv::Mat mask = cv::Mat::zeros(median.size(), CV_8UC1);
                cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

                // 윤곽선 경계면을 안쪽으로 밀기 (erode 사용)
                int erosionSize = 15; // 이 값을 조정하여 크롭 범위 조정
                cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                    cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),
                    cv::Point(erosionSize, erosionSize));
                cv::erode(mask, mask, element);

                // 원본 컬러 이미지에서 마스크를 사용하여 크롭
                cv::Mat croppedImage;
                ori.copyTo(croppedImage, mask); // 마스크에 의해 선택된 영역만 복사

                // 최종적으로 배경을 흰색으로 설정하기
                cv::Mat whiteCroppedImage(median.size(), ori.type(), cv::Scalar(255, 255, 255)); // 흰색 배경 생성
                croppedImage.copyTo(whiteCroppedImage, mask); // 마스크에 의해 선택된 영역만 복사

                croppedImage = croppedImage(boundingRect);
                whiteCroppedImage = whiteCroppedImage(boundingRect);

                // 크롭된 이미지가 비어있지 않은지 확인
                if (croppedImage.empty() || whiteCroppedImage.empty()) {
                    printf("    [WARNING] Cropped image is empty for contour %zu\n", i);
                    continue;
                }

                // 색상 검출 및 검은색 마커 검출 (camera.cpp와 동일한 로직)
                int buffer = 10;
                Point cropCenter;
                int radius;
                Point secondCropCenter;
                int secondRadius = 0;
                Scalar OnStickerColor;
                cv::Point cvtCenter;
                int colorIndex = cfg->ProductNum - 1;

                // colorIndex 범위 확인
                if (colorIndex < 0 || colorIndex >= 20) {
                    colorIndex = 0; // 기본값으로 설정
                }



                // 현재 제품에 해당하는 색상 ID 확인
                int target_color_id = cfg->productNumColor[colorIndex];
                printf("    [DEBUG] ProductNum: %d, colorIndex: %d, target_color_id: %d\n",
                    cfg->ProductNum, colorIndex, target_color_id);


                double angle = 0.0;
                bool color_detected = false;

                // 색상 검출 시도
                for (int j = 0; j < cfg->maxColorCount; j++) {
                    HsvColor hsvColor = cfg->hsvColors[j];
                    

                    
                    // 핵심: hsvColor.id가 target_color_id와 일치하는지 확인
                    if (hsvColor.id != target_color_id) {
                        /*
                        printf("    [DEBUG] Skipping HsvColor[%d] id=%d (target=%d)\n",
                            j, hsvColor.id, target_color_id);
                         */
                        continue;
                    }

                    cout << "HsvColor 검사 시작 ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                    cout << "검사 시작 productNum : " << colorIndex + 1 << ", productNumColor : " << cfg->productNumColor[colorIndex] << endl;

                    // 유효한 HSV 값인지 확인
                    if (hsvColor.h <= 0 || hsvColor.h > 180 || hsvColor.s <= 0 || hsvColor.s > 255 || hsvColor.v <= 0 || hsvColor.v > 255) {
                        cout << "HsvColor 값 이상 : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << endl;
                        continue;  // HSV 값이 유효하지 않으면 다음으로 넘어감
                    }

                    // HSV 값을 Scalar로 변환
                    Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                    int productNumColor = cfg->productNumColor[colorIndex];

                    // 색상 감지
                    bool detected = detectColorRegion(croppedImage, cfg, productNumColor, targetHsv, cropCenter, radius, secondCropCenter, secondRadius);

                    if (detected) {
                        cout << "해당 색상 스티커 감지되었습니다." << endl;
                        cout << "중심: (" << cropCenter.x << ", " << cropCenter.y << "), 반지름: " << radius << endl;
                        printf("    [SUCCESS] Target color detected! HSV(%d,%d,%d) ID=%d\n",
                            hsvColor.h, hsvColor.s, hsvColor.v, hsvColor.id);
                        printf("    [SUCCESS] Center: (%d, %d), Radius: %d\n",
                            cropCenter.x, cropCenter.y, radius);
                        printf("    [SUCCESS] cvtCenter: (%d, %d)\n",
                            cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);
                        cvtCenter = Point(cropCenter.x + boundingRect.tl().x, cropCenter.y + boundingRect.tl().y);

                        // HSV를 BGR로 변환
                        Mat hsvColorMat(1, 1, CV_8UC3, targetHsv);  // 1x1 크기의 Mat 생성 후 HSV 값 할당
                        Mat bgrColorMat;
                        cvtColor(hsvColorMat, bgrColorMat, COLOR_HSV2BGR);  // HSV -> BGR 변환

                        isStickerOn = true;
                        color_detected = true;


                        // BGR 색상 추출
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

                // 매칭되는 색상이 없는 경우 로그 출력
                /*
                if (!color_detected) {
                    printf("    [INFO] No matching color found for target_color_id=%d\n", target_color_id);

                    // 사용 가능한 모든 색상 ID 출력 (디버깅용)
                    printf("    [DEBUG] Available color IDs: ");
                    for (int j = 0; j < cfg->maxColorCount; j++) {
                        printf("%d ", cfg->hsvColors[j].id);
                    }
                    printf("\n");
                }
                */

                // 검은색 마커 검출 시도
                if (!isStickerOn && cfg->productNumColor[colorIndex] == cfg->BlackTagNum) {
                    bool blackDetected = detectBlackArea(whiteCroppedImage, buffer, cropCenter, radius, *cfg);
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
                        cout << "검은 스티커 미 감지" << endl;
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

                    //2025.03.10
                    //두번째 마커 표기
                    if (secondRadius != 0) {
                        Point adjustedSecondCenter = Point(secondCropCenter.x + boundingRect.tl().x, secondCropCenter.y + boundingRect.tl().y);
                        if (secondRadius + 8 >= 0) {
                            circle(resultImage, adjustedSecondCenter, secondRadius + 8, OnStickerColor, 3); // 감지된 색상으로 원 그리기
                        }
                        else
                        {
                            circle(resultImage, adjustedSecondCenter, 20, OnStickerColor, 3); // 감지된 색상으로 원 그리기
                        }
                    }

                    // 결과 이미지 표시
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 30);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker On", tmp, FONT_HERSHEY_SIMPLEX, 1.5, OnStickerColor, 3);
                    }

                    // 검출 결과 저장
                    DetectionResult* det = &result->detections[result->success_count];
                    det->detected = 1;
                    det->center_x = cvtCenter.x;
                    det->center_y = cvtCenter.y;
                    det->radius = radius;
                    det->area = static_cast<int>(area);
                    strcpy_s(det->result_msg, sizeof(det->result_msg), "OK");
                    strcpy_s(det->color_info, sizeof(det->color_info), isStickerOn ? "Color" : "Black");

                    result->success_count++;

                    // 좌표값을 문자열로 변환
                    std::ostringstream ossX;
                    ossX << std::fixed << std::setprecision(3) << std::setw(7) << std::setfill('0') << resultX;
                    std::string strResultX = ossX.str();

                    std::ostringstream ossY;
                    ossY << std::fixed << std::setprecision(3) << std::setw(7) << std::setfill('0') << resultY;
                    std::string strResultY = ossY.str();

                    // 좌표값을 문자열로 변환
                    std::string coordinates = "(" + strResultY + ", " + strResultX + ")";

                    printf("    [SUCCESS] coordinates: %s\n", coordinates);

                    // 중심점 좌표를 이미지에 텍스트로 표시
                    putText(resultImage, coordinates, Point(center.x + 10, center.y), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3);

                    // 각도 계산 및 표시 (camera.cpp의 calculateAngle 함수 로직)
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
                    printf("    [SUCCESS] formattedAngle: %s\n" , formattedAngle);
                    putText(resultImage, formattedAngle, Point(center.x + 10, center.y + 40), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 0), 3);

                    isStickerOn = false;
                }
                else {
                    cout << "해당 색상 영역이 감지되지 않았습니다." << endl;
                    cv::Point tmp = cv::Point(boundingRect.tl().x, boundingRect.tl().y - 35);
                    if (cfg->debugMode == 1) {
                        putText(resultImage, "Marker NG", tmp, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
                    }
                    continue;
                }

                // ROI를 둘러싼 사각형을 이미지에 그리기
                rectangle(resultImage, boundingRect, textColor, 2);
            }
            else if (area >= cfg->MinContourArea / 3 && area <= cfg->MaxContourArea * 3) {
                printf("    [DEBUG] %zu contours area is not bondary min : %zu , max : %zu \n", area, cfg->MinContourArea, cfg->MaxContourArea);
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

        // 점선 사각형 그리기 (camera.cpp의 drawDashedRectangle 로직 간소화)
        rectangle(resultImage, Point(leftTop, leftTop), Point(rightBot, rightBot), Scalar(255, 0, 0), 2);

        // 결과 이미지 저장
        save_result_image_opencv(filepath, result->result_filename, result , resultImage);

        free_image_data(image_data);

        printf("    [DEBUG] ProcessImageFromFile_Mode1 completed successfully\n");
        printf("    [DEBUG] totalCount : %d ,   Total detections: %d, Success count: %d\n", totalCount, result->total_detections, result->success_count);
        printf("    [DEBUG] totalArea : %f ,   Total area/count: %d, abs(totalProductNum_as_is - totalProductNum) count: %d\n", totalArea, totalArea / (double)(35000), abs(totalArea / (double)(35000) - totalCount));
        

        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in ProcessImageFromFile_Mode1: " << e.what() << endl;
        if (image_data) free_image_data(image_data);
        return 0;
    }
}

// 모드 2 (3rd detection mode) - camera.cpp의 CaptureImage3rd 로직 이식
extern "C" int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result) {
    void* image_data = nullptr;

    try {
        // 이미지 로드
        if (!load_image_opencv(filepath, &image_data)) {
            printf("    [ERROR] Failed to load image\n");
            return 0;
        }

        ImageData* data = static_cast<ImageData*>(image_data);
        Mat& ori = data->original;
        Mat& resultMat = data->result;

        printf("    [DEBUG] Processing Mode 2 (3rd detection): %dx%d\n", ori.cols, ori.rows);

        // camera.cpp의 CaptureImage3rd에서 사용하는 크롭 영역 설정
        int buffer = 25;
        int buffer_y = 45;
        int comSize = 270; // 10M 25mm일때 187
        int startPointX[2] = { 83, 1585 };
        int startPointY[2] = { 140 - buffer_y, 1600 - buffer_y };
        buffer_y = buffer_y + buffer_y;

        // 크롭할 영역들을 사전에 정의 (20개의 Rect)
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

        // 크롭된 이미지를 저장할 벡터
        vector<Mat> croppedImages;
        string detectedResult[20];

        result->total_detections = 20;
        result->success_count = 0;

        // 크롭 영역에 대해 이미지 크롭 및 영역을 결과 이미지에 그리기
        for (int i = 0; i < cropRegions.size(); ++i) {
            cfg->ProductNum = i;
            // 이미지에서 해당 영역을 크롭하여 Mat에 저장
            Mat cropped = ori(cropRegions[i]);

            // 색상 감지 및 결과 확인
            int buffer = 5; // 버퍼 크기 설정
            Point cropCenter;
            int radius;
            Point secondCropCenter;
            int secondRadius = 0;
            Scalar OnStickerColor;
            cv::Point cvtCenter;

            bool isColorMarkDetected = false;
            int MarkAreaSum = 0;
            int MarkAreaNow;
            // cfg->ProductNum 1~20 값
            detectedResult[i] = "NG";

            // 각 영역(i)에 대해 해당하는 target_color_id 확인
            int target_color_id = cfg->productNumColor[i]; // i번째 영역의 색상 ID
            printf("    [DEBUG] Area[%d]: target_color_id=%d\n", i, target_color_id);

            for (int j = 0; j < cfg->maxColorCount; j++) { // 컬러 스캔

                HsvColor hsvColor = cfg->hsvColors[j];
                if (cfg->productNumColor[i] != hsvColor.id) {
                    continue;
                }
                cout << "HsvColor 검사 시작 ColorNum." << j << " : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << "," << hsvColor.id << endl;
                cout << "검사 시작 productNum : " << i + 1 << ", productNumColor : " << cfg->productNumColor[i] << endl;

                // 유효한 HSV 값인지 확인 (이 예시에서는 모든 h, s, v 값이 0이 아니면 유효하다고 가정)
                if (hsvColor.h <= 0 || hsvColor.h > 180 || hsvColor.s <= 0 || hsvColor.s > 255 || hsvColor.v <= 0 || hsvColor.v > 255) {
                    cout << "HsvColor 값 이상 : " << hsvColor.h << "," << hsvColor.s << "," << hsvColor.v << endl;
                    continue;  // HSV 값이 유효하지 않으면 다음으로 넘어감
                }

                // HSV 값을 Scalar로 변환
                Scalar targetHsv(hsvColor.h, hsvColor.s, hsvColor.v);
                int productNumColor = cfg->productNumColor[i];

                // 색상 감지
                bool detected = detectColorRegion3rd(cropped, *cfg, productNumColor, targetHsv, cropCenter, radius, MarkAreaNow, secondCropCenter, secondRadius);

                if (detected) {
                    cout << "해당 색상 스티커 감지되었습니다." << endl;
                    cout << "중심: (" << cropCenter.x << ", " << cropCenter.y << "), 반지름: " << radius << endl;
                    MarkAreaSum = MarkAreaNow + MarkAreaSum;

                    // HSV를 BGR로 변환
                    Mat hsvMat(1, 1, CV_8UC3, targetHsv);  // 1x1 크기의 Mat 생성 후 HSV 값 할당
                    Mat bgrMat;
                    cvtColor(hsvMat, bgrMat, COLOR_HSV2BGR);  // HSV -> BGR 변환

                    // BGR 색상 추출
                    Vec3b bgr = bgrMat.at<Vec3b>(0, 0);
                    OnStickerColor = Scalar(bgr[0], bgr[1], bgr[2]);

                    // 감지된 색상으로 원 그리기
                    circle(cropped, cropCenter, radius + 8, OnStickerColor, 3);

                    if (secondRadius != 0) {
                        circle(cropped, secondCropCenter, secondRadius + 8, OnStickerColor, 3);
                    }

                    detectedResult[i] = "OK";
                    isColorMarkDetected = true;
                    break;
                }
                else
                {

                }
            }

            // 컬러 감지 영역 합산 계산 코드
            if (MarkAreaSum >= cfg->minSUMMarkArea && MarkAreaSum < cfg->maxSUMMarkArea) {
                isColorMarkDetected = true;
                detectedResult[i] = "OK";
                cout << "min:" << cfg->minSUMMarkArea << ", max : " << cfg->maxSUMMarkArea << endl;
                cout << "***************** Mark Area Sum : " << MarkAreaSum << "*******************" << endl;
            }
            else {
                cout << "-------- Mark Area Sum : " << MarkAreaSum << "--------" << endl;
            }

            // 블랙 마커 감지 코드
            if (cfg->productNumColor[i] == 7) { // 7이 블랙
                bool isBlackMarkDetected = detectBlackArea3rd(cropped, buffer, cropCenter, radius, *cfg);
                if (isBlackMarkDetected) {
                    isColorMarkDetected = true;
                    detectedResult[i] = "OK";
                    cout << "Black Marker 감지 완료 " << MarkAreaSum << "" << endl;
                    circle(cropped, cropCenter, radius + 8, Scalar(255, 255, 255), 3);
                }
                else {
                    detectedResult[i] = "BK.NG";
                    cout << "Black Marker 감지 안됨 " << MarkAreaSum << "" << endl;
                }
            }

            // 검사 기능 OFF 설정
            if (cfg->isDetect[i] == 0) {
                detectedResult[i] = "OFF";
                cout << "검사 켜짐 제외" << endl;
            }

            // 벡터에 크롭된 이미지 저장
            croppedImages.push_back(cropped);

            // 크롭할 영역을 원본 이미지에 그리기 (파란색 사각형, 두께 2)
            rectangle(resultMat, cropRegions[i], Scalar(255, 0, 0), 2);

            // 검출 결과 저장
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

        // 결과 이미지 저장
        save_result_image_opencv(filepath, result->result_filename, result , resultMat);

        free_image_data(image_data);
        return 1;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Error in ProcessImageFromFile_Mode2: " << e.what() << endl;
        if (image_data) free_image_data(image_data);
        return 0;
    }
}

// 결과 이미지 저장 (실제 구현)
extern "C" int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result, const Mat& processedImage) {
    try {
        // 원본 이미지 로드
        Mat originalImage = imread(input_path, IMREAD_COLOR);
        if (originalImage.empty()) {
            return 0;
        }

        //Mat resultImage = originalImage.clone();
        // 전달받은 processedImage를 복사해서 사용
        Mat resultImage = processedImage.clone();

        if (result->mode == 1) {
            // 모드 1: 단일 검출 결과 표시
            /*
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
            */
        }
        else if (result->mode == 2) {
            // 모드 2: 20개 영역 검출 결과 표시
            for (int i = 0; i < 20; i++) {
                DetectionResult* det = &result->detections[i];

                Scalar color;
                if (strcmp(det->result_msg, "OK") == 0) {
                    color = Scalar(0, 255, 0); // 초록색
                }
                else if (strcmp(det->result_msg, "OFF") == 0) {
                    color = Scalar(0, 255, 255); // 노란색
                }
                else {
                    color = Scalar(0, 0, 255); // 빨간색
                }

                // 영역 번호와 결과 텍스트 표시
                string label = string(det->result_msg) + to_string(i + 1);
                Point textPos(det->center_x - 20, det->center_y - 20);
                putText(resultImage, label, textPos, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

                if (det->detected) {
                    circle(resultImage, Point(det->center_x, det->center_y), det->radius + 8, color, 3);
                }
            }
        }

        // 처리 시간 표시
        string timeInfo = "Time: " + to_string(result->processing_time) + "s";
        putText(resultImage, timeInfo, Point(10, resultImage.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // 성공률 표시
        string successInfo = "totalCount : " +to_string(totalCount) + "  total_detections : " + to_string(result->total_detections) + "  Success: " + to_string(result->success_count) + " / " + to_string(result->total_detections);
        putText(resultImage, successInfo, Point(10, resultImage.rows - 50),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string areaInfo = "totalArea : " + to_string(totalArea) + ", Total area / count : " + to_string((int)(totalArea / (double)(35000))) +", abs(totalProductNum_as_is - totalProductNum)  : "+ to_string(abs((int)(totalArea / (double)(35000)) - totalCount));
        putText(resultImage, areaInfo, Point(10, resultImage.rows - 80),
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

// 기타 함수들은 사용하지 않음 (더미 구현)
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
