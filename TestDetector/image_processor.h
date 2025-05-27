#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "simple_types.h"
#include "image_result.h"

#ifdef __cplusplus
extern "C" {
#endif

    // 메인 이미지 처리 함수들 (camera.cpp의 함수들을 대체)
    int ProcessImageFromFile_Mode1(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result);
    int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result);

    // OpenCV 기반 이미지 처리 함수들
    int load_image_opencv(const char* filepath, void** image_data);
    //int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result, const Mat* processedImage);

    // 색상 검출 함수들 (camera.cpp에서 이식)
    int detect_color_region(void* image_data, Modoo_cfg* cfg, int product_color,
        int* center_x, int* center_y, int* radius, int* area);
    int detect_black_area(void* image_data, Modoo_cfg* cfg,
        int* center_x, int* center_y, int* radius);

    // 이미지 전처리 함수들
    int convert_to_binary(void* image_data, int threshold);
    int find_contours(void* image_data, Modoo_cfg* cfg, DetectionResult results[], int* count);

    // 메모리 관리
    void free_image_data(void* image_data);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_PROCESSOR_H