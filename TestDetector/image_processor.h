#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "simple_types.h"
#include "image_result.h"

#ifdef __cplusplus
extern "C" {
#endif

    // ���� �̹��� ó�� �Լ��� (camera.cpp�� �Լ����� ��ü)
    int ProcessImageFromFile_Mode1(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result);
    int ProcessImageFromFile_Mode2(const char* filepath, Modoo_cfg* cfg, ImageProcessResult* result);

    // OpenCV ��� �̹��� ó�� �Լ���
    int load_image_opencv(const char* filepath, void** image_data);
    //int save_result_image_opencv(const char* input_path, const char* output_path, ImageProcessResult* result, const Mat* processedImage);

    // ���� ���� �Լ��� (camera.cpp���� �̽�)
    int detect_color_region(void* image_data, Modoo_cfg* cfg, int product_color,
        int* center_x, int* center_y, int* radius, int* area);
    int detect_black_area(void* image_data, Modoo_cfg* cfg,
        int* center_x, int* center_y, int* radius);

    // �̹��� ��ó�� �Լ���
    int convert_to_binary(void* image_data, int threshold);
    int find_contours(void* image_data, Modoo_cfg* cfg, DetectionResult results[], int* count);

    // �޸� ����
    void free_image_data(void* image_data);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_PROCESSOR_H