#ifndef IMAGE_RESULT_H
#define IMAGE_RESULT_H

#include "simple_types.h"

// ���� ��� ����ü
typedef struct {
    int detected;           // ���� ���� (1: �����, 0: ���� �ȵ�)
    int center_x, center_y; // �߽� ��ǥ
    int radius;             // ������
    int area;               // ���� ũ��
    char result_msg[256];   // ��� �޽��� ("OK", "NG", "OFF" ��)
    char color_info[64];    // ���� ����
} DetectionResult;

// �̹��� ó�� ��� ����ü
typedef struct {
    char original_filename[256];    // ���� ���ϸ�
    char result_filename[256];      // ��� ���ϸ�
    int mode;                       // �˻� ��� (1 or 2)
    int total_detections;           // �� ���� ����
    int success_count;              // ���� ����
    DetectionResult detections[20]; // �ִ� 20�� ���� ��� (3rd ����)
    char final_result[512];         // ���� ��� �޽���
    double processing_time;         // ó�� �ð� (��)
} ImageProcessResult;

#ifdef __cplusplus
extern "C" {
#endif

    // ��� �̹��� ���� �Լ���
    int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg);
    int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg);

    // ���ϸ� ��ƿ��Ƽ
    void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size);
    void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_RESULT_H