// image_result.c�� �߰��ؾ� �� �Լ���

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "image_result.h"
#include "simple_types.h"

// ���ϸ��� Ȯ���ڸ� ������ ���ϸ� ����
void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size) {
    const char* lastSlash = strrchr(filepath, '\\');
    if (!lastSlash) lastSlash = strrchr(filepath, '/');

    const char* basename = lastSlash ? lastSlash + 1 : filepath;

    // Ȯ���� ����
    const char* lastDot = strrchr(basename, '.');
    if (lastDot) {
        size_t len = lastDot - basename;
        if (len >= filename_size) len = filename_size - 1;
        strncpy_s(filename, filename_size, basename, len);
        filename[len] = '\0';
    }
    else {
        strncpy_s(filename, filename_size, basename, filename_size - 1);
        filename[filename_size - 1] = '\0';
    }
}

// ��� �̹��� ���ϸ� ����
void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size) {
    char filename[256];
    extract_filename_without_ext(original_path, filename, sizeof(filename));

    // results ������ _result ���̻� �߰��Ͽ� ����
    snprintf(result_path, result_path_size, "results/%s_result.jpg", filename);
}

// ���� ���� �Լ��� (������� ������ �������� ���� �ʿ�)
int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg) {
    return 1; // ���� ����
}

int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg) {
    return 1; // ���� ����
}