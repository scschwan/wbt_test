// image_result.c에 추가해야 할 함수들

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "image_result.h"
#include "simple_types.h"

// 파일명에서 확장자를 제거한 파일명 추출
void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size) {
    const char* lastSlash = strrchr(filepath, '\\');
    if (!lastSlash) lastSlash = strrchr(filepath, '/');

    const char* basename = lastSlash ? lastSlash + 1 : filepath;

    // 확장자 제거
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

// 결과 이미지 파일명 생성
void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size) {
    char filename[256];
    extract_filename_without_ext(original_path, filename, sizeof(filename));

    // results 폴더에 _result 접미사 추가하여 저장
    snprintf(result_path, result_path_size, "results/%s_result.jpg", filename);
}

// 더미 구현 함수들 (사용하지 않지만 컴파일을 위해 필요)
int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg) {
    return 1; // 더미 구현
}

int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg) {
    return 1; // 더미 구현
}