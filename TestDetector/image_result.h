#ifndef IMAGE_RESULT_H
#define IMAGE_RESULT_H

#include "simple_types.h"

// 검출 결과 구조체
typedef struct {
    int detected;           // 검출 여부 (1: 검출됨, 0: 검출 안됨)
    int center_x, center_y; // 중심 좌표
    int radius;             // 반지름
    int area;               // 영역 크기
    char result_msg[256];   // 결과 메시지 ("OK", "NG", "OFF" 등)
    char color_info[64];    // 색상 정보
} DetectionResult;

// 이미지 처리 결과 구조체
typedef struct {
    char original_filename[256];    // 원본 파일명
    char result_filename[256];      // 결과 파일명
    int mode;                       // 검사 모드 (1 or 2)
    int total_detections;           // 총 검출 개수
    int success_count;              // 성공 개수
    DetectionResult detections[20]; // 최대 20개 검출 결과 (3rd 모드용)
    char final_result[512];         // 최종 결과 메시지
    double processing_time;         // 처리 시간 (초)
} ImageProcessResult;

#ifdef __cplusplus
extern "C" {
#endif

    // 결과 이미지 저장 함수들
    int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg);
    int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg);

    // 파일명 유틸리티
    void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size);
    void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_RESULT_H