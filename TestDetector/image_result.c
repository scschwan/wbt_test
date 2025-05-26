#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include "image_result.h"



// 결과 파일명 생성 (원본명_result.확장자)
void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size) {
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    // 경로 분해
    _splitpath_s(original_path, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);

    // results 폴더에 저장
    sprintf_s(result_path, result_path_size, "results\\%s_result%s", fname, ext);
}

// 확장자 없는 파일명 추출
void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size) {
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    _splitpath_s(filepath, NULL, 0, NULL, 0, fname, _MAX_FNAME, ext, _MAX_EXT);
    strcpy_s(filename, filename_size, fname);
}

// 모드 1용 결과 이미지 저장 (더미 구현)
int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg) {
    char result_filename[512];
    generate_result_filename(original_path, result_filename, sizeof(result_filename));

    // 실제 구현에서는 OpenCV를 사용해서:
    // 1. 원본 이미지 로드
    // 2. 검출된 영역에 원 그리기
    // 3. 텍스트 라벨 추가 (OK/NG, 좌표, 영역 크기)
    // 4. 결과 이미지 저장

    printf("    [DUMMY] Saving result image: %s\n", result_filename);
    printf("    [DUMMY] Drawing circle at (%d, %d) with radius %d\n",
        result->center_x, result->center_y, result->radius);
    printf("    [DUMMY] Adding label: %s, Area: %d\n", result->result_msg, result->area);

    // 더미 파일 생성 (실제로는 이미지 파일)
    FILE* dummy_file = fopen(result_filename, "w");
    if (dummy_file) {
        fprintf(dummy_file, "Result image for: %s\n", original_path);
        fprintf(dummy_file, "Detection: %s\n", result->result_msg);
        fprintf(dummy_file, "Center: (%d, %d)\n", result->center_x, result->center_y);
        fprintf(dummy_file, "Area: %d\n", result->area);
        fclose(dummy_file);
        return 1;
    }

    return 0;
}

// 모드 2용 결과 이미지 저장 (더미 구현)
int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg) {
    char result_filename[512];
    generate_result_filename(original_path, result_filename, sizeof(result_filename));

    // 실제 구현에서는 OpenCV를 사용해서:
    // 1. 원본 이미지 로드
    // 2. 20개 영역에 사각형 그리기
    // 3. 각 영역에 검출 결과 라벨 추가
    // 4. 색상별로 다른 색깔로 표시
    // 5. 결과 이미지 저장

    printf("    [DUMMY] Saving 3rd mode result image: %s\n", result_filename);

    // 더미 파일 생성
    FILE* dummy_file = fopen(result_filename, "w");
    if (dummy_file) {
        fprintf(dummy_file, "3rd Mode Result image for: %s\n", original_path);
        fprintf(dummy_file, "Total areas: %d\n", count);

        for (int i = 0; i < count; i++) {
            fprintf(dummy_file, "Area %02d: %s - (%d,%d) Area:%d Color:%s\n",
                i + 1, results[i].result_msg,
                results[i].center_x, results[i].center_y,
                results[i].area, results[i].color_info);
        }

        fclose(dummy_file);
        return 1;
    }

    return 0;
}