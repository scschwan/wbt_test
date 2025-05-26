#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include "image_result.h"



// ��� ���ϸ� ���� (������_result.Ȯ����)
void generate_result_filename(const char* original_path, char* result_path, size_t result_path_size) {
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    // ��� ����
    _splitpath_s(original_path, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);

    // results ������ ����
    sprintf_s(result_path, result_path_size, "results\\%s_result%s", fname, ext);
}

// Ȯ���� ���� ���ϸ� ����
void extract_filename_without_ext(const char* filepath, char* filename, size_t filename_size) {
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    _splitpath_s(filepath, NULL, 0, NULL, 0, fname, _MAX_FNAME, ext, _MAX_EXT);
    strcpy_s(filename, filename_size, fname);
}

// ��� 1�� ��� �̹��� ���� (���� ����)
int save_result_image_mode1(const char* original_path, DetectionResult* result, Modoo_cfg* cfg) {
    char result_filename[512];
    generate_result_filename(original_path, result_filename, sizeof(result_filename));

    // ���� ���������� OpenCV�� ����ؼ�:
    // 1. ���� �̹��� �ε�
    // 2. ����� ������ �� �׸���
    // 3. �ؽ�Ʈ �� �߰� (OK/NG, ��ǥ, ���� ũ��)
    // 4. ��� �̹��� ����

    printf("    [DUMMY] Saving result image: %s\n", result_filename);
    printf("    [DUMMY] Drawing circle at (%d, %d) with radius %d\n",
        result->center_x, result->center_y, result->radius);
    printf("    [DUMMY] Adding label: %s, Area: %d\n", result->result_msg, result->area);

    // ���� ���� ���� (�����δ� �̹��� ����)
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

// ��� 2�� ��� �̹��� ���� (���� ����)
int save_result_image_mode2(const char* original_path, DetectionResult results[], int count, Modoo_cfg* cfg) {
    char result_filename[512];
    generate_result_filename(original_path, result_filename, sizeof(result_filename));

    // ���� ���������� OpenCV�� ����ؼ�:
    // 1. ���� �̹��� �ε�
    // 2. 20�� ������ �簢�� �׸���
    // 3. �� ������ ���� ��� �� �߰�
    // 4. ���󺰷� �ٸ� ����� ǥ��
    // 5. ��� �̹��� ����

    printf("    [DUMMY] Saving 3rd mode result image: %s\n", result_filename);

    // ���� ���� ����
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