#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <direct.h>
#include "logger.h"
#include "image_result.h"

static FILE* g_log_file = NULL;
static char g_log_filename[256];
static int g_total_processed = 0;
static int g_total_success = 0;

// �α� �ý��� �ʱ�ȭ
int initialize_logger() {
    // results ���� Ȯ��
    if (_access("results", 0) == -1) {
        if (_mkdir("results") != 0) {
            printf("Error: Cannot create results directory\n");
            return 0;
        }
    }

    // ���� �ð����� �α� ���ϸ� ����
    time_t now = time(NULL);
    struct tm* local_time = localtime(&now);

    sprintf_s(g_log_filename, sizeof(g_log_filename),
        "results/detection_log_%04d%02d%02d_%02d%02d%02d.txt",
        local_time->tm_year + 1900,
        local_time->tm_mon + 1,
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min,
        local_time->tm_sec);

    g_log_file = fopen(g_log_filename, "w");
    if (g_log_file == NULL) {
        printf("Error: Cannot create log file %s\n", g_log_filename);
        return 0;
    }

    // �α� ��� �ۼ�
    fprintf(g_log_file, "=== Test Detector Log ===\n");
    fprintf(g_log_file, "Start Time: %04d-%02d-%02d %02d:%02d:%02d\n",
        local_time->tm_year + 1900,
        local_time->tm_mon + 1,
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min,
        local_time->tm_sec);
    fprintf(g_log_file, "=============================\n\n");

    fflush(g_log_file);

    printf("Log file created: %s\n", g_log_filename);
    return 1;
}

// �Ϲ� �޽��� �α�
void log_message(const char* message) {
    if (g_log_file != NULL) {
        time_t now = time(NULL);
        struct tm* local_time = localtime(&now);

        fprintf(g_log_file, "[%02d:%02d:%02d] %s\n",
            local_time->tm_hour,
            local_time->tm_min,
            local_time->tm_sec,
            message);
        fflush(g_log_file);
    }

    // �ֿܼ��� ���
    printf("%s\n", message);
}

// �̹��� ó�� ��� �α�
void log_image_result(ImageProcessResult* result) {
    if (g_log_file == NULL) return;

    fprintf(g_log_file, "\n--- Image Processing Result ---\n");
    fprintf(g_log_file, "Original File: %s\n", result->original_filename);
    fprintf(g_log_file, "Result File: %s\n", result->result_filename);
    fprintf(g_log_file, "Mode: %d (%s)\n", result->mode,
        result->mode == 1 ? "Normal Detection" : "3rd Detection");
    fprintf(g_log_file, "Processing Time: %.3f seconds\n", result->processing_time);
    fprintf(g_log_file, "Total Detections: %d\n", result->total_detections);
    fprintf(g_log_file, "Success Count: %d\n", result->success_count);

    // ���� �� ����
    if (result->mode == 2) { // 3rd ����� ��� �� ����
        fprintf(g_log_file, "\nDetailed Results:\n");
        for (int i = 0; i < result->total_detections; i++) {
            DetectionResult* det = &result->detections[i];
            fprintf(g_log_file, "  [%02d] %s - Area: %d, Center: (%d, %d), Color: %s\n",
                i + 1, det->result_msg, det->area,
                det->center_x, det->center_y, det->color_info);
        }
    }

    fprintf(g_log_file, "Final Result: %s\n", result->final_result);
    fprintf(g_log_file, "------------------------------\n\n");

    fflush(g_log_file);

    // ��� ������Ʈ
    g_total_processed++;
    if (result->success_count > 0) {
        g_total_success++;
    }
}

// ���� �α�
void log_error(const char* error_message) {
    if (g_log_file != NULL) {
        time_t now = time(NULL);
        struct tm* local_time = localtime(&now);

        fprintf(g_log_file, "[%02d:%02d:%02d] ERROR: %s\n",
            local_time->tm_hour,
            local_time->tm_min,
            local_time->tm_sec,
            error_message);
        fflush(g_log_file);
    }

    printf("ERROR: %s\n", error_message);
}

// ��� ���� ����
void generate_summary_report() {
    if (g_log_file == NULL) return;

    time_t now = time(NULL);
    struct tm* local_time = localtime(&now);

    fprintf(g_log_file, "\n=== Processing Summary ===\n");
    fprintf(g_log_file, "End Time: %04d-%02d-%02d %02d:%02d:%02d\n",
        local_time->tm_year + 1900,
        local_time->tm_mon + 1,
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min,
        local_time->tm_sec);
    fprintf(g_log_file, "Total Processed: %d images\n", g_total_processed);
    fprintf(g_log_file, "Successful: %d images\n", g_total_success);
    fprintf(g_log_file, "Failed: %d images\n", g_total_processed - g_total_success);

    if (g_total_processed > 0) {
        double success_rate = (100.0 * g_total_success) / g_total_processed;
        fprintf(g_log_file, "Success Rate: %.1f%%\n", success_rate);
    }

    fprintf(g_log_file, "========================\n");
    fflush(g_log_file);

    // ���� ��� ���ϵ� ����
    char summary_filename[256];
    sprintf_s(summary_filename, sizeof(summary_filename), "results/summary_%04d%02d%02d_%02d%02d%02d.txt",
        local_time->tm_year + 1900,
        local_time->tm_mon + 1,
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min,
        local_time->tm_sec);

    FILE* summary_file = fopen(summary_filename, "w");
    if (summary_file != NULL) {
        fprintf(summary_file, "Test Detector Processing Summary\n");
        fprintf(summary_file, "================================\n\n");
        fprintf(summary_file, "Processing Date: %04d-%02d-%02d %02d:%02d:%02d\n",
            local_time->tm_year + 1900,
            local_time->tm_mon + 1,
            local_time->tm_mday,
            local_time->tm_hour,
            local_time->tm_min,
            local_time->tm_sec);
        fprintf(summary_file, "Total Images Processed: %d\n", g_total_processed);
        fprintf(summary_file, "Successful Detections: %d\n", g_total_success);
        fprintf(summary_file, "Failed Detections: %d\n", g_total_processed - g_total_success);

        if (g_total_processed > 0) {
            double success_rate = (100.0 * g_total_success) / g_total_processed;
            fprintf(summary_file, "Success Rate: %.1f%%\n", success_rate);
        }

        fprintf(summary_file, "\nDetailed log: %s\n", g_log_filename);
        fclose(summary_file);

        printf("Summary report created: %s\n", summary_filename);
    }
}

// �α� �ý��� ����
void cleanup_logger() {
    if (g_log_file != NULL) {
        fclose(g_log_file);
        g_log_file = NULL;
    }

    printf("Log file saved: %s\n", g_log_filename);
}
