// logger.c에 추가해야 할 함수들

#include <stdio.h>
#include <time.h>
#include <string.h>
#include "logger.h"
#include "image_result.h"

static FILE* log_file = NULL;

// 로거 초기화
int initialize_logger() {
    if (fopen_s(&log_file, "results/processing_log.txt", "w") != 0) {
        printf("Warning: Could not create log file\n");
        return 0;
    }

    if (log_file) {
        time_t now = time(NULL);
        char time_str[100];
        ctime_s(time_str, sizeof(time_str), &now);
        time_str[strlen(time_str) - 1] = '\0'; // 개행 제거

        fprintf(log_file, "=== Processing Log Started at %s ===\n", time_str);
        fflush(log_file);
    }

    return 1;
}

// 로그 메시지 기록
void log_message(const char* message) {
    time_t now = time(NULL);
    char time_str[100];
    ctime_s(time_str, sizeof(time_str), &now);
    time_str[strlen(time_str) - 1] = '\0'; // 개행 제거

    // 콘솔에 출력
    printf("[%s] %s\n", time_str, message);

    // 파일에 기록
    if (log_file) {
        fprintf(log_file, "[%s] %s\n", time_str, message);
        fflush(log_file);
    }
}

// 이미지 처리 결과 로그
void log_image_result(ImageProcessResult* result) {
    if (!result) return;

    char log_msg[1024];
    snprintf(log_msg, sizeof(log_msg),
        "Image: %s | Mode: %d | Success: %d/%d | Time: %.3fs | Result: %s",
        result->original_filename,
        result->mode,
        result->success_count,
        result->total_detections,
        result->processing_time,
        result->final_result
    );

    log_message(log_msg);
}

// 요약 리포트 생성
void generate_summary_report() {
    if (log_file) {
        fprintf(log_file, "\n=== Processing Summary ===\n");
        fprintf(log_file, "Check individual log entries above for detailed results\n");
        fprintf(log_file, "========================\n");
        fflush(log_file);
    }

    printf("Summary report generated in results/processing_log.txt\n");
}

// 로거 정리
void cleanup_logger() {
    if (log_file) {
        time_t now = time(NULL);
        char time_str[100];
        ctime_s(time_str, sizeof(time_str), &now);
        time_str[strlen(time_str) - 1] = '\0'; // 개행 제거

        fprintf(log_file, "=== Processing Log Ended at %s ===\n", time_str);
        fclose(log_file);
        log_file = NULL;
    }
}