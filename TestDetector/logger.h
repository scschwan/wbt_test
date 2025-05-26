#ifndef LOGGER_H
#define LOGGER_H

#include "image_result.h"

#ifdef __cplusplus
extern "C" {
#endif

	// 로그 시스템 초기화 및 정리
	int initialize_logger();
	void cleanup_logger();

	// 로그 기록 함수들
	void log_message(const char* message);
	void log_image_result(ImageProcessResult* result);
	void log_error(const char* error_message);

	// 보고서 생성
	void generate_summary_report();

#ifdef __cplusplus
}
#endif

#endif // LOGGER_H