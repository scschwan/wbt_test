#ifndef LOGGER_H
#define LOGGER_H

#include "image_result.h"

#ifdef __cplusplus
extern "C" {
#endif

	// �α� �ý��� �ʱ�ȭ �� ����
	int initialize_logger();
	void cleanup_logger();

	// �α� ��� �Լ���
	void log_message(const char* message);
	void log_image_result(ImageProcessResult* result);
	void log_error(const char* error_message);

	// ���� ����
	void generate_summary_report();

#ifdef __cplusplus
}
#endif

#endif // LOGGER_H