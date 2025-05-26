#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "simple_types.h"

#ifdef __cplusplus
extern "C" {
#endif

	// 설정 파일 읽기/쓰기
	int read_config_file(const char* filepath, Modoo_cfg* cfg);
	int write_config_file(const char* filepath, Modoo_cfg* cfg);

	// 개별 설정 파싱
	int parse_hsv_colors(const char* line, HsvColor* colors, int max_count);
	int parse_product_color_mapping(const char* line, int* mapping, int max_count);
	int parse_detection_flags(const char* line, int* flags, int max_count);

	// 설정 검증
	int validate_config(Modoo_cfg* cfg);
	void print_config(Modoo_cfg* cfg);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_PARSER_H