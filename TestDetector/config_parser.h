#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "simple_types.h"

#ifdef __cplusplus
extern "C" {
#endif

	// ���� ���� �б�/����
	int read_config_file(const char* filepath, Modoo_cfg* cfg);
	int write_config_file(const char* filepath, Modoo_cfg* cfg);

	// ���� ���� �Ľ�
	int parse_hsv_colors(const char* line, HsvColor* colors, int max_count);
	int parse_product_color_mapping(const char* line, int* mapping, int max_count);
	int parse_detection_flags(const char* line, int* flags, int max_count);

	// ���� ����
	int validate_config(Modoo_cfg* cfg);
	void print_config(Modoo_cfg* cfg);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_PARSER_H