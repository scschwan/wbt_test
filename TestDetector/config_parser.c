#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "config_parser.h"

// 안전한 isspace 래퍼 함수
static int safe_isspace(int c) {
    if (c < 0 || c > 255) return 0;
    return isspace(c);
}

// 문자열 트림 함수 (앞뒤 공백 제거) - 완전히 안전한 버전
static void trim_string(char* str) {
    if (!str) return;

    char* start = str;
    char* end;

    // 앞쪽 공백 제거
    while (*start && safe_isspace((unsigned char)*start)) {
        start++;
    }

    // 빈 문자열인 경우
    if (*start == 0) {
        *str = 0;
        return;
    }

    // 뒤쪽 공백 제거
    end = start + strlen(start) - 1;
    while (end > start && safe_isspace((unsigned char)*end)) {
        end--;
    }

    // 결과 복사
    size_t len = end - start + 1;
    if (len > 0) {
        memmove(str, start, len);
        str[len] = 0;
    }
    else {
        *str = 0;
    }
}

// 키=값 형태의 라인 파싱
static int parse_key_value(const char* line, char* key, char* value, size_t key_size, size_t value_size) {
    const char* equals = strchr(line, '=');
    if (equals == NULL) return 0;

    // 키 추출
    size_t key_len = equals - line;
    if (key_len >= key_size) return 0;
    strncpy_s(key, key_size, line, key_len);
    key[key_len] = 0;
    trim_string(key);

    // 값 추출
    strcpy_s(value, value_size, equals + 1);
    trim_string(value);

    return 1;
}

// HSV 색상 배열 파싱 (예: "116,171,116,0;164,145,244,1;5,179,71,2")
int parse_hsv_colors(const char* line, HsvColor* colors, int max_count) {
    char* line_copy = _strdup(line);
    char* context = NULL;
    char* token = strtok_s(line_copy, ";", &context);
    int count = 0;

    while (token != NULL && count < max_count) {
        int h, s, v, id;
        if (sscanf_s(token, "%d,%d,%d,%d", &h, &s, &v, &id) == 4) {
            colors[count].h = h;
            colors[count].s = s;
            colors[count].v = v;
            colors[count].id = id;
            count++;
        }
        token = strtok_s(NULL, ";", &context);
    }

    free(line_copy);
    return count;
}

// 제품별 색상 매핑 파싱 (예: "1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6")
int parse_product_color_mapping(const char* line, int* mapping, int max_count) {
    char* line_copy = _strdup(line);
    char* context = NULL;
    char* token = strtok_s(line_copy, ",", &context);
    int count = 0;

    while (token != NULL && count < max_count) {
        mapping[count] = atoi(token);
        count++;
        token = strtok_s(NULL, ",", &context);
    }

    free(line_copy);
    return count;
}

// 검사 활성화 플래그 파싱 (예: "1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
int parse_detection_flags(const char* line, int* flags, int max_count) {
    char* line_copy = _strdup(line);
    char* context = NULL;
    char* token = strtok_s(line_copy, ",", &context);
    int count = 0;

    while (token != NULL && count < max_count) {
        flags[count] = atoi(token);
        count++;
        token = strtok_s(NULL, ",", &context);
    }

    free(line_copy);
    return count;
}

// 설정 파일 읽기 (더 안전한 버전)
int read_config_file(const char* filepath, Modoo_cfg* cfg) {
    FILE* fp;
    errno_t err = fopen_s(&fp, filepath, "r");
    if (err != 0 || fp == NULL) {
        printf("Cannot open config file: %s (error code: %d)\n", filepath, err);
        return 0;
    }

    char line[512];
    char key[128];
    char value[384];
    int line_number = 0;
    int successful_reads = 0;

    printf("Reading configuration from: %s\n", filepath);

    while (fgets(line, sizeof(line), fp)) {
        line_number++;

        // 라인 길이 체크
        if (strlen(line) >= sizeof(line) - 1) {
            printf("Warning: Line %d too long, skipping\n", line_number);
            continue;
        }

        // 주석 및 빈 줄 제거
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }

        // 개행 문자 제거 (안전한 방식)
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
            len--;
        }

        // 빈 라인 체크
        if (len == 0) continue;

        // 키=값 파싱
        if (!parse_key_value(line, key, value, sizeof(key), sizeof(value))) {
            printf("Warning: Cannot parse line %d: %s\n", line_number, line);
            continue;
        }

        // 설정값 적용 (원본 modoo.h 구조체 기준)
        if (strcmp(key, "VisionID") == 0) {
            int val = atoi(value);
            if (val >= 1 && val <= 10) {
                cfg->VisionID = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "ProductNum") == 0) {
            int val = atoi(value);
            if (val >= 1 && val <= 100) {
                cfg->ProductNum = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "detectionMode") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 2) {
                cfg->detectionMode = (DetectionMode)val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "triggerMode") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 1) {
                cfg->triggerMode = (TriggerMode)val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "MinContourArea") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 1000000) {
                cfg->MinContourArea = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "MaxContourArea") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 1000000) {
                cfg->MaxContourArea = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "minMarkArea") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 100000) {
                cfg->minMarkArea = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "maxMarkArea") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 100000) {
                cfg->maxMarkArea = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "HsvBufferH") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 180) {
                cfg->HsvBufferH = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "HsvBufferS") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 255) {
                cfg->HsvBufferS = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "HsvBufferV") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 255) {
                cfg->HsvBufferV = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "BlackTagNum") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 10) {
                cfg->BlackTagNum = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "binaryValue") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 255) {
                cfg->binaryValue = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "debugMode") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 1) {
                cfg->debugMode = val;
                successful_reads++;
            }
        }
        else if (strcmp(key, "hsvEnable") == 0) {
            int val = atoi(value);
            if (val >= 0 && val <= 1) {
                cfg->hsvEnable = val;
                successful_reads++;
            }
        }
        // 복잡한 설정들은 일단 제외 (HSV 배열 등)
        else {
            printf("Info: Skipping unknown or complex key '%s' at line %d\n", key, line_number);
        }
    }

    fclose(fp);
    printf("Configuration loaded: %d settings applied from %d lines\n", successful_reads, line_number);
    return (successful_reads > 0) ? 1 : 0;
}

// 설정 파일 쓰기
int write_config_file(const char* filepath, Modoo_cfg* cfg) {
    FILE* fp;
    errno_t err = fopen_s(&fp, filepath, "w");
    if (err != 0 || fp == NULL) {
        printf("Cannot create config file: %s\n", filepath);
        return 0;
    }

    fprintf(fp, "# Test Detector Configuration File\n");
    fprintf(fp, "# Generated automatically\n\n");

    fprintf(fp, "# Basic Settings\n");
    fprintf(fp, "VisionID=%d\n", cfg->VisionID);
    fprintf(fp, "ProductNum=%d\n", cfg->ProductNum);
    fprintf(fp, "detectionMode=%d\n", cfg->detectionMode);
    fprintf(fp, "\n");

    fprintf(fp, "# Contour Detection Thresholds\n");
    fprintf(fp, "MinContourArea=%d\n", cfg->MinContourArea);
    fprintf(fp, "MaxContourArea=%d\n", cfg->MaxContourArea);
    fprintf(fp, "\n");

    fprintf(fp, "# Color Mark Detection\n");
    fprintf(fp, "minMarkArea=%d\n", cfg->minMarkArea);
    fprintf(fp, "maxMarkArea=%d\n", cfg->maxMarkArea);
    fprintf(fp, "minSUMMarkArea=%d\n", cfg->minSUMMarkArea);
    fprintf(fp, "maxSUMMarkArea=%d\n", cfg->maxSUMMarkArea);
    fprintf(fp, "\n");

    fprintf(fp, "# Black Mark Detection\n");
    fprintf(fp, "blackMinMarkArea=%d\n", cfg->blackMinMarkArea);
    fprintf(fp, "blackMaxMarkArea=%d\n", cfg->blackMaxMarkArea);
    fprintf(fp, "BlackEllipseMinSize=%d\n", cfg->BlackEllipseMinSize);
    fprintf(fp, "BlackEllipseMaxSize=%d\n", cfg->BlackEllipseMaxSize);
    fprintf(fp, "BlackTagNum=%d\n", cfg->BlackTagNum);
    fprintf(fp, "\n");

    fprintf(fp, "# HSV Color Tolerance\n");
    fprintf(fp, "HsvBufferH=%d\n", cfg->HsvBufferH);
    fprintf(fp, "HsvBufferS=%d\n", cfg->HsvBufferS);
    fprintf(fp, "HsvBufferV=%d\n", cfg->HsvBufferV);
    fprintf(fp, "\n");

    fprintf(fp, "# Image Processing\n");
    fprintf(fp, "binaryValue=%d\n", cfg->binaryValue);
    fprintf(fp, "hsvEnable=%d\n", cfg->hsvEnable);
    fprintf(fp, "\n");

    fprintf(fp, "# Debug Settings\n");
    fprintf(fp, "debugMode=%d\n", cfg->debugMode);
    fprintf(fp, "\n");

    fprintf(fp, "# Color Configuration\n");
    fprintf(fp, "maxColorCount=%d\n", cfg->maxColorCount);

    // HSV 색상 배열 출력
    if (cfg->maxColorCount > 0) {
        fprintf(fp, "hsvColors=");
        for (int i = 0; i < cfg->maxColorCount; i++) {
            fprintf(fp, "%d,%d,%d,%d", cfg->hsvColors[i].h, cfg->hsvColors[i].s,
                cfg->hsvColors[i].v, cfg->hsvColors[i].id);
            if (i < cfg->maxColorCount - 1) fprintf(fp, ";");
        }
        fprintf(fp, "\n");
    }

    // 제품별 색상 매핑 출력
    fprintf(fp, "productNumColor=");
    for (int i = 0; i < 20; i++) {
        fprintf(fp, "%d", cfg->productNumColor[i]);
        if (i < 19) fprintf(fp, ",");
    }
    fprintf(fp, "\n");

    // 검사 활성화 플래그 출력
    fprintf(fp, "isDetect=");
    for (int i = 0; i < 20; i++) {
        fprintf(fp, "%d", cfg->isDetect[i]);
        if (i < 19) fprintf(fp, ",");
    }
    fprintf(fp, "\n");

    fclose(fp);
    printf("Configuration saved to: %s\n", filepath);
    return 1;
}

// 설정 검증
int validate_config(Modoo_cfg* cfg) {
    int errors = 0;

    if (cfg->VisionID < 1 || cfg->VisionID > 10) {
        printf("Error: VisionID must be between 1-10\n");
        errors++;
    }

    if (cfg->MinContourArea >= cfg->MaxContourArea) {
        printf("Error: MinContourArea must be less than MaxContourArea\n");
        errors++;
    }

    if (cfg->minMarkArea >= cfg->maxMarkArea) {
        printf("Error: minMarkArea must be less than maxMarkArea\n");
        errors++;
    }

    if (cfg->HsvBufferH < 0 || cfg->HsvBufferH > 180) {
        printf("Error: HsvBufferH must be between 0-180\n");
        errors++;
    }

    if (cfg->binaryValue < 0 || cfg->binaryValue > 255) {
        printf("Error: binaryValue must be between 0-255\n");
        errors++;
    }

    return (errors == 0);
}

// 설정 출력
void print_config(Modoo_cfg* cfg) {
    printf("\n=== Current Configuration ===\n");
    printf("VisionID: %d\n", cfg->VisionID);
    printf("ProductNum: %d\n", cfg->ProductNum);
    printf("detectionMode: %d\n", cfg->detectionMode);
    printf("MinContourArea: %d\n", cfg->MinContourArea);
    printf("MaxContourArea: %d\n", cfg->MaxContourArea);
    printf("minMarkArea: %d\n", cfg->minMarkArea);
    printf("maxMarkArea: %d\n", cfg->maxMarkArea);
    printf("HsvBufferH: %d\n", cfg->HsvBufferH);
    printf("HsvBufferS: %d\n", cfg->HsvBufferS);
    printf("HsvBufferV: %d\n", cfg->HsvBufferV);
    printf("binaryValue: %d\n", cfg->binaryValue);
    printf("debugMode: %d\n", cfg->debugMode);
    printf("maxColorCount: %d\n", cfg->maxColorCount);

    if (cfg->maxColorCount > 0) {
        printf("HSV Colors:\n");
        for (int i = 0; i < cfg->maxColorCount; i++) {
            printf("  [%d] H:%d S:%d V:%d ID:%d\n", i,
                cfg->hsvColors[i].h, cfg->hsvColors[i].s,
                cfg->hsvColors[i].v, cfg->hsvColors[i].id);
        }
    }

    printf("============================\n\n");
}