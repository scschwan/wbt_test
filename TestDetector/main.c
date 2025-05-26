#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>    // Windows용 mkdir
#include <io.h>        // Windows용 파일 접근
#include <windows.h>   // Windows API
#include <time.h>      // 시간 측정용

#include "simple_types.h"  // 이제 modoo.h와 동일
#include "logger.h"
#include "image_result.h"
#include "image_processor.h"
#include "config_parser.h"

// 함수 선언 (프로토타입)
void print_usage(const char* program_name);
int check_directories();
int count_image_files(const char* folder_path);
int load_simple_config(Modoo_cfg* cfg);
void create_default_config_file(Modoo_cfg* cfg);
int process_single_image(const char* image_path, Modoo_cfg* cfg, int mode);
void init_default_hsv_colors(Modoo_cfg* cfg);
// OpenCV 테스트 함수들
int test_opencv_basic();
int test_image_load(const char* filepath);  // 추가

void print_usage(const char* program_name) {
    printf("Usage: %s <mode>\n", program_name);
    printf("Modes:\n");
    printf("  1 : Normal detection mode (CaptureImage)\n");
    printf("  2 : 3rd detection mode (CaptureImage3rd)\n");
    printf("\nExample:\n");
    printf("  %s 1\n", program_name);
}

int check_directories() {
    // test_images 폴더 확인
    if (_access("test_images", 0) == -1) {
        printf("Creating test_images directory...\n");
        if (_mkdir("test_images") != 0) {
            printf("Error: Cannot create test_images directory\n");
            return 0;
        }
    }

    // results 폴더 확인
    if (_access("results", 0) == -1) {
        printf("Creating results directory...\n");
        if (_mkdir("results") != 0) {
            printf("Error: Cannot create results directory\n");
            return 0;
        }
    }

    // option 폴더 확인
    if (_access("option", 0) == -1) {
        printf("Warning: option directory not found. Creating...\n");
        if (_mkdir("option") != 0) {
            printf("Error: Cannot create option directory\n");
            return 0;
        }
    }

    return 1;
}

int count_image_files(const char* folder_path) {
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind;
    int count = 0;
    char search_path[260];

    sprintf_s(search_path, sizeof(search_path), "%s\\*.*", folder_path);

    hFind = FindFirstFileA(search_path, &findFileData);
    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Error: Cannot open directory %s\n", folder_path);
        return 0;
    }

    printf("Scanning images in %s:\n", folder_path);
    do {
        if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            char* ext = strrchr(findFileData.cFileName, '.');
            if (ext != NULL) {
                if (_stricmp(ext, ".jpg") == 0 || _stricmp(ext, ".jpeg") == 0 ||
                    _stricmp(ext, ".png") == 0 || _stricmp(ext, ".bmp") == 0) {
                    count++;
                    printf("  [%d] %s\n", count, findFileData.cFileName);
                }
            }
        }
    } while (FindNextFileA(hFind, &findFileData) != 0);

    FindClose(hFind);
    return count;
}

// 설정 파일 읽기 (원본 modoo.h 구조체 기준)
int load_simple_config(Modoo_cfg* cfg) {
    // 기본값 설정
    memset(cfg, 0, sizeof(Modoo_cfg));

    // 기본 설정값들
    cfg->triggerMode = TRIGGER_MODE_SW;
    cfg->detectionMode = DETECTION_MODE_ONLY_RULE;
    cfg->thresholdMode = THRESHOLD_MODE_BINARY;
    cfg->VisionID = 1;
    cfg->ProductNum = 1;
    cfg->PortNum = 9999;
    cfg->maxColorCount = 6;

    // 카메라 설정 (동적 할당)
    cfg->camModelName = (char*)malloc(256);
    cfg->camSerialNum = (char*)malloc(256);
    if (cfg->camModelName) strcpy_s(cfg->camModelName, 256, "TestCamera");
    if (cfg->camSerialNum) strcpy_s(cfg->camSerialNum, 256, "123456");

    // 기본 임계값들
    cfg->MinContourArea = 10000;
    cfg->MaxContourArea = 50000;
    cfg->minMarkArea = 100;
    cfg->maxMarkArea = 5000;
    cfg->minSUMMarkArea = 200;
    cfg->maxSUMMarkArea = 10000;

    cfg->blackMinMarkArea = 50;
    cfg->blackMaxMarkArea = 3000;
    cfg->BlackEllipseMinSize = 10;
    cfg->BlackEllipseMaxSize = 100;
    cfg->BlackTagNum = 7;

    cfg->HsvBufferH = 10;
    cfg->HsvBufferS = 50;
    cfg->HsvBufferV = 50;

    cfg->binaryValue = 128;
    cfg->hsvEnable = 1;
    cfg->debugMode = 1;

    // 좌표 변환 설정
    cfg->origin_vision_x = 780.0;
    cfg->origin_vision_y = 422.0;
    cfg->origin_robot_x = 196.0;
    cfg->origin_robot_y = 1.260;
    cfg->res_x = 0.02303163445;
    cfg->res_y = 0.0231194784;

    cfg->Max_vision_x = 1920;
    cfg->Max_vision_y = 1080;

    // 면적 임계값
    cfg->areaThresholdPercentLower = 0.8;
    cfg->areaThresholdPercentUpper = 1.2;

    // 기본 면적 배열 초기화
    for (int i = 0; i < MAX_AREAS; i++) {
        cfg->areas[i] = 35000; // 기본 면적값
    }

    // 기본 HSV 색상들 설정
    init_default_hsv_colors(cfg);

    // 제품별 색상 매핑 기본값
    for (int i = 0; i < 20; i++) {
        cfg->productNumColor[i] = (i % 6) + 1; // 1-6 순환
        cfg->isDetect[i] = 1; // 모든 검사 활성화
    }

    // 설정 파일 읽기 시도 (안전한 버전)
    printf("Attempting to load configuration file...\n");
    if (read_config_file("option/main.txt", cfg)) {
        printf("Configuration loaded from file successfully\n");

        // 설정 검증
        if (!validate_config(cfg)) {
            printf("Warning: Configuration validation failed, using default values\n");
            // 검증 실패 시 기본값 재설정
            init_default_hsv_colors(cfg);
        }
    }
    else {
        printf("Configuration file not found or read failed, creating default file\n");
        create_default_config_file(cfg);
    }

    // 디버그 모드면 설정 출력
    if (cfg->debugMode) {
        print_config(cfg);
    }

    return 1;
}

// 기본 설정 파일 생성 (업데이트된 버전)
void create_default_config_file(Modoo_cfg* cfg) {
    write_config_file("option/main.txt", cfg);
}

// 실제 이미지 처리 함수 (더미에서 실제 구현으로 교체)
int process_single_image(const char* image_path, Modoo_cfg* cfg, int mode) {
    char log_msg[512];
    sprintf_s(log_msg, sizeof(log_msg), "Processing: %s (mode %d)", image_path, mode);
    log_message(log_msg);

    // 처리 시간 측정 시작
    clock_t start_time = clock();

    // 결과 구조체 초기화
    ImageProcessResult result;
    memset(&result, 0, sizeof(result));

    // 파일명 추출
    extract_filename_without_ext(image_path, result.original_filename, sizeof(result.original_filename));
    generate_result_filename(image_path, result.result_filename, sizeof(result.result_filename));
    result.mode = mode;

    // 실제 이미지 처리 수행
    int processing_result = 0;

    printf("  -> Attempting to process image with OpenCV...\n");

    if (mode == 1) {
        // Normal detection mode
        processing_result = ProcessImageFromFile_Mode1(image_path, cfg, &result);

        printf("  -> ProcessImageFromFile_Mode1 returned: %d\n", processing_result);
        printf("  -> Result success_count: %d\n", result.success_count);
        printf("  -> Result total_detections: %d\n", result.total_detections);

        if (processing_result && result.success_count > 0) {
            DetectionResult* det = &result.detections[0];
            sprintf_s(result.final_result, sizeof(result.final_result),
                "ordi,1,01,210.000,5.000,001.000,002.000,1");

            printf("  -> Normal detection completed\n");
            printf("  -> Center: (%d, %d), Radius: %d, Area: %d\n",
                det->center_x, det->center_y, det->radius, det->area);
            printf("  -> Color: %s, Result: %s\n", det->color_info, det->result_msg);
        }
        else {
            sprintf_s(result.final_result, sizeof(result.final_result),
                "ordi,0,00,000.000,0.000,000.000,000.000,0");
            printf("  -> Normal detection failed - no valid detections\n");
        }

    }
    else {
        // 3rd detection mode (20개 영역 검사)
        processing_result = ProcessImageFromFile_Mode2(image_path, cfg, &result);

        sprintf_s(result.final_result, sizeof(result.final_result),
            "3rd,%s,%d,00,210.000,5.000,001.000,002.000",
            (result.success_count >= 18) ? "OK" : "NG", result.success_count);

        printf("  -> 3rd detection completed\n");
        printf("  -> Success: %d/20 areas\n", result.success_count);

        // 각 영역 결과 요약 출력
        int ok_count = 0, ng_count = 0, off_count = 0;
        for (int i = 0; i < 20; i++) {
            if (strcmp(result.detections[i].result_msg, "OK") == 0) ok_count++;
            else if (strcmp(result.detections[i].result_msg, "OFF") == 0) off_count++;
            else ng_count++;
        }
        printf("  -> Breakdown: OK:%d, NG:%d, OFF:%d\n", ok_count, ng_count, off_count);
    }

    // 처리 시간 계산
    clock_t end_time = clock();
    result.processing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("  -> Processing time: %.3f seconds\n", result.processing_time);
    printf("  -> Result: %s\n", result.final_result);
    printf("  -> Result image saved: %s\n", result.result_filename);

    // 로그에 결과 기록
    log_image_result(&result);

    return processing_result;
}

int main(int argc, char** argv) {
    printf("=== Test Detector Program ===\n");
    printf("OpenCV Version: 3.4.16\n");
    printf("Build Date: %s %s\n\n", __DATE__, __TIME__);

    // 터미널에서 모드 입력 받기
    int mode;
    char input[10];

    while (1) {
        printf("Select detection mode:\n");
        printf("  1 : Normal detection mode (CaptureImage)\n");
        printf("  2 : 3rd detection mode (CaptureImage3rd)\n");
        printf("  q : Quit program\n");
        printf("Enter mode (1, 2, or q): ");

        if (fgets(input, sizeof(input), stdin) != NULL) {
            // 개행 문자 제거
            input[strcspn(input, "\n")] = 0;

            // 종료 체크
            if (strcmp(input, "q") == 0 || strcmp(input, "Q") == 0) {
                printf("Program terminated by user.\n");
                return 0;
            }

            // 모드 파싱
            mode = atoi(input);
            if (mode == 1 || mode == 2) {
                break; // 유효한 입력이면 루프 탈출
            }

            printf("Error: Invalid input '%s'. Please enter 1, 2, or q.\n\n", input);
        }
        else {
            printf("Error: Failed to read input.\n");
            return -1;
        }
    }

    printf("Selected Mode: %d (%s)\n", mode,
        mode == 1 ? "Normal Detection" : "3rd Detection");

    // 디렉토리 확인 및 생성
    if (!check_directories()) {
        printf("Error: Directory setup failed\n");
        return -1;
    }

    // 이미지 파일 개수 확인
    int image_count = count_image_files("test_images");
    if (image_count == 0) {
        printf("\nWarning: No image files found in test_images directory\n");
        printf("Please add .jpg, .jpeg, .png, or .bmp files to test_images folder\n");
        return -1;
    }

    printf("\nFound %d image files\n\n", image_count);

    // 설정 로드
    Modoo_cfg cfg;
    if (!load_simple_config(&cfg)) {
        printf("Error: Failed to load configuration\n");
        return -1;
    }

    // 로그 시스템 초기화
    if (!initialize_logger()) {
        printf("Error: Failed to initialize logging system\n");
        return -1;
    }

    // OpenCV 연결 테스트
    printf("Testing OpenCV connection...\n");
    if (!test_opencv_basic()) {
        printf("ERROR: OpenCV basic test failed!\n");
        printf("Check if OpenCV libraries are properly linked.\n");
        cleanup_logger();
        return -1;
    }

    // 첫 번째 이미지로 로드 테스트
    WIN32_FIND_DATAA testFileData;
    HANDLE testFind;
    char test_search_path[260];
    sprintf_s(test_search_path, sizeof(test_search_path), "test_images\\*.*");
    testFind = FindFirstFileA(test_search_path, &testFileData);

    if (testFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(testFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char* ext = strrchr(testFileData.cFileName, '.');
                if (ext != NULL && (_stricmp(ext, ".jpg") == 0 || _stricmp(ext, ".jpeg") == 0 ||
                    _stricmp(ext, ".png") == 0 || _stricmp(ext, ".bmp") == 0)) {

                    char test_full_path[512];
                    sprintf_s(test_full_path, sizeof(test_full_path), "test_images\\%s", testFileData.cFileName);

                    printf("Testing image load with: %s\n", test_full_path);
                    if (!test_image_load(test_full_path)) {
                        printf("ERROR: Failed to load test image!\n");
                        printf("Check if the image file is valid and OpenCV can read it.\n");
                    }
                    else {
                        printf("Image load test: SUCCESS\n");
                    }
                    break; // 첫 번째 이미지만 테스트
                }
            }
        } while (FindNextFileA(testFind, &testFileData) != 0);
        FindClose(testFind);
    }

    // 배치 처리 시작
    printf("Starting batch processing...\n");
    log_message("=== Batch Processing Started ===");
    printf("==============================================\n");

    // 실제 이미지 처리
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind;
    char search_path[260];
    int processed_count = 0;
    int success_count = 0;

    sprintf_s(search_path, sizeof(search_path), "test_images\\*.*");
    hFind = FindFirstFileA(search_path, &findFileData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char* ext = strrchr(findFileData.cFileName, '.');
                if (ext != NULL) {
                    if (_stricmp(ext, ".jpg") == 0 || _stricmp(ext, ".jpeg") == 0 ||
                        _stricmp(ext, ".png") == 0 || _stricmp(ext, ".bmp") == 0) {

                        char full_path[512];
                        sprintf_s(full_path, sizeof(full_path), "test_images\\%s", findFileData.cFileName);

                        processed_count++;
                        if (process_single_image(full_path, &cfg, mode)) {
                            success_count++;
                        }
                        printf("\n");
                    }
                }
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }

    printf("==============================================\n");
    printf("Batch processing completed!\n");
    printf("Processed: %d/%d images\n", success_count, processed_count);
    printf("Success rate: %.1f%%\n", processed_count > 0 ? (100.0 * success_count / processed_count) : 0.0);

    printf("\nResults will be saved in 'results' directory\n");

    // 결과 요약 생성
    generate_summary_report();

    // 로그 시스템 정리
    cleanup_logger();

    // 계속 실행할지 묻기
    char continue_input[10];
    printf("\nDo you want to process more images? (y/n): ");
    if (fgets(continue_input, sizeof(continue_input), stdin) != NULL) {
        continue_input[strcspn(continue_input, "\n")] = 0;
        if (strcmp(continue_input, "y") == 0 || strcmp(continue_input, "Y") == 0) {
            printf("\n");
            main(0, NULL); // 재귀 호출로 다시 시작
            return 0;
        }
    }

    printf("Program finished successfully!\n");
    printf("Press any key to exit...");
    getchar(); // 마지막 엔터키 대기

    return 0;
}

// HSV 색상 초기화 함수 (원본 modoo.h 구조체 기준)
void init_default_hsv_colors(Modoo_cfg* cfg) {
    // camera.cpp의 targetHsvColors3rd와 동일하게 설정
    // 보라
    cfg->hsvColors[0].h = 116; cfg->hsvColors[0].s = 171; cfg->hsvColors[0].v = 116; cfg->hsvColors[0].id = 0;
    // 핑크  
    cfg->hsvColors[1].h = 164; cfg->hsvColors[1].s = 145; cfg->hsvColors[1].v = 244; cfg->hsvColors[1].id = 1;
    // 빨강
    cfg->hsvColors[2].h = 5;   cfg->hsvColors[2].s = 179; cfg->hsvColors[2].v = 71;  cfg->hsvColors[2].id = 2;
    // 노랑
    cfg->hsvColors[3].h = 36;  cfg->hsvColors[3].s = 130; cfg->hsvColors[3].v = 120; cfg->hsvColors[3].id = 3;
    // 초록
    cfg->hsvColors[4].h = 63;  cfg->hsvColors[4].s = 125; cfg->hsvColors[4].v = 82;  cfg->hsvColors[4].id = 4;
    // 파랑
    cfg->hsvColors[5].h = 111; cfg->hsvColors[5].s = 211; cfg->hsvColors[5].v = 82;  cfg->hsvColors[5].id = 5;

    // maxColorCount 설정
    cfg->maxColorCount = 6;
}