#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>    // Windows용 mkdir
#include <io.h>        // Windows용 파일 접근
#include <windows.h>   // Windows API
#include <time.h>      // 시간 측정용

#include "simple_types.h"  // modoo.h와 동일한 구조체 사용
#include "logger.h"
#include "image_result.h"
#include "image_processor.h"
// config_parser.h 제거하고 modoo.h의 함수 사용

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
int test_image_load(const char* filepath);

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

// 설정 파일 읽기 - modoo.c의 readMainSetFromFile 함수 사용
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
    cfg->hsvEnable = 0; // 기본적으로 HSV 비활성화
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

    // 제품별 색상 매핑 기본값 - 순환하지 않고 기본값으로 설정
    for (int i = 0; i < 20; i++) {
        cfg->productNumColor[i] = 1; // 모든 제품을 기본적으로 1번 색상으로 설정
        cfg->isDetect[i] = 1; // 모든 검사 활성화
    }

    // modoo.c의 readMainSetFromFile 함수 사용
    printf("Attempting to load configuration file using modoo.c...\n");
    if (readMainSetFromFile("option/main.txt", cfg) == 0) {
        printf("Configuration loaded from file successfully using modoo.c\n");

        // 설정 검증
        if (cfg->maxColorCount <= 0) {
            printf("Warning: No HSV colors loaded, using default values\n");
            init_default_hsv_colors(cfg);
        }

        // HSV 색상 정보 출력
        printf("Loaded %d HSV colors:\n", cfg->maxColorCount);
        for (int i = 0; i < cfg->maxColorCount && i < 6; i++) {
            printf("  Color %d: H=%d, S=%d, V=%d, ID=%d\n",
                i, cfg->hsvColors[i].h, cfg->hsvColors[i].s, cfg->hsvColors[i].v, cfg->hsvColors[i].id);
        }
    }
    else {
        printf("Configuration file not found or read failed, creating default file\n");
        create_default_config_file(cfg);
    }

    // 디버그 모드면 주요 설정 출력
    if (cfg->debugMode) {
        printf("\n=== Configuration Summary ===\n");
        printf("HSVEnable: %d\n", cfg->hsvEnable);
        printf("BinaryValue: %d\n", cfg->binaryValue);
        printf("MinContourArea: %d\n", cfg->MinContourArea);
        printf("MaxContourArea: %d\n", cfg->MaxContourArea);
        printf("debugMode: %d\n", cfg->debugMode);
        printf("VisionID: %d\n", cfg->VisionID);
        printf("ProductNum: %d\n", cfg->ProductNum);
        printf("HSV Buffer (H,S,V): %d,%d,%d\n", cfg->HsvBufferH, cfg->HsvBufferS, cfg->HsvBufferV);
        printf("Mark Area Range: %d ~ %d\n", cfg->minMarkArea, cfg->maxMarkArea);
        printf("Black Mark Area Range: %d ~ %d\n", cfg->blackMinMarkArea, cfg->blackMaxMarkArea);
        printf("===============================\n\n");
    }

    return 1;
}

// 기본 설정 파일 생성
void create_default_config_file(Modoo_cfg* cfg) {
    FILE* file = fopen("option/main.txt", "w");
    if (file == NULL) {
        printf("Error: Cannot create default config file\n");
        return;
    }

    fprintf(file, "# Default configuration file for TestDetector\n");
    fprintf(file, "TriggerMode=SW\n");
    fprintf(file, "DetectionMode=Rule\n");
    fprintf(file, "ThresholdMode=Binary\n");
    fprintf(file, "BinaryValue=%d\n", cfg->binaryValue);
    fprintf(file, "CameraModelName=%s\n", cfg->camModelName ? cfg->camModelName : "TestCamera");
    fprintf(file, "CameraSerialNum=%s\n", cfg->camSerialNum ? cfg->camSerialNum : "123456");
    fprintf(file, "HSVEnable=False\n");
    fprintf(file, "MinContourArea=%d\n", cfg->MinContourArea);
    fprintf(file, "MaxContourArea=%d\n", cfg->MaxContourArea);
    fprintf(file, "VisionID=%d\n", cfg->VisionID);
    fprintf(file, "PortNum=%d\n", cfg->PortNum);
    fprintf(file, "origin_vision_x=%.0f\n", cfg->origin_vision_x);
    fprintf(file, "origin_vision_y=%.0f\n", cfg->origin_vision_y);
    fprintf(file, "origin_robot_x=%.3f\n", cfg->origin_robot_x);
    fprintf(file, "origin_robot_y=%.3f\n", cfg->origin_robot_y);
    fprintf(file, "res_x=%.8f\n", cfg->res_x);
    fprintf(file, "res_y=%.8f\n", cfg->res_y);
    fprintf(file, "Max_vision_x=%d\n", cfg->Max_vision_x);
    fprintf(file, "Max_vision_y=%d\n", cfg->Max_vision_y);
    fprintf(file, "minMarkArea=%d\n", cfg->minMarkArea);
    fprintf(file, "maxMarkArea=%d\n", cfg->maxMarkArea);
    fprintf(file, "HsvBufferH=%d\n", cfg->HsvBufferH);
    fprintf(file, "HsvBufferS=%d\n", cfg->HsvBufferS);
    fprintf(file, "HsvBufferV=%d\n", cfg->HsvBufferV);
    fprintf(file, "debugMode=%d\n", cfg->debugMode);
    fprintf(file, "minSUMMarkArea=%d\n", cfg->minSUMMarkArea);
    fprintf(file, "maxSUMMarkArea=%d\n", cfg->maxSUMMarkArea);
    fprintf(file, "BlackTagNum=%d\n", cfg->BlackTagNum);
    fprintf(file, "blackMinMarkArea=%d\n", cfg->blackMinMarkArea);
    fprintf(file, "blackMaxMarkArea=%d\n", cfg->blackMaxMarkArea);
    fprintf(file, "BlackEllipseMinSize=%d\n", cfg->BlackEllipseMinSize);
    fprintf(file, "BlackEllipseMaxSize=%d\n", cfg->BlackEllipseMaxSize);

    // HSV 색상들 저장
    for (int i = 0; i < cfg->maxColorCount; i++) {
        fprintf(file, "HsvColor%d=%d,%d,%d,%d\n", i,
            cfg->hsvColors[i].h, cfg->hsvColors[i].s, cfg->hsvColors[i].v, cfg->hsvColors[i].id);
    }

    // 제품별 색상 매핑 저장
    for (int i = 0; i < 20; i++) {
        fprintf(file, "productNumColor%d=%d\n", i + 1, cfg->productNumColor[i]);
        fprintf(file, "isDetect%d=%d\n", i + 1, cfg->isDetect[i]);
        fprintf(file, "area%d=%d\n", i + 1, cfg->areas[i]);
    }

    fclose(file);
    printf("Default configuration file created: option/main.txt\n");
}

// main.c에 추가할 PN 번호 추출 함수
int extract_pn_from_filename(const char* filename) {
    // 파일명에서 "_PN" 패턴을 찾기
    char* pn_pos = strstr(filename, "_PN");
    if (pn_pos == NULL) {
        pn_pos = strstr(filename, "PN"); // PN만 있는 경우도 체크
    }

    if (pn_pos != NULL) {
        // "_PN" 다음 문자부터 숫자 추출
        char* num_start = pn_pos + (pn_pos[0] == '_' ? 3 : 2); // "_PN" 또는 "PN" 건너뛰기
        int pn_number = atoi(num_start);
        printf("    [DEBUG] Extracted PN number: %d from filename: %s\n", pn_number, filename);
        return pn_number;
    }

    printf("    [WARNING] No PN number found in filename: %s, using default 1\n", filename);
    return 1; // 기본값
}

// setContourAreaByProduct 함수 (detector.c에서 가져옴)
void setContourAreaByProduct(Modoo_cfg* modoo_cfg, int productNum) {
    // productNum이 유효한 인덱스인지 확인
    if (productNum < 0 || productNum >= MAX_AREAS) {
        fprintf(stderr, "Invalid productNum: %d\n", productNum);
        return;
    }

    // 해당 productNum에 대응하는 area 값
    int areaValue = modoo_cfg->areas[productNum];

    // areaThresholdPercentLower와 areaThresholdPercentUpper를 곱한 값을 각각 MinContourArea, MaxContourArea에 설정
    modoo_cfg->MinContourArea = (int)(areaValue * modoo_cfg->areaThresholdPercentLower);
    modoo_cfg->MaxContourArea = (int)(areaValue * modoo_cfg->areaThresholdPercentUpper);

    printf("Set MinContourArea to %d and MaxContourArea to %d for productNum %d\n",
        modoo_cfg->MinContourArea, modoo_cfg->MaxContourArea, productNum);
}


// 실제 이미지 처리 함수 (동일)
// process_single_image 함수 수정 부분
int process_single_image(const char* image_path, Modoo_cfg* cfg, int mode) {
    char log_msg[512];
    sprintf_s(log_msg, sizeof(log_msg), "Processing: %s (mode %d)", image_path, mode);
    log_message(log_msg);

    // 처리 시간 측정 시작
    clock_t start_time = clock();

    // 파일명에서 PN 번호 추출 (detector.c 로직 적용)
    int extracted_pn = extract_pn_from_filename(image_path);
    cfg->ProductNum = extracted_pn;

    // VisionID가 2인 경우 10을 더함 (detector.c와 동일)
    if (cfg->VisionID == 2) {
        cfg->ProductNum = cfg->ProductNum + 10;
    }

    printf("    [INFO] ProductNum set to: %d (extracted PN: %d, VisionID: %d)\n",
        cfg->ProductNum, extracted_pn, cfg->VisionID);

    // 해당 ProductNum에 맞는 Contour Area 설정
    setContourAreaByProduct(cfg, cfg->ProductNum - 1);

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
    printf("  -> ProductNum: %d, Target productNumColor: %d\n",
        cfg->ProductNum, cfg->productNumColor[cfg->ProductNum - 1]);

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

    // 설정 로드 (modoo.c 사용)
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
    cfg->hsvColors[0].h = 116; cfg->hsvColors[0].s = 171; cfg->hsvColors[0].v = 116; cfg->hsvColors[0].id = 1;
    // 핑크  
    cfg->hsvColors[1].h = 164; cfg->hsvColors[1].s = 145; cfg->hsvColors[1].v = 244; cfg->hsvColors[1].id = 2;
    // 빨강
    cfg->hsvColors[2].h = 5;   cfg->hsvColors[2].s = 179; cfg->hsvColors[2].v = 71;  cfg->hsvColors[2].id = 3;
    // 노랑
    cfg->hsvColors[3].h = 36;  cfg->hsvColors[3].s = 130; cfg->hsvColors[3].v = 120; cfg->hsvColors[3].id = 4;
    // 초록
    cfg->hsvColors[4].h = 63;  cfg->hsvColors[4].s = 125; cfg->hsvColors[4].v = 82;  cfg->hsvColors[4].id = 5;
    // 파랑
    cfg->hsvColors[5].h = 111; cfg->hsvColors[5].s = 211; cfg->hsvColors[5].v = 82;  cfg->hsvColors[5].id = 6;

    // maxColorCount 설정
    cfg->maxColorCount = 6;
}