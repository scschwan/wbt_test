#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>    // Windows�� mkdir
#include <io.h>        // Windows�� ���� ����
#include <windows.h>   // Windows API
#include <time.h>      // �ð� ������

#include "simple_types.h"  // modoo.h�� ������ ����ü ���
#include "logger.h"
#include "image_result.h"
#include "image_processor.h"
// config_parser.h �����ϰ� modoo.h�� �Լ� ���

// �Լ� ���� (������Ÿ��)
void print_usage(const char* program_name);
int check_directories();
int count_image_files(const char* folder_path);
int load_simple_config(Modoo_cfg* cfg);
void create_default_config_file(Modoo_cfg* cfg);
int process_single_image(const char* image_path, Modoo_cfg* cfg, int mode);
void init_default_hsv_colors(Modoo_cfg* cfg);
// OpenCV �׽�Ʈ �Լ���
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
    // test_images ���� Ȯ��
    if (_access("test_images", 0) == -1) {
        printf("Creating test_images directory...\n");
        if (_mkdir("test_images") != 0) {
            printf("Error: Cannot create test_images directory\n");
            return 0;
        }
    }

    // results ���� Ȯ��
    if (_access("results", 0) == -1) {
        printf("Creating results directory...\n");
        if (_mkdir("results") != 0) {
            printf("Error: Cannot create results directory\n");
            return 0;
        }
    }

    // option ���� Ȯ��
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

// ���� ���� �б� - modoo.c�� readMainSetFromFile �Լ� ���
int load_simple_config(Modoo_cfg* cfg) {
    // �⺻�� ����
    memset(cfg, 0, sizeof(Modoo_cfg));

    // �⺻ ��������
    cfg->triggerMode = TRIGGER_MODE_SW;
    cfg->detectionMode = DETECTION_MODE_ONLY_RULE;
    cfg->thresholdMode = THRESHOLD_MODE_BINARY;
    cfg->VisionID = 1;
    cfg->ProductNum = 1;
    cfg->PortNum = 9999;
    cfg->maxColorCount = 6;

    // ī�޶� ���� (���� �Ҵ�)
    cfg->camModelName = (char*)malloc(256);
    cfg->camSerialNum = (char*)malloc(256);
    if (cfg->camModelName) strcpy_s(cfg->camModelName, 256, "TestCamera");
    if (cfg->camSerialNum) strcpy_s(cfg->camSerialNum, 256, "123456");

    // �⺻ �Ӱ谪��
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
    cfg->hsvEnable = 0; // �⺻������ HSV ��Ȱ��ȭ
    cfg->debugMode = 1;

    // ��ǥ ��ȯ ����
    cfg->origin_vision_x = 780.0;
    cfg->origin_vision_y = 422.0;
    cfg->origin_robot_x = 196.0;
    cfg->origin_robot_y = 1.260;
    cfg->res_x = 0.02303163445;
    cfg->res_y = 0.0231194784;

    cfg->Max_vision_x = 1920;
    cfg->Max_vision_y = 1080;

    // ���� �Ӱ谪
    cfg->areaThresholdPercentLower = 0.8;
    cfg->areaThresholdPercentUpper = 1.2;

    // �⺻ ���� �迭 �ʱ�ȭ
    for (int i = 0; i < MAX_AREAS; i++) {
        cfg->areas[i] = 35000; // �⺻ ������
    }

    // �⺻ HSV ����� ����
    init_default_hsv_colors(cfg);

    // ��ǰ�� ���� ���� �⺻�� - ��ȯ���� �ʰ� �⺻������ ����
    for (int i = 0; i < 20; i++) {
        cfg->productNumColor[i] = 1; // ��� ��ǰ�� �⺻������ 1�� �������� ����
        cfg->isDetect[i] = 1; // ��� �˻� Ȱ��ȭ
    }

    // modoo.c�� readMainSetFromFile �Լ� ���
    printf("Attempting to load configuration file using modoo.c...\n");
    if (readMainSetFromFile("option/main.txt", cfg) == 0) {
        printf("Configuration loaded from file successfully using modoo.c\n");

        // ���� ����
        if (cfg->maxColorCount <= 0) {
            printf("Warning: No HSV colors loaded, using default values\n");
            init_default_hsv_colors(cfg);
        }

        // HSV ���� ���� ���
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

    // ����� ���� �ֿ� ���� ���
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

// �⺻ ���� ���� ����
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

    // HSV ����� ����
    for (int i = 0; i < cfg->maxColorCount; i++) {
        fprintf(file, "HsvColor%d=%d,%d,%d,%d\n", i,
            cfg->hsvColors[i].h, cfg->hsvColors[i].s, cfg->hsvColors[i].v, cfg->hsvColors[i].id);
    }

    // ��ǰ�� ���� ���� ����
    for (int i = 0; i < 20; i++) {
        fprintf(file, "productNumColor%d=%d\n", i + 1, cfg->productNumColor[i]);
        fprintf(file, "isDetect%d=%d\n", i + 1, cfg->isDetect[i]);
        fprintf(file, "area%d=%d\n", i + 1, cfg->areas[i]);
    }

    fclose(file);
    printf("Default configuration file created: option/main.txt\n");
}

// main.c�� �߰��� PN ��ȣ ���� �Լ�
int extract_pn_from_filename(const char* filename) {
    // ���ϸ��� "_PN" ������ ã��
    char* pn_pos = strstr(filename, "_PN");
    if (pn_pos == NULL) {
        pn_pos = strstr(filename, "PN"); // PN�� �ִ� ��쵵 üũ
    }

    if (pn_pos != NULL) {
        // "_PN" ���� ���ں��� ���� ����
        char* num_start = pn_pos + (pn_pos[0] == '_' ? 3 : 2); // "_PN" �Ǵ� "PN" �ǳʶٱ�
        int pn_number = atoi(num_start);
        printf("    [DEBUG] Extracted PN number: %d from filename: %s\n", pn_number, filename);
        return pn_number;
    }

    printf("    [WARNING] No PN number found in filename: %s, using default 1\n", filename);
    return 1; // �⺻��
}

// setContourAreaByProduct �Լ� (detector.c���� ������)
void setContourAreaByProduct(Modoo_cfg* modoo_cfg, int productNum) {
    // productNum�� ��ȿ�� �ε������� Ȯ��
    if (productNum < 0 || productNum >= MAX_AREAS) {
        fprintf(stderr, "Invalid productNum: %d\n", productNum);
        return;
    }

    // �ش� productNum�� �����ϴ� area ��
    int areaValue = modoo_cfg->areas[productNum];

    // areaThresholdPercentLower�� areaThresholdPercentUpper�� ���� ���� ���� MinContourArea, MaxContourArea�� ����
    modoo_cfg->MinContourArea = (int)(areaValue * modoo_cfg->areaThresholdPercentLower);
    modoo_cfg->MaxContourArea = (int)(areaValue * modoo_cfg->areaThresholdPercentUpper);

    printf("Set MinContourArea to %d and MaxContourArea to %d for productNum %d\n",
        modoo_cfg->MinContourArea, modoo_cfg->MaxContourArea, productNum);
}


// ���� �̹��� ó�� �Լ� (����)
// process_single_image �Լ� ���� �κ�
int process_single_image(const char* image_path, Modoo_cfg* cfg, int mode) {
    char log_msg[512];
    sprintf_s(log_msg, sizeof(log_msg), "Processing: %s (mode %d)", image_path, mode);
    log_message(log_msg);

    // ó�� �ð� ���� ����
    clock_t start_time = clock();

    // ���ϸ��� PN ��ȣ ���� (detector.c ���� ����)
    int extracted_pn = extract_pn_from_filename(image_path);
    cfg->ProductNum = extracted_pn;

    // VisionID�� 2�� ��� 10�� ���� (detector.c�� ����)
    if (cfg->VisionID == 2) {
        cfg->ProductNum = cfg->ProductNum + 10;
    }

    printf("    [INFO] ProductNum set to: %d (extracted PN: %d, VisionID: %d)\n",
        cfg->ProductNum, extracted_pn, cfg->VisionID);

    // �ش� ProductNum�� �´� Contour Area ����
    setContourAreaByProduct(cfg, cfg->ProductNum - 1);

    // ��� ����ü �ʱ�ȭ
    ImageProcessResult result;
    memset(&result, 0, sizeof(result));

    // ���ϸ� ����
    extract_filename_without_ext(image_path, result.original_filename, sizeof(result.original_filename));
    generate_result_filename(image_path, result.result_filename, sizeof(result.result_filename));
    result.mode = mode;

    // ���� �̹��� ó�� ����
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
        // 3rd detection mode (20�� ���� �˻�)
        processing_result = ProcessImageFromFile_Mode2(image_path, cfg, &result);

        sprintf_s(result.final_result, sizeof(result.final_result),
            "3rd,%s,%d,00,210.000,5.000,001.000,002.000",
            (result.success_count >= 18) ? "OK" : "NG", result.success_count);

        printf("  -> 3rd detection completed\n");
        printf("  -> Success: %d/20 areas\n", result.success_count);

        // �� ���� ��� ��� ���
        int ok_count = 0, ng_count = 0, off_count = 0;
        for (int i = 0; i < 20; i++) {
            if (strcmp(result.detections[i].result_msg, "OK") == 0) ok_count++;
            else if (strcmp(result.detections[i].result_msg, "OFF") == 0) off_count++;
            else ng_count++;
        }
        printf("  -> Breakdown: OK:%d, NG:%d, OFF:%d\n", ok_count, ng_count, off_count);
    }

    // ó�� �ð� ���
    clock_t end_time = clock();
    result.processing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("  -> Processing time: %.3f seconds\n", result.processing_time);
    printf("  -> Result: %s\n", result.final_result);
    printf("  -> Result image saved: %s\n", result.result_filename);

    // �α׿� ��� ���
    log_image_result(&result);

    return processing_result;
}

int main(int argc, char** argv) {
    printf("=== Test Detector Program ===\n");
    printf("OpenCV Version: 3.4.16\n");
    printf("Build Date: %s %s\n\n", __DATE__, __TIME__);

    // �͹̳ο��� ��� �Է� �ޱ�
    int mode;
    char input[10];

    while (1) {
        printf("Select detection mode:\n");
        printf("  1 : Normal detection mode (CaptureImage)\n");
        printf("  2 : 3rd detection mode (CaptureImage3rd)\n");
        printf("  q : Quit program\n");
        printf("Enter mode (1, 2, or q): ");

        if (fgets(input, sizeof(input), stdin) != NULL) {
            // ���� ���� ����
            input[strcspn(input, "\n")] = 0;

            // ���� üũ
            if (strcmp(input, "q") == 0 || strcmp(input, "Q") == 0) {
                printf("Program terminated by user.\n");
                return 0;
            }

            // ��� �Ľ�
            mode = atoi(input);
            if (mode == 1 || mode == 2) {
                break; // ��ȿ�� �Է��̸� ���� Ż��
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

    // ���丮 Ȯ�� �� ����
    if (!check_directories()) {
        printf("Error: Directory setup failed\n");
        return -1;
    }

    // �̹��� ���� ���� Ȯ��
    int image_count = count_image_files("test_images");
    if (image_count == 0) {
        printf("\nWarning: No image files found in test_images directory\n");
        printf("Please add .jpg, .jpeg, .png, or .bmp files to test_images folder\n");
        return -1;
    }

    printf("\nFound %d image files\n\n", image_count);

    // ���� �ε� (modoo.c ���)
    Modoo_cfg cfg;
    if (!load_simple_config(&cfg)) {
        printf("Error: Failed to load configuration\n");
        return -1;
    }

    // �α� �ý��� �ʱ�ȭ
    if (!initialize_logger()) {
        printf("Error: Failed to initialize logging system\n");
        return -1;
    }

    // OpenCV ���� �׽�Ʈ
    printf("Testing OpenCV connection...\n");
    if (!test_opencv_basic()) {
        printf("ERROR: OpenCV basic test failed!\n");
        printf("Check if OpenCV libraries are properly linked.\n");
        cleanup_logger();
        return -1;
    }

    // ù ��° �̹����� �ε� �׽�Ʈ
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
                    break; // ù ��° �̹����� �׽�Ʈ
                }
            }
        } while (FindNextFileA(testFind, &testFileData) != 0);
        FindClose(testFind);
    }

    // ��ġ ó�� ����
    printf("Starting batch processing...\n");
    log_message("=== Batch Processing Started ===");
    printf("==============================================\n");

    // ���� �̹��� ó��
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

    // ��� ��� ����
    generate_summary_report();

    // �α� �ý��� ����
    cleanup_logger();

    // ��� �������� ����
    char continue_input[10];
    printf("\nDo you want to process more images? (y/n): ");
    if (fgets(continue_input, sizeof(continue_input), stdin) != NULL) {
        continue_input[strcspn(continue_input, "\n")] = 0;
        if (strcmp(continue_input, "y") == 0 || strcmp(continue_input, "Y") == 0) {
            printf("\n");
            main(0, NULL); // ��� ȣ��� �ٽ� ����
            return 0;
        }
    }

    printf("Program finished successfully!\n");
    printf("Press any key to exit...");
    getchar(); // ������ ����Ű ���

    return 0;
}

// HSV ���� �ʱ�ȭ �Լ� (���� modoo.h ����ü ����)
void init_default_hsv_colors(Modoo_cfg* cfg) {
    // camera.cpp�� targetHsvColors3rd�� �����ϰ� ����
    // ����
    cfg->hsvColors[0].h = 116; cfg->hsvColors[0].s = 171; cfg->hsvColors[0].v = 116; cfg->hsvColors[0].id = 1;
    // ��ũ  
    cfg->hsvColors[1].h = 164; cfg->hsvColors[1].s = 145; cfg->hsvColors[1].v = 244; cfg->hsvColors[1].id = 2;
    // ����
    cfg->hsvColors[2].h = 5;   cfg->hsvColors[2].s = 179; cfg->hsvColors[2].v = 71;  cfg->hsvColors[2].id = 3;
    // ���
    cfg->hsvColors[3].h = 36;  cfg->hsvColors[3].s = 130; cfg->hsvColors[3].v = 120; cfg->hsvColors[3].id = 4;
    // �ʷ�
    cfg->hsvColors[4].h = 63;  cfg->hsvColors[4].s = 125; cfg->hsvColors[4].v = 82;  cfg->hsvColors[4].id = 5;
    // �Ķ�
    cfg->hsvColors[5].h = 111; cfg->hsvColors[5].s = 211; cfg->hsvColors[5].v = 82;  cfg->hsvColors[5].id = 6;

    // maxColorCount ����
    cfg->maxColorCount = 6;
}