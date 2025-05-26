#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "modoo.h"

int readMainSetFromFile(const char* filename, Modoo_cfg* modoo_cfg) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }

    int colorIndex = 0;
    char line[300];
    while (fgets(line, sizeof(line), file)) {
        char key[150], value[150];
        printf("key : %s, value : %s\n", key, value);
        if (sscanf(line, "%149[^=]=%149s", key, value) == 2) {
            if (strcmp(key, "TriggerMode") == 0) {
                if (strcmp(value, "SW") == 0) {
                    modoo_cfg->triggerMode = TRIGGER_MODE_SW;
                }
                else if (strcmp(value, "HW") == 0) {
                    modoo_cfg->triggerMode = TRIGGER_MODE_HW;
                }
                else {
                    fprintf(stderr, "Unknown trigger mode: %s\n", value);
                    fclose(file);
                    return -1;
                }
            }
            else if (strcmp(key, "DetectionMode") == 0) {
                if (strcmp(value, "Rule") == 0) {
                    modoo_cfg->detectionMode = DETECTION_MODE_ONLY_RULE;
                }
                else if (strcmp(value, "Deep") == 0) {
                    modoo_cfg->detectionMode = DETECTION_MODE_ONLY_DEEP;
                }
                else if (strcmp(value, "Both") == 0) {
                    modoo_cfg->detectionMode = DETECTION_MODE_BOTH;
                }
                else {
                    fprintf(stderr, "Unknown detection mode: %s\n", value);
                    fclose(file);
                    return -1;
                }
            }
            else if (strcmp(key, "ThresholdMode") == 0){
                if (strcmp(value, "None") == 0) {
                    modoo_cfg->thresholdMode = THRESHOLD_MODE_NONE;
                }
                else if (strcmp(value, "Binary") == 0) {
                    modoo_cfg->thresholdMode = THRESHOLD_MODE_BINARY;
                }
                else {
                    fprintf(stderr, "Unknown threshold mode: %s\n", value);
                    fclose(file);
                    return -1;
                }
            }
            else if (strcmp(key, "BinaryValue") == 0) {
                modoo_cfg->binaryValue = atoi(value);
                
            }
            else if (strcmp(key, "CameraModelName") == 0) {
                modoo_cfg->camModelName = _strdup(value);
                fprintf(stderr, "Cam name : %s\n", value);
            }
            else if (strcmp(key, "CameraSerialNum") == 0) {
                modoo_cfg->camSerialNum = _strdup(value);
                fprintf(stderr, "Cam Serial Num : %s\n", value);
            }
            else if (strcmp(key, "HSVEnable") == 0) {
                if (strcmp(value, "False") == 0) {
                    modoo_cfg->hsvEnable = 0;
                }
                else {

                }
            }
            else if (strcmp(key, "MinContourArea") == 0) {
                modoo_cfg->MinContourArea = atoi(value);
            }
            else if (strcmp(key, "MaxContourArea") == 0) {
                modoo_cfg->MaxContourArea = atoi(value);
            }
            else if (strcmp(key, "VisionID") == 0) {
                modoo_cfg->VisionID = atoi(value);
            }
            else if (strcmp(key, "PortNum") == 0) {
                modoo_cfg->PortNum = atoi(value);
            }
            else if (strcmp(key, "origin_vision_x") == 0) {
                modoo_cfg->origin_vision_x = atoi(value);
            }
            else if (strcmp(key, "origin_vision_y") == 0) {
                modoo_cfg->origin_vision_y = atoi(value);
            }
            else if (strcmp(key, "origin_robot_x") == 0) {
                modoo_cfg->origin_robot_x = atof(value);
            }
            else if (strcmp(key, "origin_robot_y") == 0) {
                modoo_cfg->origin_robot_y = atof(value);
            }
            else if (strcmp(key, "res_x") == 0) {
                modoo_cfg->res_x = atof(value);
            }
            else if (strcmp(key, "res_y") == 0) {
                modoo_cfg->res_y = atof(value);
            }
            else if (strcmp(key, "Max_vision_x") == 0) {
                modoo_cfg->Max_vision_x = atoi(value);
            }
            else if (strcmp(key, "Max_vision_y") == 0) {
                modoo_cfg->Max_vision_y = atoi(value);
            }
            else if (strcmp(key, "boundaryBuff_minX") == 0) {
                modoo_cfg->boundaryBuff_minX = atof(value);
            }
            else if (strcmp(key, "boundaryBuff_minY") == 0) {
                modoo_cfg->boundaryBuff_minY = atof(value);
            }
            else if (strcmp(key, "boundaryBuff_maxX") == 0) {
                modoo_cfg->boundaryBuff_maxX = atof(value);
            }
			else if (strcmp(key, "boundaryBuff_maxY") == 0) {
			    modoo_cfg->boundaryBuff_maxY = atof(value);
			}
            else if (strcmp(key, "areaThresholdPercentLower") == 0) {
                modoo_cfg->areaThresholdPercentLower = atof(value);
            }
            else if (strcmp(key, "areaThresholdPercentUpper") == 0) {
                modoo_cfg->areaThresholdPercentUpper = atof(value);
            }
            else if (strcmp(key, "minMarkArea") == 0) {
            modoo_cfg->minMarkArea = atoi(value);
            }
            else if (strcmp(key, "maxMarkArea") == 0) {
            modoo_cfg->maxMarkArea = atoi(value);
            }
            else if (strcmp(key, "HsvBufferH") == 0) {
            modoo_cfg->HsvBufferH = atoi(value);
            }
            else if (strcmp(key, "HsvBufferS") == 0) {
            modoo_cfg->HsvBufferS = atoi(value);
            }
            else if (strcmp(key, "HsvBufferV") == 0) {
            modoo_cfg->HsvBufferV = atoi(value);
            }
            else if (strcmp(key, "debugMode") == 0) {
            modoo_cfg->debugMode = atoi(value);
            }
            else if (strcmp(key, "minSUMMarkArea") == 0) {
            modoo_cfg->minSUMMarkArea = atoi(value);
            }
            else if (strcmp(key, "maxSUMMarkArea") == 0) {
            modoo_cfg->maxSUMMarkArea = atoi(value);
            }
            else if (strcmp(key, "BlackTagNum") == 0) {
            modoo_cfg->BlackTagNum = atoi(value);
            }
            else if (strcmp(key, "blackMinMarkArea") == 0) {
            modoo_cfg->blackMinMarkArea = atoi(value);
            }
            else if (strcmp(key, "blackMaxMarkArea") == 0) {
            modoo_cfg->blackMaxMarkArea = atoi(value);
            }
            else if (strcmp(key, "BlackEllipseMinSize") == 0) {
            modoo_cfg->BlackEllipseMinSize = atoi(value);
            }
            else if (strcmp(key, "BlackEllipseMaxSize") == 0) {
            modoo_cfg->BlackEllipseMaxSize = atoi(value);
            }
			// 추가: area1 ~ area20 처리
			else if (strncmp(key, "area", 4) == 0) {
			    int areaIndex;
			    if (sscanf(key, "area%d", &areaIndex) == 1 && areaIndex >= 1 && areaIndex <= MAX_AREAS) {
				    modoo_cfg->areas[areaIndex - 1] = atoi(value);  // area1은 areas[0]에 저장
			    }
			    else {
				    fprintf(stderr, "Invalid area index: %s\n", key);
					fclose(file);
					return -1;
				}
			}
            else if (strncmp(key, "isDetect", 8) == 0) {
            int index;
            if (sscanf(key, "isDetect%d", &index) == 1 && index >= 1 && index <= MAX_AREAS) {
                modoo_cfg->isDetect[index - 1] = atoi(value);  // area1은 areas[0]에 저장
            }
            else {
                fprintf(stderr, "Invalid isDetect index: %s\n", key);
                fclose(file);
                return -1;
            }
            }
			// 추가: area1 ~ area20 처리
			else if (strncmp(key, "productNumColor", 15) == 0) {
			int index;
			if (sscanf(key, "productNumColor%d", &index) == 1 && index >= 1 && index <= MAX_AREAS) {
				modoo_cfg->productNumColor[index - 1] = atoi(value);  // productNumColor1은 areas[0]에 저장
			}
			else {
				fprintf(stderr, "Invalid productNumColor index: %s\n", key);
				fclose(file);
				return -1;
			}
			}
            else if (strncmp(key, "HsvColor", 8) == 0) {
                if (colorIndex < 1024) {  // 배열 범위 체크
                    int h, s, v, id;
                    if (sscanf(value, "%d,%d,%d,%d", &h, &s, &v, &id) == 4) {
                        modoo_cfg->hsvColors[colorIndex].h = h;
                        modoo_cfg->hsvColors[colorIndex].s = s;
                        modoo_cfg->hsvColors[colorIndex].v = v;
                        modoo_cfg->hsvColors[colorIndex].id = id;
                        printf("hsv 불러오기 완료[%d] : %d,%d,%d,%d\n", colorIndex, modoo_cfg->hsvColors[colorIndex].h, modoo_cfg->hsvColors[colorIndex].s, modoo_cfg->hsvColors[colorIndex].v, modoo_cfg->hsvColors[colorIndex].id);
                        colorIndex++;
                    }
                    else {
                        fprintf(stderr, "Failed to parse HSV values: %s\n", value);
                    }
                }
                else {
                    fprintf(stderr, "Max HSV colors reached. Ignoring additional values.\n");
                }
            }
			else {
                fprintf(stderr, "Unknown key: %s\n", key);
                //fclose(file);
                continue;
            }        
        }
        else {
            fprintf(stderr, "Failed to parse line: %s\n", line);
            //fclose(file);
            continue;
        }
    }
    fclose(file);

    modoo_cfg->maxColorCount = colorIndex;
    printf("hsv colorIndex1 %d\n", colorIndex);
    printf("hsv colorIndex2 %d\n", modoo_cfg->maxColorCount);

    return 0;
}