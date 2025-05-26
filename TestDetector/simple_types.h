#ifndef MODOO_H  // MODOO_H가 정의되지 않았을 때
#define MODOO_H  // MODOO_H를 정의
#define MAX_AREAS 20

// 기본 이미지 구조체 (darknet의 image 구조체 단순화)
typedef struct {
    int w;
    int h;
    int c;
    float* data;
} image;

// 헤더 파일의 실제 내용
// 예를 들어, 함수 선언, 매크로 정의, 구조체 정의 등을 포함할 수 있습니다.
typedef enum {
    TRIGGER_MODE_SW,
    TRIGGER_MODE_HW
} TriggerMode;

typedef enum {
    DETECTION_MODE_ONLY_RULE,
    DETECTION_MODE_ONLY_DEEP,
    DETECTION_MODE_BOTH
} DetectionMode;

typedef enum {
    THRESHOLD_MODE_NONE,
    THRESHOLD_MODE_BINARY,
} ThresholdMode;

// HSV 색상 구조체
typedef struct {
    int h;
    int s;
    int v;
    int id;
} HsvColor;

typedef struct Modoo_cfg {
    TriggerMode triggerMode;
    DetectionMode detectionMode;
    ThresholdMode thresholdMode;
    int binaryValue;
    char* camModelName;
    char* camSerialNum;
    int hsvEnable;
    int MinContourArea;
    int MaxContourArea;
    int VisionID;
    int PortNum;
    int ProductNum;
    double origin_vision_x, origin_vision_y;
    double origin_robot_x, origin_robot_y;
    double res_x, res_y;
    int Max_vision_x, Max_vision_y;
    int FileNameBuff;
    double boundaryBuff_minX;
    double boundaryBuff_minY;
    double boundaryBuff_maxX;
    double boundaryBuff_maxY;
    double areaThresholdPercentLower;
    double areaThresholdPercentUpper;
    int areas[MAX_AREAS]; // 추가: area1 ~ area20을 위한 배열;
    int minMarkArea;
    int maxMarkArea;
    HsvColor hsvColors[1024];
    int maxColorCount;
    int HsvBufferH;
    int HsvBufferS;
    int HsvBufferV;
    int debugMode;
    int productNumColor[20];
    int minSUMMarkArea;
    int maxSUMMarkArea;
    int BlackTagNum;
    int blackMinMarkArea;
    int blackMaxMarkArea;
    int BlackEllipseMinSize;
    int BlackEllipseMaxSize;
    int isDetect[20];
} Modoo_cfg;

int readMainSetFromFile(const char* filename, Modoo_cfg* modoo_cfg);

// 기본 함수들
image make_image(int w, int h, int c);
void free_image(image m);

#endif // #ifndef MODOO_H의 끝