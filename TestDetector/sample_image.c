// simple_image.c - image 구조체 관련 함수들 구현
#include <stdlib.h>
#include <string.h>
#include "simple_types.h"

// 이미지 생성 함수
image make_image(int w, int h, int c) {
    image img;
    img.w = w;
    img.h = h;
    img.c = c;

    // 메모리 할당
    img.data = (float*)calloc(w * h * c, sizeof(float));

    return img;
}

// 이미지 메모리 해제 함수
void free_image(image m) {
    if (m.data) {
        free(m.data);
        m.data = NULL;
    }
}