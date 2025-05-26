// simple_image.c - image ����ü ���� �Լ��� ����
#include <stdlib.h>
#include <string.h>
#include "simple_types.h"

// �̹��� ���� �Լ�
image make_image(int w, int h, int c) {
    image img;
    img.w = w;
    img.h = h;
    img.c = c;

    // �޸� �Ҵ�
    img.data = (float*)calloc(w * h * c, sizeof(float));

    return img;
}

// �̹��� �޸� ���� �Լ�
void free_image(image m) {
    if (m.data) {
        free(m.data);
        m.data = NULL;
    }
}