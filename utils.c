#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#include "calculations.h"
#include "main.h"

/// float型の1次元配列`x`として与えられた(`m`,`n`)行列を表示する
void print(int m, int n, const float *x) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", x[i * n + j]);
        }
        printf("\n");
    }
}

/// 長さ`n`の配列`o`に`x`を足す
void add(int n, const float *x, float *o) {
    int i;
    for (i = 0; i < n; i++) {
        o[i] += x[i];
    }
}

/// 長さ`n`の配列`o`の各要素に`x`を掛ける
void scale(int n, float x, float *o) {
    int i;
    for (i = 0; i < n; i++) {
        o[i] *= x;
    }
}

/// 長さ`n`の配列`o`の値を全て`x`にする
void init(int n, float x, float *o) {
    int i;
    for (i = 0; i < n; i++) {
        o[i] = x;
    }
}

/// 長さ`n`の配列`o`の各要素を-1から1の間のランダムなfloat型の値にする
void rand_init(int n, float *o) {
    int i;
    for (i = 0; i < n; i++) {
        o[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
}

/// int型の値を持つ`n`行の列ベクトル`x`をランダムに並び替える
void shuffle(int n, int *x) {
    int i, tmp, random_i;

    for (i = 0; i < n; i++) {
        tmp = x[i];
        random_i = (int)(rand() / (1.0 + RAND_MAX) * n);
        x[i] = x[random_i];
        x[random_i] = tmp;
    }
}

/// 正解率を計算する
float correct_rate(const float *A1, const float *b1, const float *A2,
                   const float *b2, const float *A3, const float *b3,
                   const float *test_x, const unsigned char *test_y,
                   int test_count) {
    int sum = 0, i;
    unsigned char inference_result;
    for (i = 0; i < test_count; i++) {
        inference_result =
            inference6(A1, b1, A2, b2, A3, b3, test_x + IMAGE_SIZE * i);
        if (inference_result == test_y[i]) {
            sum++;
        }
    }

    return sum * 100.0 / test_count;
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

void load(const char *filename, int m, int n, float *A, float *b) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("file %s cannot be opened\n", filename);
        return;
    }

    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);

    fclose(fp);
}
