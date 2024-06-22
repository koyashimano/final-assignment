#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#include "calculations.h"
#include "main.h"

/// 長さmnの行列を(m, n)行列として表示する
/// `x`: (`m`,`n`)行列を表す長さ`m * n`のfloat型配列
void print(int m, int n, const float *x) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", x[i * n + j]);
        }
        printf("\n");
    }
}

/// n次元ベクトルx, oに対してo + xを計算する
/// `x`, `o`: `n`次元ベクトルを表す長さ`n`のfloat型配列
/// 計算結果は`o`に格納する
void add(int n, const float *x, float *o) {
    int i;
    for (i = 0; i < n; i++) {
        o[i] += x[i];
    }
}

/// 長さ`n`の配列`o`の各要素に`x`を掛ける
/// スカラー値`x`とn次元ベクトルoに対してx * oを計算する
/// `x`: float型の値
/// `o`: `n`次元ベクトルを表す長さ`n`のfloat型配列
/// 計算結果は`o`に格納する
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

/// int型の値を持つ長さ`n`の配列`x`をランダムに並び替える
void shuffle(int n, int *x) {
    int i, tmp, random_i;

    for (i = 0; i < n; i++) {
        tmp = x[i];
        random_i = (int)(rand() / (1.0 + RAND_MAX) * n);
        x[i] = x[random_i];
        x[random_i] = tmp;
    }
}

/// 与えられた重みとバイアスでのテストデータに対する正解率を計算する
/// `A1`, `A2`, `A3`: 重み
/// `b1`, `b2`, `b3`: バイアス
/// `test_x`: テストデータ
/// `test_y`: テストデータの正解の数字を格納した配列
/// `test_count`: テストデータの数
/// 返り値: 正解率(%)
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

/// 重みAとバイアスbをファイルに保存する
/// `filename`: 保存先のファイル名
/// `A`: (`m`,`n`)行列を表す長さ`m * n`のfloat型配列
/// `b`: `m`行の列ベクトルを表す長さ`m`のfloat型配列
void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

/// ファイルに保存された重みAとバイアスbを読み込む
/// `filename`: 保存先のファイル名
/// `A`: (`m`,`n`)行列を格納するための長さ`m * n`のfloat型配列
/// `b`: `m`行の列ベクトルを格納するための長さ`m`のfloat型配列
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
