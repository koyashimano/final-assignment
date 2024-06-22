#include "layers.h"

#include <math.h>

/// Ax + bを計算する
/// `x`: `n`行の列ベクトルを表す長さ`n`のfloat型配列
/// `A`: (`m`,`n`)行列を表す長さ`m * n`のfloat型配列
/// `b`: `m`行の列ベクトルを表す長さ`m`のfloat型配列
/// `y`: 計算結果を格納するための長さ`m`のfloat型配列
void fc(int m, int n, const float *x, const float *A, const float *b,
        float *y) {
    int i, j;
    float Ax;
    for (i = 0; i < m; i++) {
        Ax = 0;
        for (j = 0; j < n; j++) {
            Ax += A[i * n + j] * x[j];
        }
        y[i] = Ax + b[i];
    }
}

/// ベクトルxの各要素に対して0より小さい値は0にし、
/// それ以外はそのままにする計算を行う
/// `x`: `n`行の列ベクトルを表す長さ`n`のfloat型配列
/// `y`: 計算結果を格納するための長さ`n`のfloat型配列
void relu(int n, const float *x, float *y) {
    int i;
    for (i = 0; i < n; i++) {
        y[i] = x[i] >= 0 ? x[i] : 0;
    }
}

/// ベクトルxにソフトマックス関数を作用させる
/// `x`: `n`行の列ベクトルを表す長さ`n`のfloat型配列
/// `y`: 計算結果を格納するための長さ`n`のfloat型配列
void softmax(int n, const float *x, float *y) {
    int i;
    float x_max = x[0];
    float exp_sum = 0;

    // get x_max
    for (i = 0; i < n; i++) {
        if (x[i] > x_max) {
            x_max = x[i];
        }
    }

    // calculate exp_sum (denominator)
    for (i = 0; i < n; i++) {
        exp_sum += exp(x[i] - x_max);
    }

    for (i = 0; i < n; i++) {
        y[i] = exp(x[i] - x_max) / exp_sum;
    }
}

/// Softmax層の逆伝播を行う
/// `y`: Softmax層の出力である長さ`n`のfloat型配列
/// `t`: 訓練データの画像中に書かれた数字
/// `dEdx`: 損失の偏微分の計算結果を格納するための長さ`n`のfloat型配列
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
    int i;
    for (i = 0; i < n; i++) {
        dEdx[i] = y[i] - (i == t);
    }
}

/// ReLU層の逆伝播を行う
/// `x`: 順方向の入力である長さ`n`のfloat型配列
/// `dEdy`: 上層の偏微分結果である長さ`n`のfloat型配列
/// `dEdx`: 損失の偏微分の計算結果を格納するための長さ`n`のfloat型配列
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) {
    int i;
    for (i = 0; i < n; i++) {
        dEdx[i] = x[i] > 0 ? dEdy[i] : 0;
    }
}

/// FC層の逆伝播を行う
/// `x`: 順方向の入力である長さ`m`のfloat型配列
/// `dEdy`: 上層の偏微分結果である長さ`n`のfloat型配列
/// `A`: (`m`,`n`)行列を表す長さ`m * n`のfloat型配列
/// `dEdA`, `dEdb`, `dEdx`: 各パラメータによる損失の偏微分の計算結果を
/// 格納するためのfloat型配列
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx) {
    int i, j;
    float dot;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
    }

    for (j = 0; j < m; j++) {
        dEdb[j] = dEdy[j];
    }

    for (i = 0; i < n; i++) {
        dot = 0;
        for (j = 0; j < m; j++) {
            dot += A[i + j * n] * dEdy[j];
        }
        dEdx[i] = dot;
    }
}

/// 出力yを正解と比較して損失を計算する
/// `y`: 長さ`NUM_DIGITS`のfloat型配列
/// `t`: 正解の数字
float cross_entropy_error(const float *y, int t) {
    float eps = 1e-7;
    return -t * log(y[t] + eps);
}
