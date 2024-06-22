#include "layers.h"

#include <math.h>

/// float型の1次元配列`A`として与えられた(`m`,`n`)行列、
/// float型の1次元配列として与えられた`m`行の列ベクトル`b`、
/// float型の1次元配列として与えられた`n`行の列ベクトル`x`
/// が入力されたときに、Ax + bを計算し、その結果を
/// float型の1次元配列として与えられた`m`行の列ベクトル`y`に書き込む
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

/// float型の1次元配列として与えられた`n`行の列ベクトル`x`が入力されたときに、
/// 0より小さい値は0にし、それ以外はそのままにする計算を行い、
/// float型の1次元配列として与えられた`n`行の列ベクトル`y`に書き込む
void relu(int n, const float *x, float *y) {
    int i;
    for (i = 0; i < n; i++) {
        y[i] = x[i] >= 0 ? x[i] : 0;
    }
}

/// float型の1次元配列として与えられた`n`行の列ベクトル`x`に
/// ソフトマックス関数を作用させ、その結果を
/// float型の1次元配列として与えられた`n`行の列ベクトル`y`に書き込む
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

/// Softmax層逆伝播:
/// float型の1次元配列として与えられた`n`行の列ベクトル`y`、
/// 正解を表す数字tに対して、偏微分dE/dxを計算する
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
    int i;
    for (i = 0; i < n; i++) {
        dEdx[i] = y[i] - (i == t);
    }
}

/// ReLU層: `dEdy` -> `dEdx`
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) {
    int i;
    for (i = 0; i < n; i++) {
        dEdx[i] = x[i] > 0 ? dEdy[i] : 0;
    }
}

/// FC層: float型の1次元配列`A`として与えられた(`m`,`n`)行列、
/// `x`: (1, `m`)
/// `dEdy`: (`n`, 1)
/// float型の1次元配列`dEdA`として(`m`,`n`)行列
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

/// 損失関数: NUM_DIGITS行の列ベクトル`y`, `t`
float cross_entropy_error(const float *y, int t) {
    float eps = 1e-7;
    return -t * log(y[t] + eps);
}
