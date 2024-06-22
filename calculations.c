#include "calculations.h"

#include <stdlib.h>

#include "layers.h"
#include "main.h"

/// 誤差逆伝播によって勾配を計算する
/// `A1`, `A2`, `A3`: 重み
/// `b1`, `b2`, `b3`: バイアス
/// `x`: 訓練データ
/// `t`: 訓練データが表す数字
/// `y`: 出力
/// `dEdA1`, `dEdb1`, `dEdA2`, `dEdb2`, `dEdA3`, `dEdb3`:
/// 損失関数の各パラメータによる偏微分を格納するための配列
void backward6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x, unsigned char t, float *y, float *dEdA1,
               float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3,
               float *dEdb3) {
    float y1[DIM_Y1], y2[DIM_Y2];
    float dEdy[NUM_DIGITS], dEdx3[DIM_Y2], dEdx2[DIM_Y1], dEdx1[IMAGE_SIZE];

    fc(DIM_Y1, IMAGE_SIZE, x, A1, b1, y1);
    relu(DIM_Y1, y1, y1);
    fc(DIM_Y2, DIM_Y1, y1, A2, b2, y2);
    relu(DIM_Y2, y2, y2);
    fc(NUM_DIGITS, DIM_Y2, y2, A3, b3, y);
    softmax(NUM_DIGITS, y, y);

    softmaxwithloss_bwd(NUM_DIGITS, y, t, dEdy);
    fc_bwd(NUM_DIGITS, DIM_Y2, y2, dEdy, A3, dEdA3, dEdb3, dEdx3);
    relu_bwd(DIM_Y2, y2, dEdx3, dEdx3);
    fc_bwd(DIM_Y2, DIM_Y1, y1, dEdx3, A2, dEdA2, dEdb2, dEdx2);
    relu_bwd(DIM_Y1, y1, dEdx2, dEdx2);
    fc_bwd(DIM_Y1, IMAGE_SIZE, x, dEdx2, A1, dEdA1, dEdb1, dEdx1);
}

/// データとして与えられた画像中の数字を認識する
/// `A1`, `A2`, `A3`: 重み
/// `b1`, `b2`, `b3`: バイアス
/// `x`: 画像を配列に変換したもの
/// 返り値: 画像に書かれている数字の推論結果
int inference6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x) {
    float y1[DIM_Y1], y2[DIM_Y2], y3[NUM_DIGITS];
    int ans = 0, i;

    fc(DIM_Y1, IMAGE_SIZE, x, A1, b1, y1);
    relu(DIM_Y1, y1, y1);
    fc(DIM_Y2, DIM_Y1, y1, A2, b2, y2);
    relu(DIM_Y2, y2, y2);
    fc(NUM_DIGITS, DIM_Y2, y2, A3, b3, y3);
    softmax(NUM_DIGITS, y3, y3);

    for (i = 0; i < NUM_DIGITS; i++) {
        if (y3[i] > y3[ans]) {
            ans = i;
        }
    }

    return ans;
}
