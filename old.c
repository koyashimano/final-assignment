
void train3(int train_count, const float *train_x, const unsigned char *train_y,
            const float *test_x, const unsigned char *test_y, int test_count) {
    int epoch_num = 10;
    int n = 100;     // ミニバッチサイズ
    float lr = 0.1;  // learning rate
    int N = train_count;

    float *A = malloc(sizeof(float) * NUM_DIGITS * IMAGE_SIZE);
    float *b = malloc(sizeof(float) * NUM_DIGITS);
    float *y = malloc(sizeof(float) * NUM_DIGITS);
    float *dEdA = malloc(sizeof(float) * NUM_DIGITS * IMAGE_SIZE);
    float *dEdb = malloc(sizeof(float) * NUM_DIGITS);
    float *ave_dEdA = malloc(sizeof(float) * NUM_DIGITS * IMAGE_SIZE);
    float *ave_dEdb = malloc(sizeof(float) * NUM_DIGITS);
    int indexes[N];
    int index, i, j, k;

    rand_init(NUM_DIGITS * IMAGE_SIZE, A);
    rand_init(NUM_DIGITS, b);

    for (i = 0; i < epoch_num; i++) {
        for (j = 0; j < N; j++) {
            indexes[j] = j;
        }
        shuffle(N, indexes);

        for (j = 0; j < N / n; j++) {
            init(NUM_DIGITS * IMAGE_SIZE, 0, ave_dEdA);
            init(NUM_DIGITS, 0, ave_dEdb);

            for (k = 0; k < n; k++) {
                index = indexes[j * n + k];
                backward3(A, b, train_x + IMAGE_SIZE * index, train_y[index], y,
                          dEdA, dEdb);
                add(NUM_DIGITS * IMAGE_SIZE, dEdA, ave_dEdA);
                add(NUM_DIGITS, dEdb, ave_dEdb);
            }

            scale(NUM_DIGITS * IMAGE_SIZE, -lr / n, ave_dEdA);
            scale(NUM_DIGITS, -lr / n, ave_dEdb);

            add(NUM_DIGITS * IMAGE_SIZE, ave_dEdA, A);
            add(NUM_DIGITS, ave_dEdb, b);
        }
        // printf("[epoch%3d] loss: %9f  correct answer rate: %9f%%\n",
        //        i + 1,
        //        cross_entropy_error(y, train_y[index]),
        //        correct_rate(A, b, test_x, test_y, test_count));
    }

    free(A);
    free(b);
    free(y);
    free(dEdA);
    free(dEdb);
    free(ave_dEdA);
    free(ave_dEdb);
}

/// float型の1次元配列として与えられたNUM_DIGITS×IMAGE_SIZE行列`A`、
/// float型の1次元配列として与えられたNUM_DIGITS行の列ベクトル`b`、
/// float型の1次元配列として与えられたIMAGE_SIZE行の列ベクトル`x`が入力
/// されたときに、fc, relu, softmaxを順に計算し、
/// 出力yの各要素のうち最大となるものの添字[0:9]を返す
int inference3(const float *A, const float *b, const float *x) {
    float y[NUM_DIGITS];
    int ans = 0, i;

    fc(NUM_DIGITS, IMAGE_SIZE, x, A, b, y);
    relu(NUM_DIGITS, y, y);
    softmax(NUM_DIGITS, y, y);

    for (i = 0; i < NUM_DIGITS; i++) {
        if (y[i] > y[ans]) {
            ans = i;
        }
    }

    return ans;
}

/// 誤差逆伝播テスト
// void backward3(const float *A, const float *b, const float *x, unsigned char
// t, float *y, float *dEdA, float *dEdb)
// {
//     float dEdy[2], dEdx[3], *x_relu;

//     fc(2, 3, x, A, b, y);
//     relu(2, y, y);
//     x_relu = y;
//     softmax(2, x_relu, y);

//     softmaxwithloss_bwd(2, y, t, dEdy);
//     relu_bwd(2, x_relu, dEdy, dEdy);
//     fc_bwd(3, 2, x, dEdy, A, dEdA, dEdb, dEdx);
// }
/// 誤差逆伝播
void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb) {
    float dEdy[NUM_DIGITS], dEdx[IMAGE_SIZE], *x_relu;

    fc(NUM_DIGITS, IMAGE_SIZE, x, A, b, y);
    relu(NUM_DIGITS, y, y);
    x_relu = y;
    softmax(NUM_DIGITS, x_relu, y);

    softmaxwithloss_bwd(NUM_DIGITS, y, t, dEdy);
    relu_bwd(NUM_DIGITS, x_relu, dEdy, dEdy);
    fc_bwd(IMAGE_SIZE, NUM_DIGITS, x, dEdy, A, dEdA, dEdb, dEdx);
}
