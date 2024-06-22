#include "main.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "calculations.h"
#include "layers.h"
#include "nn.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        printf("number recognition by deep-learning\n\n");
        printf("commands:\n train\n inference\n");
        return 0;
    }

    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    const char *data_file_names[] = {"fc1.dat", "fc2.dat", "fc3.dat"};

    srand(time(NULL));

    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count,
               &width, &height);

    if (strcmp(argv[1], "train") == 0) {
        train(train_count, train_x, train_y, test_x, test_y, test_count,
              data_file_names);
        return 0;
    }

    if (strcmp(argv[1], "inference") == 0) {
        if (argc == 2) {
            printf("specify the bpm image file\n");
            return 1;
        }
        float *x = load_mnist_bmp(argv[2]);
        printf("%d\n", inference(x, data_file_names));

        return 0;
    }

    printf("invalid command\n");
    return 1;
}

void train(int train_count, const float *train_x, const unsigned char *train_y,
           const float *test_x, const unsigned char *test_y, int test_count,
           const char **data_file_names) {
    int epoch_num = 10;
    int n = 100;     // mini batch size
    float lr = 0.1;  // learning rate
    int N = train_count;

    float *A1 = malloc(sizeof(float) * 50 * IMAGE_SIZE);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * DIM_Y2 * DIM_Y1);
    float *b2 = malloc(sizeof(float) * DIM_Y2);
    float *A3 = malloc(sizeof(float) * NUM_DIGITS * DIM_Y2);
    float *b3 = malloc(sizeof(float) * NUM_DIGITS);
    float *dEdA1 = malloc(sizeof(float) * DIM_Y1 * IMAGE_SIZE);
    float *dEdb1 = malloc(sizeof(float) * DIM_Y1);
    float *dEdA2 = malloc(sizeof(float) * DIM_Y2 * DIM_Y1);
    float *dEdb2 = malloc(sizeof(float) * DIM_Y2);
    float *dEdA3 = malloc(sizeof(float) * NUM_DIGITS * DIM_Y2);
    float *dEdb3 = malloc(sizeof(float) * NUM_DIGITS);
    float *ave_dEdA1 = malloc(sizeof(float) * DIM_Y1 * IMAGE_SIZE);
    float *ave_dEdb1 = malloc(sizeof(float) * DIM_Y1);
    float *ave_dEdA2 = malloc(sizeof(float) * DIM_Y2 * DIM_Y1);
    float *ave_dEdb2 = malloc(sizeof(float) * DIM_Y2);
    float *ave_dEdA3 = malloc(sizeof(float) * NUM_DIGITS * DIM_Y2);
    float *ave_dEdb3 = malloc(sizeof(float) * NUM_DIGITS);
    float *y = malloc(sizeof(float) * NUM_DIGITS);
    int indexes[N];
    int index, i, j, k;

    rand_init(DIM_Y1 * IMAGE_SIZE, A1);
    rand_init(DIM_Y1, b1);
    rand_init(DIM_Y2 * DIM_Y1, A2);
    rand_init(DIM_Y2, b2);
    rand_init(NUM_DIGITS * DIM_Y2, A3);
    rand_init(NUM_DIGITS, b3);

    for (i = 0; i < epoch_num; i++) {
        for (j = 0; j < N; j++) {
            indexes[j] = j;
        }
        shuffle(N, indexes);

        for (j = 0; j < N / n; j++) {
            init(DIM_Y1 * IMAGE_SIZE, 0, dEdA1);
            init(DIM_Y1, 0, dEdb1);
            init(DIM_Y2 * DIM_Y1, 0, dEdA2);
            init(DIM_Y2, 0, dEdb2);
            init(NUM_DIGITS * DIM_Y2, 0, dEdA3);
            init(NUM_DIGITS, 0, dEdb3);

            for (k = 0; k < n; k++) {
                index = indexes[j * n + k];
                backward6(A1, b1, A2, b2, A3, b3, train_x + IMAGE_SIZE * index,
                          train_y[index], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3,
                          dEdb3);
                add(DIM_Y1 * IMAGE_SIZE, dEdA1, ave_dEdA1);
                add(DIM_Y1, dEdb1, ave_dEdb1);
                add(DIM_Y2 * DIM_Y1, dEdA2, ave_dEdA2);
                add(DIM_Y2, dEdb2, ave_dEdb2);
                add(NUM_DIGITS * DIM_Y2, dEdA3, ave_dEdA3);
                add(NUM_DIGITS, dEdb3, ave_dEdb3);
            }

            optimize(DIM_Y1 * IMAGE_SIZE, lr, n, ave_dEdA1, A1);
            optimize(DIM_Y1, lr, n, ave_dEdb1, b1);
            optimize(DIM_Y2 * DIM_Y1, lr, n, ave_dEdA2, A2);
            optimize(DIM_Y2, lr, n, ave_dEdb2, b2);
            optimize(NUM_DIGITS * DIM_Y2, lr, n, ave_dEdA3, A3);
            optimize(NUM_DIGITS, lr, n, ave_dEdb3, b3);
        }
        printf(
            "[epoch%3d] loss: %9f  correct answer rate: %9f%%\n", i + 1,
            cross_entropy_error(y, train_y[index]),
            correct_rate(A1, b1, A2, b2, A3, b3, test_x, test_y, test_count));
    }

    save(data_file_names[0], DIM_Y1, IMAGE_SIZE, A1, b1);
    save(data_file_names[1], DIM_Y2, DIM_Y1, A2, b2);
    save(data_file_names[2], NUM_DIGITS, DIM_Y2, A3, b3);

    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    free(dEdA1);
    free(dEdb1);
    free(dEdA2);
    free(dEdb2);
    free(dEdA3);
    free(dEdb3);
    free(ave_dEdA1);
    free(ave_dEdb1);
    free(ave_dEdA2);
    free(ave_dEdb2);
    free(ave_dEdA3);
    free(ave_dEdb3);
    free(y);
}

int inference(const float *x, const char **data_file_names) {
    float *A1 = malloc(sizeof(float) * 50 * IMAGE_SIZE);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * DIM_Y2 * DIM_Y1);
    float *b2 = malloc(sizeof(float) * DIM_Y2);
    float *A3 = malloc(sizeof(float) * NUM_DIGITS * DIM_Y2);
    float *b3 = malloc(sizeof(float) * NUM_DIGITS);
    int ans;

    load(data_file_names[0], 50, 784, A1, b1);
    load(data_file_names[1], 100, 50, A2, b2);
    load(data_file_names[2], 10, 100, A3, b3);

    ans = inference6(A1, b1, A2, b2, A3, b3, x);

    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);

    return ans;
}
