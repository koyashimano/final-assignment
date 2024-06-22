#ifndef UTILS_H_
#define UTILS_H_

void print(int m, int n, const float *x);
void add(int n, const float *x, float *o);
void scale(int n, float x, float *o);
void optimize(int size, float lr, int batch_size, float *ave_dEdx, float *x);
void init(int n, float x, float *o);
void rand_init(int n, float *o);
void shuffle(int n, int *x);
float correct_rate(const float *A1, const float *b1, const float *A2,
                   const float *b2, const float *A3, const float *b3,
                   const float *test_x, const unsigned char *test_y,
                   int test_count);
void save(const char *filename, int m, int n, const float *A, const float *b);
void load(const char *filename, int m, int n, float *A, float *b);

#endif  // UTILS_H_
