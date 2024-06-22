#ifndef LAYERS_H_
#define LAYERS_H_

void fc(int m, int n, const float *x, const float *A, const float *b, float *y);
void relu(int n, const float *x, float *y);
void softmax(int n, const float *x, float *y);
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx);
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx);
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx);
float cross_entropy_error(const float *y, int t);

#endif  // LAYERS_H_
