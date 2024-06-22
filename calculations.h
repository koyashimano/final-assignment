#ifndef CALCULATIONS_H_
#define CALCULATIONS_H_

void backward6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x, unsigned char t, float *y, float *dEdA1,
               float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3,
               float *dEdb3);
int inference6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x);

#endif  // CALCULATIONS_H_
