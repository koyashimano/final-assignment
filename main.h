#ifndef MAIN_H_
#define MAIN_H_

#define IMAGE_SIZE 28 * 28
#define NUM_DIGITS 10
#define DIM_Y1 50
#define DIM_Y2 100

int main(int argc, char *argv[]);
void train(const char **data_file_names);
int inference(const float *x, const char **data_file_names);

#endif  // MAIN_H_
