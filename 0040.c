/* C言語でニューラルネットワークを作る Lv.4 動的にメモリを確保する */

#include <stdio.h>
#include <stdlib.h>

#define SAMPLES 4

/* パーセプトロン */
typedef struct
{
    /* 入力数 */
    int inputs;

    /* 重み */
    double* w;

    /* バイアス */
    double b;

} Perceptron;


/* パーセプトロン生成 */
Perceptron perceptron_create(int inputs)
{
    Perceptron perceptron;

    perceptron.inputs = inputs;
    perceptron.b = 0.0;

    /* 重みを動的確保 */
    perceptron.w = calloc(inputs, sizeof(double));

    if (perceptron.w == NULL)
        fprintf(stderr, "Memory allocation failed in perceptron_create.\n");

    return perceptron;
}


/* パーセプトロンのメモリ解放 */
void perceptron_destroy(Perceptron* pPerceptron)
{
    free(pPerceptron->w);
    pPerceptron->w = NULL;
}


/* ステップ関数 */
int step(double x)
{
    if (x >= 0)
        return 1;
    else
        return 0;
}


/* 推論 */
int predict(const Perceptron* pNet, const double input[])
{
    double sum = pNet->b;

    for(int i = 0; i < pNet->inputs; i++)
        sum += input[i] * pNet->w[i];

    return step(sum);
}


/* 学習 */
void train(Perceptron* pNet, double x[][2], double t[], double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int i = 0; i < SAMPLES; i++)
        {
            int output = predict(pNet, x[i]);

            /* 誤差 */
            double error = t[i] - output;

            /* 重み更新 */
            for(int j = 0; j < pNet->inputs; j++)
                pNet->w[j] += lr * error * x[i][j];

            /* バイアス更新 */
            pNet->b += lr * error;
        }
    }
}


int main(void)
{
    /* 学習データ（ANDゲート） */
    double x[SAMPLES][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double t[SAMPLES] = {0, 0, 0, 1};

    /* 入力数2でパーセプトロン生成 */
    Perceptron net = perceptron_create(2);

    /* 学習率 */
    double lr = 0.1;

    /* 学習 */
    train(&net, x, t, lr, 100);

    printf("=== 学習結果 ===\n\n");

    for(int i = 0; i < SAMPLES; i++)
        printf("%.0f %.0f -> %d\n", x[i][0], x[i][1], predict(&net, x[i]));

    perceptron_destroy(&net);

    return 0;
}