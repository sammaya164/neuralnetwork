/* C言語でニューラルネットワークを作る Lv.3 構造体でデータをまとめる */

#include <stdio.h>

#define INPUTS 2
#define SAMPLES 4

/* パーセプトロン */
typedef struct
{
    /* 重み */
    double w[INPUTS];

    /* バイアス */
    double b;

} Perceptron;


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

    for(int i = 0; i < INPUTS; i++)
    {
        sum += input[i] * pNet->w[i];
    }

    return step(sum);
}


/* 学習 */
void train(Perceptron* pNet, double x[][INPUTS], double t[], double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int i = 0; i < SAMPLES; i++)
        {
            int output = predict(pNet, x[i]);

            /* 誤差 */
            double error = t[i] - output;

            /* 重み更新 */
            for(int j = 0; j < INPUTS; j++)
            {
                pNet->w[j] += lr * error * x[i][j];
            }

            /* バイアス更新 */
            pNet->b += lr * error;
        }
    }
}


int main(void)
{
    /* 学習データ（ANDゲート） */
    double x[SAMPLES][INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double t[SAMPLES] = {0, 0, 0, 1};

    /* パーセプトロン生成 */
    Perceptron net = {{0.2, -0.1}, 0.0};

    /* 学習率 */
    double lr = 0.1;

    /* 学習 */
    train(&net, x, t, lr, 100);

    printf("=== 学習結果 ===\n\n");

    for(int i = 0; i < SAMPLES; i++)
    {
        int output = predict(&net, x[i]);

        printf("%.0f %.0f -> %d\n", x[i][0], x[i][1], output);
    }

    return 0;
}