/* C言語でニューラルネットワークを作る Lv.1 パーセプトロンの最小実装 */

#include <stdio.h>

#define INPUTS 2
#define SAMPLES 4

int main(void)
{
    /* 学習データ（ANDゲート） */
    double x[SAMPLES][INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double t[SAMPLES] = {0, 0, 0, 1};

    /* 重み */
    double w[INPUTS] = {0.2, -0.1};

    /* バイアス */
    double b = 0.0;

    /* 学習率 */
    double lr = 0.1;

    /* 学習回数 */
    int epoch = 100;

    /* 学習 */
    for(int e = 0; e < epoch; e++)
    {
        for(int i = 0; i < SAMPLES; i++)
        {
            /* 重み付き和 */
            double sum = b;

            for(int j = 0; j < INPUTS; j++)
            {
                sum += x[i][j] * w[j];
            }

            /* ステップ関数 */
            int output;

            if(sum >= 0)
                output = 1;
            else
                output = 0;

            /* 誤差 */
            double error = t[i] - output;

            /* 重み更新 */
            for(int j = 0; j < INPUTS; j++)
            {
                w[j] += lr * error * x[i][j];
            }

            /* バイアス更新 */
            b += lr * error;
        }
    }

    printf("=== 学習結果 ===\n\n");

    for(int i = 0; i < SAMPLES; i++)
    {
        double sum = b;

        for(int j = 0; j < INPUTS; j++)
        {
            sum += x[i][j] * w[j];
        }

        int output;

        if(sum >= 0)
            output = 1;
        else
            output = 0;

        printf("%.0f %.0f -> %d\n", x[i][0], x[i][1], output);
    }

    return 0;
}