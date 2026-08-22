/* C言語でニューラルネットワークを作る Lv.5 ベクトル演算を導入する */

#include <stdio.h>
#include <stdlib.h>

#define SAMPLES 4

/* ===== Vector ===== */

/* ベクトル */
 typedef struct
{
    /* サイズ */
    int size;

    /* データ */
    double* data;

} Vector;


/* ベクトル生成 */
Vector vector_create(int size)
{
    Vector vector;

    vector.size = size;
    vector.data = calloc(size, sizeof(double));

    if (vector.data == NULL)
        fprintf(stderr, "Memory allocation failed in vector_create.\n");

    return vector;
}


/* ベクトルのメモリ解放 */
void vector_destroy(Vector* pVector)
{
    free(pVector->data);
    pVector->data = NULL;
}

/* ===== Perceptron ===== */

/* パーセプトロン */
typedef struct
{
    /* 重み */
    Vector w;
    
    /* バイアス */
    double b;
    
} Perceptron;


/* パーセプトロン生成 */
Perceptron perceptron_create(int inputs)
{
    Perceptron perceptron;
    
    perceptron.w = vector_create(inputs);
    perceptron.b = 0.0;
    
    return perceptron;
}


/* パーセプトロンのメモリ解放 */
void perceptron_destroy(Perceptron* pPerceptron)
{
    vector_destroy(&pPerceptron->w);
}

/* ===== 活性化関数 ===== */

/* ステップ関数 */
int step(double x)
{
    if (x >= 0)
        return 1;
    else
        return 0;
}

/* ===== 演算 ===== */

/* 内積 */
double dot(const Vector* a, const double x[])
{
    double sum = 0.0;

    for(int i = 0; i < a->size; i++)
        sum += a->data[i] * x[i];

    return sum;
}

/* ===== ニューラルネットワーク ===== */

/* 推論 */
int predict(const Perceptron* pNet, const double input[])
{
    return step(dot(&pNet->w, input) + pNet->b);
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
            for(int j = 0; j < pNet->w.size; j++)
                pNet->w.data[j] += lr * error * x[i][j];

            /* バイアス更新 */
            pNet->b += lr * error;
        }
    }
}


/* 開始 */
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