/* C言語でニューラルネットワークを作る Lv.6 行列演算を導入する */

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

/* ===== Matrix ===== */

/* 行列 */
typedef struct
{
    /* 行数 */
    int rows;

    /* 列数 */
    int cols;

    /* データ */
    double* data;

} Matrix;


/* 行列生成 */
Matrix matrix_create(int rows, int cols)
{
    Matrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;

    matrix.data = calloc(rows * cols, sizeof(double));

    if (matrix.data == NULL)
        fprintf(stderr, "Memory allocation failed in matrix_create.\n");

    return matrix;
}


/* 行列のメモリ解放 */
void matrix_destroy(Matrix* pMatrix)
{
    free(pMatrix->data);
    pMatrix->data = NULL;
}


/* 行列の要素取得 */
double matrix_get(const Matrix* pMatrix, int r, int c)
{
    return pMatrix->data[r * pMatrix->cols + c];
}


/* 行列の要素設定 */
void matrix_set(Matrix* pMatrix, int r, int c, double value)
{
    pMatrix->data[r * pMatrix->cols + c] = value;
}

/* ===== Perceptron ===== */

/* パーセプトロン */
typedef struct
{
    /* 重み */
    Matrix w;
    
    /* バイアス */
    double b;
    
} Perceptron;


/* パーセプトロン生成 */
Perceptron perceptron_create(int inputs)
{
    Perceptron perceptron;
    
    perceptron.w = matrix_create(1, inputs);
    perceptron.b = 0.0;

    return perceptron;
}


/* パーセプトロンのメモリ解放 */
void perceptron_destroy(Perceptron* pPerceptron)
{
    matrix_destroy(&pPerceptron->w);
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

/* Matrix × Vector */
void matrix_mul_vector(const Matrix* pMatrix, const Vector* x, Vector* y)
{
    for(int r=0; r < pMatrix->rows; r++)
    {
        double sum = 0;

        for(int c=0; c < pMatrix->cols; c++)
            sum += matrix_get(pMatrix, r, c) * x->data[c];

        y->data[r] = sum;
    }
}

/* ===== パーセプトロン ===== */

/* 推論 */
int predict(const Perceptron* pNet, const double input[])
{
    Vector x = vector_create(pNet->w.cols);
    Vector y = vector_create(pNet->w.rows);

    /* 配列をベクトルxに移す */
    for(int i=0; i < pNet->w.cols; i++)
        x.data[i] = input[i];

    /* y = W x */
    matrix_mul_vector(&pNet->w, &x, &y);

    /* バイアスを加える */
    y.data[0] += pNet->b;

    /* 活性化関数を適用 */
    int result = step(y.data[0]);

    /* Vectorを解放 */
    vector_destroy(&x);
    vector_destroy(&y);

    return result;
}


/* 学習 */
void train(Perceptron* pNet, double x[][2], double t[], double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int i = 0; i < SAMPLES; i++)
        {
            /* 推論 */
            int output = predict(pNet, x[i]);

            /* 誤差 */
            double error = t[i] - output;

            /* 重み更新 */
            for(int j = 0; j < pNet->w.cols; j++)
            {
                double weight = matrix_get(&pNet->w, 0, j);
                matrix_set(&pNet->w, 0, j, weight + lr * error * x[i][j]);
            }

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