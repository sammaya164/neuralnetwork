/* C言語でニューラルネットワークを作る Lv.7 学習データをMatrixで表現する */

#include <stdio.h>
#include <stdlib.h>

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


/* 行列の要素に足し算 */
void matrix_add(Matrix* pMatrix, int r, int c, double value)
{
    pMatrix->data[r * pMatrix->cols + c] += value;
}

/* ===== Perceptron ===== */

/* パーセプトロン */
typedef struct
{
    /* 重み */
    Matrix w;
    
    /* バイアス */
    Vector b;
    
} Perceptron;


/* パーセプトロン生成 */
Perceptron perceptron_create(int inputs)
{
    Perceptron perceptron;
    
    perceptron.w = matrix_create(1, inputs);
    perceptron.b = vector_create(1);

    return perceptron;
}


/* パーセプトロンのメモリ解放 */
void perceptron_destroy(Perceptron* pPerceptron)
{
    matrix_destroy(&pPerceptron->w);
    vector_destroy(&pPerceptron->b);
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


/* 行列の1行をベクトルとして取得 */
Vector matrix_get_row(const Matrix* pMatrix, int row)
{
    Vector v = vector_create(pMatrix->cols);

    for(int c = 0; c < pMatrix->cols; c++)
        v.data[c] = matrix_get(pMatrix, row, c);

    return v;
}

/* ===== パーセプトロン ===== */

/* 推論 */
int predict(const Perceptron* pNet, const Vector* x)
{
    Vector y = vector_create(pNet->w.rows);

    /* y = W x */
    matrix_mul_vector(&pNet->w, x, &y);

    /* バイアスを加える */
    for(int i = 0; i < y.size; i++)
        y.data[i] += pNet->b.data[0];

    /* 活性化関数を適用 */
    int result = step(y.data[0]);

    /* Vectorを解放 */
    vector_destroy(&y);

    return result;
}


/* 学習 */
void train(Perceptron* pNet, Matrix* x, Vector* t, double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int i = 0; i < x->rows; i++)
        {
            /* Matrixのi行目をVectorとして取得 */
            Vector input = matrix_get_row(x, i);

            /* 推論 */
            int output = predict(pNet, &input);

            /* 誤差 */
            double error = t->data[i] - output;

            /* 重み更新 */
            for(int j = 0; j < pNet->w.cols; j++)
                matrix_add(&pNet->w, 0, j, lr * error * input.data[j]);

            /* バイアス更新 */
            pNet->b.data[0] += lr * error;
            
            /* Vectorを解放 */
            vector_destroy(&input);
        }
    }
}


/* 開始 */
int main(void)
{
    /* 学習データ（ANDゲート） */
    Matrix x = matrix_create(4, 2);

    double data[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 2; j++)
            matrix_set(&x, i, j, data[i][j]);

    /* 教師データ */
    Vector t = vector_create(4);

    t.data[0] = 0;
    t.data[1] = 0;
    t.data[2] = 0;
    t.data[3] = 1;

    /* 入力数2でパーセプトロン生成 */
    Perceptron net = perceptron_create(2);

    /* 学習率 */
    double lr = 0.1;

    /* 学習 */
    train(&net, &x, &t, lr, 100);

    printf("=== 学習結果 ===\n\n");

    for(int i = 0; i < x.rows; i++)
    {
        Vector input = matrix_get_row(&x, i);
        
        int output = predict(&net, &input);

        printf("%.0f %.0f -> %d\n", input.data[0], input.data[1], output);

        vector_destroy(&input);
    }

    /* メモリを解放 */
    perceptron_destroy(&net);
    matrix_destroy(&x);
    vector_destroy(&t);

    return 0;
}