/* C言語でニューラルネットワークを作る Lv.10 損失と微分を計算する */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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


/* 行列の要素に値を加える */
void matrix_add(Matrix* pMatrix, int r, int c, double value)
{
    pMatrix->data[r * pMatrix->cols + c] += value;
}

/* ===== Operation ===== */

/* Matrix × Vector */
void matrix_mul_vector(const Matrix* pMatrix, const Vector* pX, Vector* pY)
{
    for(int r = 0; r < pMatrix->rows; r++)
    {
        double sum = 0;
        
        for(int c = 0; c < pMatrix->cols; c++)
            sum += matrix_get(pMatrix, r, c) * pX->data[c];
        
        pY->data[r] = sum;
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

/* ===== 活性化関数 ===== */

/* ステップ関数 */
double step(double x)
{
    if (x >= 0)
        return 1.0;
    else
        return 0.0;
}


/* シグモイド関数 */
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/* ===== 損失関数 ===== */

/*
    二乗和誤差 (Sum of Squared Errors)

        L = (1/2) * Σ(y - t)^2

        t : 正解値
        y :  推論値
*/
double sse(const Vector* pY, const Vector* pT)
{
    double sum = 0.0;

    for(int i = 0; i < pT->size; i++)
    {
        double error = pT->data[i] - pY->data[i];
        sum += error * error;
    }

    return 0.5 * sum;
}


/*
    二乗和誤差の微分

        dL/dy = y - t

        t : 正解値
        y :  推論値
*/
void sse_derivative(const Vector* pY, const Vector* pT, Vector* pDy)
{
    for(int i = 0; i < pT->size; i++)
        pDy->data[i] = pY->data[i] - pT->data[i];
}

/* ===== Linear Layer ===== */

/* Linear */
typedef struct
{
    /* 重み */
    Matrix weight;
    
    /* バイアス */
    Vector bias;
    
} Linear;


/* Linear生成 */
Linear linear_create(int in, int out)
{
    Linear linear;
    
    linear.weight = matrix_create(out, in);
    linear.bias = vector_create(out);
    
    return linear;
}


/* Linearのメモリ解放 */
void linear_destroy(Linear* pLinear)
{
    matrix_destroy(&pLinear->weight);
    vector_destroy(&pLinear->bias);
}


/* Linear Forward */
void linear_forward(const Linear* pLinear, const Vector* pInput, Vector* pOutput)
{
    /* y = Wx */
    matrix_mul_vector(&pLinear->weight, pInput, pOutput);

    /* バイアスを加える */
    for(int i = 0; i < pOutput->size; i++)
        pOutput->data[i] += pLinear->bias.data[i];
}

/* ===== Activation Layer ===== */

/* Activation Func */
typedef double (*ActivationFunc)(double);


/* Activation */
typedef struct
{
    ActivationFunc forward;

} Activation;


/* Activation Forward */
void activation_forward(const Activation* pAct, Vector* pInput, Vector* pOutput)
{
    for(int i = 0; i < pInput->size; i++)
        pOutput->data[i] = pAct->forward(pInput->data[i]);
}

/* ===== パーセプトロン ===== */

/* 推論 */
void predict(const Linear* pLinear, const Activation* pAct, const Vector* pX, Vector* pY)
{
    /* y = W x + b*/
    linear_forward(pLinear, pX, pY);

    /* 活性化関数を適用 */
    activation_forward(pAct, pY, pY);
}


/* 学習 */
void train(Linear* pLinear, Activation* pAct,
    const Matrix* x, const Matrix* t, double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int r = 0; r < x->rows; r++)
        {
            /* Matrixのr行目をVectorとして取得 */
            Vector input = matrix_get_row(x, r);
            Vector teach = matrix_get_row(t, r);

            Vector output = vector_create(pLinear->weight.rows);
            Vector gradient = vector_create(pLinear->weight.rows);

            /* 推論 */
            predict(pLinear, pAct, &input, &output);

            /* 損失の微分 */
            sse_derivative(&output, &teach, &gradient);

            for(int i = 0; i < pLinear->weight.rows; i++)
            {
                /* 重み更新 */
                for(int j = 0; j < pLinear->weight.cols; j++)
                    matrix_add(&pLinear->weight, i, j, - lr * gradient.data[i] * input.data[j]);

                /* バイアス更新 */
                pLinear->bias.data[i] -= lr * gradient.data[i];
            }

            /* メモリを解放 */
            vector_destroy(&input);
            vector_destroy(&teach);
            vector_destroy(&output);
            vector_destroy(&gradient);
        }
    }
}


/* 結果を表示 */
void show_result(Linear* pLinear, Activation* pAct, Matrix* pX, Matrix* pT)
{
    for(int r = 0; r < pX->rows; r++)
    {
        /* Matrixのr行目をVectorとして取得 */
        Vector input = matrix_get_row(pX, r);
        Vector teach = matrix_get_row(pT, r);

        Vector output = vector_create(pLinear->weight.rows);
        Vector gradient = vector_create(pLinear->weight.rows);
        
        /* 推論 */
        predict(pLinear, pAct, &input, &output);

        /* 損失 */
        double loss = sse(&output, &teach);
        
        /* 損失の微分 */
        sse_derivative(&output, &teach, &gradient);

        /* 表示 */
        printf(
            "%.0f %.0f -> %.3f   loss = %.3f   gradient = %.3f\n",
            input.data[0], input.data[1], output.data[0], loss, gradient.data[0]
        );

        /* メモリを解放 */
        vector_destroy(&input);
        vector_destroy(&teach);
        vector_destroy(&output);
        vector_destroy(&gradient);
    }
}


/* 開始 */
int main(void)
{
    /* 学習データ（ANDゲート） */
    double data_x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double data_t[][1] = {{0}, {0}, {0}, {1}};
    
    Matrix x = matrix_create(4, 2);
    Matrix t = matrix_create(4, 1);

    for(int i = 0; i < x.rows; i++)
    {
        for(int j = 0; j < x.cols; j++)
            matrix_set(&x, i, j, data_x[i][j]);
    }

    for(int i = 0; i < t.rows; i++)
    {
        for(int j = 0; j < t.cols; j++)
            matrix_set(&t, i, j, data_t[i][j]);
    }
    
    /* 2入力1出力のLinear生成 */
    Linear linear1 = linear_create(2,1);
    Linear linear2 = linear_create(2,1);

    /* Activation生成 */
    Activation act1 = {step};
    Activation act2 = {sigmoid};
    
    /* 学習率 */
    double lr = 0.1;

    /* 学習前の結果を表示 */
    printf("\n=== Step (before) ===\n");

    show_result(&linear1, &act1, &x, &t);
    
    printf("\n=== Sigmoid (before) ===\n");

    show_result(&linear2, &act2, &x, &t);

    /* 学習 */
    train(&linear1, &act1, &x, &t, lr, 1000);
    train(&linear2, &act2, &x, &t, lr, 1000);

    /* 学習後の結果を表示 */
    printf("\n=== Step (after) ===\n");

    show_result(&linear1, &act1, &x, &t);
    
    printf("\n=== Sigmoid (after) ===\n");

    show_result(&linear2, &act2, &x, &t);
    
    /* メモリを解放 */
    linear_destroy(&linear1);
    linear_destroy(&linear2);
    matrix_destroy(&x);
    matrix_destroy(&t);

    return 0;
}
