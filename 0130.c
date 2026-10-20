/* C言語でニューラルネットワークを作る Lv.13 重みとバイアスを更新する */

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


/* Matrix ^ T × Vector */
void matrix_transpose_mul_vector(const Matrix* pMatrix, const Vector* x, Vector* y)
{
    for(int c = 0; c < pMatrix->cols; c++)
    {
        double sum = 0;
        
        for(int r = 0; r < pMatrix->rows; r++)
            sum += matrix_get(pMatrix, r, c) * x->data[r];
        
        y->data[c] = sum;
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

/* シグモイド関数 */
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


/* シグモイド関数の逆伝搬 */
double sigmoid_backward(double y, double dy)
{
    return y * (1 - y) * dy;

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
    
    /* 重みの勾配 */
    Matrix dWeight;
    
    /* バイアス */
    Vector bias;
    
    /* バイアスの勾配 */
    Vector dBias;
    
    /* 入力を保持 */
    Vector input;

} Linear;


/* Linear生成 */
Linear linear_create(int in, int out)
{
    Linear linear;
    
    linear.weight = matrix_create(out, in);
    linear.dWeight = matrix_create(out, in);
    linear.bias = vector_create(out);
    linear.dBias = vector_create(out);
    linear.input = vector_create(in);
    
    return linear;
}


/* Linearのメモリ解放 */
void linear_destroy(Linear* pLinear)
{
    matrix_destroy(&pLinear->weight);
    matrix_destroy(&pLinear->dWeight);
    vector_destroy(&pLinear->bias);
    vector_destroy(&pLinear->dBias);
    vector_destroy(&pLinear->input);
}


/* Linear Forward */
void linear_forward(Linear* pLinear, const Vector* pInput, Vector* pOutput)
{
    /* y = Wx */
    matrix_mul_vector(&pLinear->weight, pInput, pOutput);

    /* バイアスを加える */
    for(int i = 0; i < pOutput->size; i++)
        pOutput->data[i] += pLinear->bias.data[i];

    /* 入力を保存 */
    for (int i = 0; i < pInput->size; i++)
        pLinear->input.data[i] = pInput->data[i];
}


/* Linear Backward */
void linear_backward(Linear* pLinear, const Vector* pDoutput, Vector* pDinput)
{
    /* 重みの勾配 */
    for(int i = 0; i < pLinear->weight.rows; i++)
        for(int j = 0; j < pLinear->weight.cols; j++)
            matrix_set(&pLinear->dWeight, i, j, pDoutput->data[i] * pLinear->input.data[j]);

    /* バイアスの勾配 */
    for(int i = 0; i < pLinear->weight.rows; i++)
        pLinear->dBias.data[i] = pDoutput->data[i];

    /* 入力の勾配 */
    matrix_transpose_mul_vector(&pLinear->weight, pDoutput, pDinput);
}


/* Linear Update */
void linear_update(Linear* pLinear, double lr)
{
    /* 重み更新 */
    for(int i = 0; i <pLinear->weight.rows; i++)
        for(int j = 0; j < pLinear->weight.cols; j++)
            matrix_add(&pLinear->weight, i, j, - lr * matrix_get(&pLinear->dWeight, i, j));

    /* バイアス更新 */
    for(int i = 0; i <pLinear->weight.rows; i++)
        pLinear->bias.data[i] -= lr * pLinear->dBias.data[i];
}

/* ===== Activation Layer ===== */

/* Forward用の関数ポインタ型 */
typedef double (*ActivationForward)(double);


/* Backward用の関数ポインタ型 */
typedef double (*ActivationBackward)(double, double);


/* Activation */
typedef struct
{
    /* 出力を保持 */
    Vector output;

    /*  Forward用の関数ポインタ */
    ActivationForward forward;

    /* Backward用の関数ポインタ */
    ActivationBackward backward;

} Activation;


/* Activation Create */
Activation activation_create(int size, ActivationForward forward, ActivationBackward backward)
{
    Activation act;

    act.output = vector_create(size);
    act.forward = forward;
    act.backward = backward;

    return act;
}


/* Activationのメモリ解放 */
void activation_destroy(Activation* pAct)
{
    vector_destroy(&pAct->output);
}


/* Activation Forward */
void activation_forward(Activation* pAct, const Vector* pInput, Vector* pOutput)
{
    for(int i = 0; i < pInput->size; i++)
    {
        pAct->output.data[i] = pAct->forward(pInput->data[i]);
        pOutput->data[i] = pAct->output.data[i];
    }
}


/* Activation Backward */
void activation_backward(const Activation* pAct,  const Vector* pDoutput, Vector* pDinput)
{
    for(int i = 0; i < pDoutput->size; i++)
        pDinput->data[i] = pAct->backward(pAct->output.data[i], pDoutput->data[i]);
}

/* ===== パーセプトロン ===== */

/* 推論 */
void predict(Linear* pLinear, Activation* pAct, const Vector* pX, Vector* pY)
{
    /* 入力 → Linear → Activation → 出力 */
    linear_forward(pLinear, pX, pY);
    activation_forward(pAct, pY, pY);
}


/* 学習 */
void train(Linear* pLinear, Activation* pAct,
    const Matrix* pX, const Matrix* pT, double lr, int epoch)
{
    for(int e = 0; e < epoch; e++)
    {
        for(int r = 0; r < pX->rows; r++)
        {
            /* Matrixのr行目をVectorとして取得 */
            Vector x = matrix_get_row(pX, r);
            Vector t = matrix_get_row(pT, r);

            /* 出力 */
            Vector y = vector_create(pLinear->weight.rows);

            /* 勾配 */
            Vector dy = vector_create(y.size);
            Vector dz = vector_create(y.size);
            Vector dx = vector_create(pLinear->weight.cols);

            /* Forward */
            predict(pLinear, pAct, &x, &y);

            /* Loss */
            sse_derivative(&y, &t, &dy);

            /* Backward */
            activation_backward(pAct, &dy, &dz);
            linear_backward(pLinear, &dz, &dx);

            /* Update */
            linear_update(pLinear, lr);

            /* メモリ解放 */
            vector_destroy(&x);
            vector_destroy(&t);
            vector_destroy(&y);
            vector_destroy(&dy);
            vector_destroy(&dz);
            vector_destroy(&dx);
        }
    }
}


/* 結果を表示 */
void show_result(Linear* pLinear, Activation* pAct, Matrix* pX, Matrix* pT)
{
    for(int r = 0; r < pX->rows; r++)
    {
        /* Matrixのr行目をVectorとして取得 */
        Vector x = matrix_get_row(pX, r);
        Vector t = matrix_get_row(pT, r);

        /* 出力 */
        Vector y = vector_create(pLinear->weight.rows);

        /* 推論 */
        predict(pLinear, pAct, &x, &y);

        /* 表示 */
        printf("%.0f %.0f -> %.3f   target = %.0f\n", x.data[0], x.data[1], y.data[0], t.data[0]);

        /* メモリ解放 */
        vector_destroy(&x);
        vector_destroy(&t);
        vector_destroy(&y);
    }
}


/* 開始 */
int main(void)
{
    /* 入力データ（ANDゲート） */
    double data_x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Matrix x = matrix_create(4, 2);
    
    for(int i = 0; i < x.rows; i++)
    {
        for(int j = 0; j < x.cols; j++)
        matrix_set(&x, i, j, data_x[i][j]);
    }

    /* 教師データ（ANDゲート） */
    double data_t[][1] = {{0}, {0}, {0}, {1}};
    
    Matrix t = matrix_create(4, 1);

    for(int i = 0; i < t.rows; i++)
    {
        for(int j = 0; j < t.cols; j++)
            matrix_set(&t, i, j, data_t[i][j]);
    }
    
    /* Linear：2入力 → 1出力 */
    Linear linear = linear_create(2,1);

    /* Activation */
    Activation act = activation_create(1, sigmoid, sigmoid_backward);

    double lr = 1.0;
    int epoch = 1000;

    /* 学習 */
    train(&linear, &act, &x, &t, lr, epoch);

    /* 結果を表示 */ 
    printf("=== Result ===\n");
    show_result(&linear, &act, &x, &t);

    /* メモリ解放 */
    linear_destroy(&linear);
    activation_destroy(&act);
    matrix_destroy(&x);
    matrix_destroy(&t);

    return 0;
}
