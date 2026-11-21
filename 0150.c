/* C言語でニューラルネットワークを作る Lv.15 Layerを共通化する */

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
    {
        fprintf(stderr, "Memory allocation failed in vector_create.\n");
        vector.size = 0;
        return vector;
    }

    return vector;
}


/* ベクトルのメモリ解放 */
void vector_destroy(Vector* pVector)
{
    free(pVector->data);
    pVector->data = NULL;
}


/* ベクトルのコピー */
void vector_copy(const Vector* v1, Vector* v2)
{
    /* サイズが一致しない場合 */
    if(v1->size != v2->size)
    {
        fprintf(stderr, "Error: Vector sizes do not match.\n");
        return;
    }

    /* v1 → v2 */
    for(int i = 0; i < v2->size; i++)
        v2->data[i] = v1->data[i];
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
    {
        fprintf(stderr, "Memory allocation failed in matrix_create.\n");
        matrix.rows = 0;
        matrix.cols = 0;
        return matrix;
    }

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
        y : 推論値
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
        y : 推論値
*/
void sse_derivative(const Vector* pY, const Vector* pT, Vector* pDy)
{
    for(int i = 0; i < pT->size; i++)
        pDy->data[i] = pY->data[i] - pT->data[i];
}

/* ===== 基底Layer ===== */

typedef struct Layer Layer;

struct Layer
{
    /* Forward */
    void (*forward)(Layer* layer, const Vector* input, Vector* output);

    /* Bacward */
    void (*backward)(Layer* layer, const Vector* dOutput, Vector* dInput);

    /* Update */
    void (*update)(Layer* layer, double lr);

    /* Destroy */
    void (*destroy)(Layer* layer);

     /* 入力のサイズ */
    int input_size;

    /* 出力のサイズ */
    int output_size;
};

/* ===== Linear Layer ===== */

/* Linear */
typedef struct
{
    /* 基底Layerを先頭メンバーに埋め込む */
    Layer base;             

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
Linear linear_create(int inputSize, int outputSize)
{
    Linear linear = {0};
    
    linear.base.input_size = inputSize;
    linear.base.output_size = outputSize;

    linear.weight = matrix_create(outputSize, inputSize);
    linear.dWeight = matrix_create(outputSize, inputSize);
    linear.bias = vector_create(outputSize);
    linear.dBias = vector_create(outputSize);
    linear.input = vector_create(inputSize);
    
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


/* Xavier型の一様分布による初期化 */
double xavier_init(int nIn, int nOut)
{
    double limit = sqrt(6.0 / (nIn + nOut));

    double r = (double)rand() / RAND_MAX;

    return (2.0 * r - 1.0) * limit;
}


/* 重みを乱数化 */
void linear_randomize(Linear* pLinear)
{
    /* 重みを乱数化 */
    for(int r = 0; r < pLinear->weight.rows; r++)
        for(int c = 0; c < pLinear->weight.cols; c++)
            matrix_set(&pLinear->weight, r, c, 
                xavier_init(pLinear->weight.cols, pLinear->weight.rows));
}


/* Linear Forward */
void linear_forward(Linear* pLinear, const Vector* pX, Vector* pY)
{
    /* y = Wx */
    matrix_mul_vector(&pLinear->weight, pX, pY);

    /* y = Wx + b */
    for(int i = 0; i < pY->size; i++)
        pY->data[i] += pLinear->bias.data[i];

    /* 入力を保存 */
    vector_copy(pX, &pLinear->input);
}


/* Linear Backward */
void linear_backward(Linear* pLinear, const Vector* pDy, Vector* pDx)
{
    /* 重みの勾配 */
    for(int r = 0; r < pLinear->weight.rows; r++)
        for(int c = 0; c < pLinear->weight.cols; c++)
            matrix_set(&pLinear->dWeight, r, c, pDy->data[r] * pLinear->input.data[c]);

    /* バイアスの勾配 */
    vector_copy(pDy, &pLinear->dBias);

    /* 入力の勾配 */
    matrix_transpose_mul_vector(&pLinear->weight, pDy, pDx);
}


/* Linear Update */
void linear_update(Linear* pLinear, double lr)
{
    /* 重み更新 */
    for(int r = 0; r <pLinear->weight.rows; r++)
        for(int c = 0; c < pLinear->weight.cols; c++)
            matrix_add(&pLinear->weight, r, c, 
                - lr * matrix_get(&pLinear->dWeight, r, c));

    /* バイアス更新 */
    for(int r = 0; r <pLinear->weight.rows; r++)
        pLinear->bias.data[r] -= lr * pLinear->dBias.data[r];
}

/* ===== Activation Layer ===== */

/* Forward用の関数ポインタ型 */
typedef double (*ActivationForward)(double);


/* Backward用の関数ポインタ型 */
typedef double (*ActivationBackward)(double, double);


/* Activation */
typedef struct
{
    /* 基底Layerを先頭メンバーに埋め込む */
    Layer base;

    /* 出力を保持 */
    Vector output;

    /* Forward用の関数ポインタ */
    ActivationForward activation_forward;

    /* Backward用の関数ポインタ */
    ActivationBackward activation_backward;

} Activation;


/* Activation Create */
Activation activation_create(int size, ActivationForward forward, ActivationBackward backward)
{
    Activation activation = {0};
    activation.base.input_size = size;
    activation.base.output_size = size;

    activation.output = vector_create(size);
    activation.activation_forward = forward;
    activation.activation_backward = backward;

    return activation;
}


/* Activationのメモリ解放 */
void activation_destroy(Activation* pActivation)
{
    vector_destroy(&pActivation->output);
}


/* Activation Forward */
void activation_forward(Activation* pActivation, const Vector* pX, Vector* pY)
{
    /* y = activation(x) */
    for(int i = 0; i < pX->size; i++)
        pY->data[i]
            = pActivation->output.data[i] 
            = pActivation->activation_forward(pX->data[i]);
}


/* Activation Backward */
void activation_backward(const Activation* pActivation,  const Vector* pDy, Vector* pDx)
{
    /* 勾配 */
    for(int i = 0; i < pDy->size; i++)
        pDx->data[i] = pActivation->activation_backward(pActivation->output.data[i], pDy->data[i]);
}

/* ===== ニューラルネットワーク ===== */

/* 推論 */
void predict(Linear* pLinear1, Activation* pAct1, Linear* pLinear2, Activation* pAct2,
    const Vector* pX, Vector* pY)
{
    Vector z1 = vector_create(pLinear1->weight.rows);
    Vector a1 = vector_create(pLinear1->weight.rows);
    Vector z2 = vector_create(pLinear2->weight.rows);

    /* 入力 → Linear 1 → Activation 1 */
    linear_forward(pLinear1, pX, &z1);
    activation_forward(pAct1, &z1, &a1);

    /* Linear 2 → Activation 2 → 出力*/
    linear_forward(pLinear2, &a1, &z2);
    activation_forward(pAct2, &z2, pY);

    vector_destroy(&z1);
    vector_destroy(&a1);
    vector_destroy(&z2);
}


/* 学習 */
void train(Linear* pLinear1, Activation* pAct1, Linear* pLinear2, Activation* pAct2, 
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
            Vector y = vector_create(pLinear2->weight.rows);

            /* 勾配 */
            Vector dy = vector_create(y.size);
            Vector dz2 = vector_create(y.size);
            Vector da1 = vector_create(pLinear1->weight.rows);
            Vector dz1 = vector_create(da1.size);
            Vector dx = vector_create(x.size);

            /* Forward */
            predict(pLinear1, pAct1, pLinear2, pAct2, &x, &y);

            /* Loss */
            sse_derivative(&y, &t, &dy);

            /* Backward */
            activation_backward(pAct2, &dy, &dz2);
            linear_backward(pLinear2, &dz2, &da1);
            activation_backward(pAct1, &da1, &dz1);
            linear_backward(pLinear1, &dz1, &dx);

            /* Update */
            linear_update(pLinear1, lr);
            linear_update(pLinear2, lr);

            /* メモリ解放 */
            vector_destroy(&x);
            vector_destroy(&t);
            vector_destroy(&y);
            vector_destroy(&dy);
            vector_destroy(&dz2);
            vector_destroy(&da1);
            vector_destroy(&dz1);
            vector_destroy(&dx);
        }
    }
}


/* 結果を表示 */
void show_result(Linear* pLinear1, Activation* pAct1, Linear* pLinear2, Activation* pAct2, 
    Matrix* pX, Matrix* pT)
{
    for(int r = 0; r < pX->rows; r++)
    {
        /* Matrixのr行目をVectorとして取得 */
        Vector x = matrix_get_row(pX, r);
        Vector t = matrix_get_row(pT, r);

        /* 出力 */
        Vector y = vector_create(pLinear2->weight.rows);

        /* 推論 */
        predict(pLinear1, pAct1, pLinear2, pAct2, &x, &y);

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
    srand(1);

    /* 入力データ（XORゲート） */
    double data_x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Matrix x = matrix_create(4, 2);
    
    for(int i = 0; i < x.rows; i++)
    {
        for(int j = 0; j < x.cols; j++)
        matrix_set(&x, i, j, data_x[i][j]);
    }

    /* 教師データ（XORゲート） */
    double data_t[][1] = {{0}, {1}, {1}, {0}};
    
    Matrix t = matrix_create(4, 1);

    for(int i = 0; i < t.rows; i++)
    {
        for(int j = 0; j < t.cols; j++)
            matrix_set(&t, i, j, data_t[i][j]);
    }
    
    /* 隠れ層のサイズ */
    int hide = 3;

    /* Linear 1：2入力 → hide出力 */
    Linear linear1 = linear_create(2,hide);
    
    /* Linear 2：hide入力 → 1出力 */
    Linear linear2 = linear_create(hide,1);
 
    /* Activation */
    Activation act1 = activation_create(hide, sigmoid, sigmoid_backward);
    Activation act2 = activation_create(1, sigmoid, sigmoid_backward);

    double lr = 0.5;
    int epoch = 2000;
    
    /* 乱数化 */
    linear_randomize(&linear1);
    linear_randomize(&linear2);
    
    /* 学習 */
    train(&linear1, &act1, &linear2, &act2, &x, &t, lr, epoch);

    /* 結果を表示 */ 
    printf("=== Result ===\n");
    show_result(&linear1, &act1, &linear2, &act2, &x, &t);

    /* メモリ解放 */
    linear_destroy(&linear1);
    linear_destroy(&linear2);
    activation_destroy(&act1);
    activation_destroy(&act2);
    matrix_destroy(&x);
    matrix_destroy(&t);

    return 0;
}
