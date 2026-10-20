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


/* Xavier型の一様分布による初期化 */
double xavier_init(int n_in, int n_out)
{
    double limit = sqrt(6.0 / (n_in + n_out));

    double r = (double)rand() / RAND_MAX;

    return (2.0 * r - 1.0) * limit;
}


/* 重みを乱数化 */
void linear_randomize(Linear* pLinear)
{
    /* 重みを乱数化 */
    for(int i = 0; i < pLinear->weight.rows; i++)
        for(int j = 0; j < pLinear->weight.cols; j++)
            matrix_set(&pLinear->weight, i, j, xavier_init(pLinear->weight.cols, pLinear->weight.rows));
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

/* ===== Layer ===== */

typedef struct Layer Layer;

struct Layer
{
    /* 実際の層への参照 */
    void* context;

    /* 入出力のサイズ */
    int input_size;
    int output_size;

    /* Forward */
    void (*forward)(Layer* pLayer, const Vector* pInput, Vector* pOutput);

    /* Backward */
    void (*backward)(Layer* pLayer, const Vector* pDoutput, Vector* pDinput);

    /* パラメータ更新（Activationでは使用しない） */
    void (*update)(Layer* pLayer, double lr);
};


/* LinearのForward */
void linear_layer_forward(Layer* pLayer, const Vector* pInput, Vector* pOutput)
{
    Linear* pLinear = pLayer->context;

    linear_forward(pLinear, pInput, pOutput);
}


/* LinearのBackward */
void linear_layer_backward(Layer* pLayer, const Vector* pDoutput, Vector* pDinput)
{
    Linear* pLinear = pLayer->context;

    linear_backward(pLinear, pDoutput, pDinput);
}

/* LinearのUpdate */
void linear_layer_update(Layer* pLayer, double lr)
{
    Linear* pLinear = pLayer->context;

    linear_update(pLinear, lr);
}


/* LinearをLayerとして登録 */
Layer layer_from_linear(Linear* pLinear)
{
    Layer layer;

    layer.context = pLinear;
    layer.input_size = pLinear->weight.cols;
    layer.output_size = pLinear->weight.rows;

    layer.forward = linear_layer_forward;
    layer.backward = linear_layer_backward;
    layer.update = linear_layer_update;

    return layer;
}


/* ActivationのForward */
void activation_layer_forward(Layer* pLayer, const Vector* pInput, Vector* pOutput)
{
    Activation* pAct = pLayer->context;

    activation_forward(pAct, pInput, pOutput);
}


/* ActivationのBackward */
void activation_layer_backward(Layer* pLayer, const Vector* pDoutput, Vector* pDinput)
{
    Activation* pAct = pLayer->context;

    activation_backward(pAct, pDoutput, pDinput);
}


/* ActivationをLayerとして登録 */
Layer layer_from_activation(Activation* pAct)
{
    Layer layer;

    layer.context = pAct;
    layer.input_size = pAct->output.size;
    layer.output_size = pAct->output.size;

    layer.forward = activation_layer_forward;
    layer.backward = activation_layer_backward;
    layer.update = NULL;

    return layer;
}

/* ===== Network ===== */

/* 推論 */
void predict(Layer* pLinear1, Layer* pAct1, Layer* pLinear2, Layer* pAct2,
    const Vector* pX, Vector* pY)
{
    Vector z1 = vector_create(pLinear1->output_size);
    Vector a1 = vector_create(pAct1->output_size);
    Vector z2 = vector_create(pLinear2->output_size);

    /* Input → Layer 1 → Activation 1 */
    pLinear1->forward(pLinear1, pX, &z1);
    pAct1->forward(pAct1, &z1, &a1);

    /* → Layer 2 → Activation 2 → Output */
    pLinear2->forward(pLinear2, &a1, &z2);
    pAct2->forward(pAct2, &z2, pY);

    vector_destroy(&z1);
    vector_destroy(&a1);
    vector_destroy(&z2);
}


void train(Layer* pLinear1, Layer* pAct1, Layer* pLinear2, Layer* pAct2, 
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
            Vector y = vector_create(pAct2->output_size);

            /* 勾配 */
            Vector dy = vector_create(y.size);
            Vector dz2 = vector_create(y.size);
            Vector da1 = vector_create(pAct1->output_size);
            Vector dz1 = vector_create(da1.size);
            Vector dx = vector_create(x.size);

            /* Forward */
            predict(pLinear1, pAct1, pLinear2, pAct2, &x, &y);

            /* Loss */
            sse_derivative(&y, &t, &dy);

            /* Backward */
            pAct2->backward(pAct2, &dy, &dz2);
            pLinear2->backward(pLinear2, &dz2, &da1);
            pAct1->backward(pAct1, &da1, &dz1);
            pLinear1->backward(pLinear1, &dz1, &dx);

            /* Update */
            if (pLinear1->update != NULL)
                pLinear1->update(pLinear1, lr);

            if (pLinear2->update != NULL)
                pLinear2->update(pLinear2, lr);

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
void show_result(Layer* pLinear1, Layer* pAct1, Layer* pLinear2, Layer* pAct2, 
    Matrix* pX, Matrix* pT)
{
    for(int r = 0; r < pX->rows; r++)
    {
        /* Matrixのr行目をVectorとして取得 */
        Vector x = matrix_get_row(pX, r);
        Vector t = matrix_get_row(pT, r);

        /* 出力 */
        Vector y = vector_create(pAct2->output_size);

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

    /* 入力データ（XRDゲート） */
    double data_x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Matrix x = matrix_create(4, 2);
    
    for(int i = 0; i < x.rows; i++)
    {
        for(int j = 0; j < x.cols; j++)
        matrix_set(&x, i, j, data_x[i][j]);
    }

    /* 教師データ（XRDゲート） */
    double data_t[][1] = {{0}, {1}, {1}, {0}};
    
    Matrix t = matrix_create(4, 1);

    for(int i = 0; i < t.rows; i++)
    {
        for(int j = 0; j < t.cols; j++)
            matrix_set(&t, i, j, data_t[i][j]);
    }
    
    /* 隠れ層のサイズ */
    int hide = 3;

    /* Linear */
    Linear linear1 = linear_create(2, hide);
    Linear linear2 = linear_create(hide, 1);

    /* 重みの初期化 */
    linear_randomize(&linear1);
    linear_randomize(&linear2);

    /* Activation */
    Activation act1 = activation_create(hide, sigmoid, sigmoid_backward);
    Activation act2 =activation_create(1, sigmoid, sigmoid_backward);

    /* Layerとして登録 */
    Layer linear_layer1 = layer_from_linear(&linear1);
    Layer linear_layer2 = layer_from_linear(&linear2);

    Layer act_layer1 = layer_from_activation(&act1);
    Layer act_layer2 = layer_from_activation(&act2);

    double lr = 0.5;
    int epoch = 2000;

    /* 学習 */
    train(&linear_layer1, &act_layer1, &linear_layer2, &act_layer2, &x, &t, lr, epoch);

    /* 結果を表示 */ 
    printf("=== Result ===\n");
    show_result(&linear_layer1, &act_layer1, &linear_layer2, &act_layer2, &x, &t);

    /* メモリ解放 */
    linear_destroy(&linear1);
    linear_destroy(&linear2);
    activation_destroy(&act1);
    activation_destroy(&act2);
    matrix_destroy(&x);
    matrix_destroy(&t);

    return 0;
}
