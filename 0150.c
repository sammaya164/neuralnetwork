/* C言語でニューラルネットワークを作る Lv.15 Sequential：層を順番につなぐ */

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

/* ===== Linear Layer ===== */

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
    
    /* backwardで使う入力を保持 */
    Vector input;

} LinearLayer;


/* Xavier型の一様分布による初期化 */
double xavier_init(int nIn, int nOut)
{
    double limit = sqrt(6.0 / (nIn + nOut));

    double r = (double)rand() / RAND_MAX;

    return (2.0 * r - 1.0) * limit;
}


/* Linear Forward */
void linear_forward(LinearLayer* pLinear, const Vector* pX, Vector* pY)
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
void linear_backward(LinearLayer* pLinear, const Vector* pDy, Vector* pDx)
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
void linear_update(LinearLayer* pLinear, double lr)
{
    /* 重みとバイアスを更新 */
    for(int r = 0; r <pLinear->weight.rows; r++)
    {
        for(int c = 0; c < pLinear->weight.cols; c++)
            matrix_add(&pLinear->weight, r, c, - lr * matrix_get(&pLinear->dWeight, r, c));

        pLinear->bias.data[r] -= lr * pLinear->dBias.data[r];
    }
}


/* Linearのメモリ解放 */
void linear_destroy(LinearLayer* pLinear)
{
    matrix_destroy(&pLinear->weight);
    matrix_destroy(&pLinear->dWeight);
    vector_destroy(&pLinear->bias);
    vector_destroy(&pLinear->dBias);
    vector_destroy(&pLinear->input);
}


/* Linear生成 */
LinearLayer linear_create(int inputSize, int outputSize)
{
    LinearLayer linear;
    
    linear.weight = matrix_create(outputSize, inputSize);
    linear.dWeight = matrix_create(outputSize, inputSize);
    linear.bias = vector_create(outputSize);
    linear.dBias = vector_create(outputSize);
    linear.input = vector_create(inputSize);
    
    return linear;
}


/* 重みを乱数化 */
void linear_randomize(LinearLayer* pLinear)
{
    int rows = pLinear->weight.rows;
    int cols = pLinear->weight.cols;

    /* 重みを乱数化 */
    for(int r = 0; r < rows ; r++)
        for(int c = 0; c < cols; c++)
            matrix_set(&pLinear->weight, r, c, xavier_init(cols, rows));
}

/* ===== Activation Layer ===== */

/* Forward用の関数ポインタ型 */
typedef double (*ActivationForward)(double);


/* Backward用の関数ポインタ型 */
typedef double (*ActivationBackward)(double, double);


/* Activation */
typedef struct
{
    /* backwardで使うforward結果 */
    Vector output;

    /* Forward用の関数ポインタ */
    ActivationForward forward;

    /* Backward用の関数ポインタ */
    ActivationBackward backward;

} ActivationLayer;


/* Activation Forward */
void activation_forward(ActivationLayer* pActivation, const Vector* pX, Vector* pY)
{
    /* y = activation(x) */
    for(int i = 0; i < pX->size; i++)
        pY->data[i] 
            = pActivation->output.data[i] 
            = pActivation->forward(pX->data[i]);
}


/* Activation Backward */
void activation_backward(const ActivationLayer* pActivation,  const Vector* pDy, Vector* pDx)
{
    /* 勾配 */
    for(int i = 0; i < pDy->size; i++)
        pDx->data[i] = pActivation->backward(pActivation->output.data[i], pDy->data[i]);
}


/* Activationのメモリ解放 */
void activation_destroy(ActivationLayer* pActivation)
{
    vector_destroy(&pActivation->output);
}


/* Activation Create */
ActivationLayer activation_create(int size, ActivationForward forward, ActivationBackward backward)
{
    ActivationLayer activation;

    activation.output = vector_create(size);
    activation.forward = forward;
    activation.backward = backward;

    return activation;
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
    void (*forward)(Layer* pLayer, const Vector* pX, Vector* pY);

    /* Backward */
    void (*backward)(Layer* pLayer, const Vector* pDy, Vector* pDx);

    /* パラメータ更新（Linearで使用） */
    void (*update)(Layer* pLayer, double lr);
};


/* LinearのForward */
void linear_layer_forward(Layer* pLayer, const Vector* pX, Vector* pY)
{
    LinearLayer* pLinear = pLayer->context;

    linear_forward(pLinear, pX, pY);
}


/* LinearのBackward */
void linear_layer_backward(Layer* pLayer, const Vector* pDy, Vector* pDx)
{
    LinearLayer* pLinear = pLayer->context;

    linear_backward(pLinear, pDy, pDx);
}


/* LinearのUpdate */
void linear_layer_update(Layer* pLayer, double lr)
{
    LinearLayer* pLinear = pLayer->context;

    linear_update(pLinear, lr);
}


/* LinearをLayerとして登録 */
Layer layer_from_linear(LinearLayer* pLinear)
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
void activation_layer_forward(Layer* pLayer, const Vector* pX, Vector* pY)
{
    ActivationLayer* pActivation = pLayer->context;

    activation_forward(pActivation, pX, pY);
}


/* ActivationのBackward */
void activation_layer_backward(Layer* pLayer, const Vector* pDy, Vector* pDx)
{
    ActivationLayer* pActivation = pLayer->context;

    activation_backward(pActivation, pDy, pDx);
}


/* ActivationをLayerとして登録 */
Layer layer_from_activation(ActivationLayer* pActivation)
{
    Layer layer;

    layer.context = pActivation;
    layer.input_size = pActivation->output.size;
    layer.output_size = pActivation->output.size;

    layer.forward = activation_layer_forward;
    layer.backward = activation_layer_backward;
    layer.update = NULL;

    return layer;
}

/* ===== Sequential ===== */

typedef struct
{
    /* 層の配列 */
    Layer** layers;

    /* 層の数 */
    int layer_count;

    /* 各層の出力 */
    Vector* outputs;

    /* 各層の勾配 */
    Vector* gradients;

} Sequential;


/* Sequential生成 */
Sequential sequential_create(Layer** layers, int layerCount)
{
    Sequential seq = {0};

    seq.layer_count = layerCount;
    seq.layers = malloc(sizeof(Layer*) * layerCount);
    seq.outputs = malloc(sizeof(Vector) * layerCount);
    seq.gradients = malloc(sizeof(Vector) * (layerCount + 1));

    /* 層の入出力サイズのチェック */
    for(int i = 1; i < layerCount; i++)
    {
        if(layers[i - 1]->output_size != layers[i]->input_size)
        {
            fprintf(stderr, "Error: Layer sizes do not match.\n");
            seq.layer_count = 0;
            return seq;
        }
    }

    for(int i = 0; i < layerCount; i++)
    {
        seq.layers[i] = layers[i];
        seq.outputs[i] = vector_create(layers[i]->output_size);
    }

    /* ネットワーク入力の勾配 */
    seq.gradients[0] =
        vector_create(layers[0]->input_size);

    /* 各層の出力に対する勾配 */
    for (int i = 0; i < layerCount; i++)
        seq.gradients[i + 1] =
            vector_create(layers[i]->output_size);

    return seq;
}




/* Sequentialのメモリ解放 */
void sequential_destroy(Sequential* pSeq)
{
    for (int i = 0; i < pSeq->layer_count; i++)
        vector_destroy(&pSeq->outputs[i]);

    for (int i = 0; i <= pSeq->layer_count; i++)
        vector_destroy(&pSeq->gradients[i]);

    free(pSeq->outputs);
    free(pSeq->gradients);
    free(pSeq->layers);

    pSeq->outputs = NULL;
    pSeq->gradients = NULL;
    pSeq->layers = NULL;
}


/* Sequential Forward */
void sequential_forward(Sequential* pSeq, const Vector* pX, Vector* pY)
{
    const Vector* pInput = pX;

    /* 各層の順伝搬処理 */
    for (int i = 0; i < pSeq->layer_count; i++)
    {
        pSeq->layers[i]->forward(pSeq->layers[i], pInput, &pSeq->outputs[i]);
        pInput = &pSeq->outputs[i];
    }

    /* 最後の層の出力をコピー */
    vector_copy(pInput, pY);
}


/* Sequential Backward */
void sequential_backward(Sequential* pSeq, const Vector* pDy, Vector* pDx)
{
    /* 層の数 */
    int layerCount = pSeq->layer_count;

    /* 最終出力に対する勾配を保存 */
    vector_copy(pDy, &pSeq->gradients[layerCount]);

    /* 出力側から入力側へ逆向きに計算 */
    for (int i = layerCount - 1; i >= 0; i--)
        pSeq->layers[i]->backward(pSeq->layers[i], &pSeq->gradients[i + 1], &pSeq->gradients[i]);
    
    /* 入力に対する勾配を返す */
    vector_copy(&pSeq->gradients[0], pDx);
}


/* Sequential Update */
void sequential_update(Sequential* pSeq, double lr)
{
    /* 有効なupdateをもつすべての層でパラメータを更新 */
    for (int i = 0; i < pSeq->layer_count; i++)
        if (pSeq->layers[i]->update != NULL)
            pSeq->layers[i]->update(pSeq->layers[i], lr);
}

/* ===== Training / Result  ===== */

void train(Sequential* pSeq, const Matrix* pX, const Matrix* pT, double lr, int epoch)
{
    int outputSize = pSeq->layers[pSeq->layer_count - 1]->output_size;

    /* ベクトルを準備 */
    Vector y = vector_create(outputSize);
    Vector dy = vector_create(outputSize);
    Vector dx = vector_create(pX->cols);

    for (int e = 0; e < epoch; e++)
    {
        for (int r = 0; r < pX->rows; r++)
        {
            /* 教師データ */
            Vector input = matrix_get_row(pX, r);
            Vector target = matrix_get_row(pT, r);
            
            /* Forward */
            sequential_forward(pSeq, &input, &y);

            /* Lossの微分 */
            sse_derivative(&y, &target, &dy);

            /* Backward */
            sequential_backward(pSeq, &dy, &dx);

            /* Update */
            sequential_update(pSeq, lr);

            /* 教師データのメモリ解放 */
            vector_destroy(&input);
            vector_destroy(&target);
        }
    }

    /* メモリ解放 */
    vector_destroy(&y);
    vector_destroy(&dy);
    vector_destroy(&dx);
}


void vector_print(const Vector* pVector)
{
    printf("(");

    for(int i = 0; i < pVector->size; i++)
    {
        printf("%.3f", pVector->data[i]);

        if(i < pVector->size - 1)
            printf(", ");
    }

    printf(")");
}


void show_result(Sequential* pSeq, Matrix* pX, Matrix* pT)
{
    /* 最後の層の出力のサイズ */
    int outputSize = pSeq->layers[pSeq->layer_count - 1]->output_size;

    for(int r = 0; r < pX->rows; r++)
    {
        /* ベクトルを準備 */
        Vector input = matrix_get_row(pX, r);
        Vector target = matrix_get_row(pT, r);
        Vector y = vector_create(outputSize);

        /* Forward */
        sequential_forward(pSeq, &input, &y);

        /* 表示 */
        printf("Input : ");
        vector_print(&input);
    
        printf(" -> ");
        vector_print(&y);

        printf(" target = ");
        vector_print(&target);

        printf("\n");
        
        /* メモリ解放 */
        vector_destroy(&input);
        vector_destroy(&target);
        vector_destroy(&y);
    }
}

/* ===== main ===== */

int main(void)
{
    srand(1);

    /* 教師データ（XORゲート） */
    double dataX[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double dataT[][1] = {{0}, {1}, {1}, {0}};
    
    Matrix x = matrix_create(4, 2);
    Matrix t = matrix_create(4, 1);
    
    for(int r = 0; r < x.rows; r++)
        for(int c = 0; c < x.cols; c++)
           matrix_set(&x, r, c, dataX[r][c]);

    for(int r = 0; r < t.rows; r++)
        for(int c = 0; c < t.cols; c++)
            matrix_set(&t, r, c, dataT[r][c]);

    /* 隠れ層のサイズ */
    int hiddenSize = 3;

    /* Linear */
    LinearLayer linear1 = linear_create(2, hiddenSize);
    LinearLayer linear2 = linear_create(hiddenSize, 1);
    linear_randomize(&linear1);
    linear_randomize(&linear2);

    /* Activation */
    ActivationLayer activation1 =
        activation_create(hiddenSize, sigmoid, sigmoid_backward);
    ActivationLayer activation2 =
        activation_create(1, sigmoid, sigmoid_backward);

    /* Layerとして登録 */
    Layer linear_layer1 = layer_from_linear(&linear1);
    Layer act_layer1 = layer_from_activation(&activation1);
    Layer linear_layer2 = layer_from_linear(&linear2);
    Layer act_layer2 = layer_from_activation(&activation2);

    /* 層を順番に並べる */
    Layer* layers[] = {
        &linear_layer1,
        &act_layer1, 
        &linear_layer2, 
        &act_layer2
    };

    /* Sequentialを生成 */
    Sequential seq = sequential_create(layers, (int)(sizeof(layers) / sizeof(layers[0])));

    /* 学習 */
    double lr = 0.5;
    int epoch = 2000;
    train(&seq, &x, &t, lr, epoch);

    /* Result */
    printf("=== Result ===\n");
    show_result(&seq, &x, &t);

    /* メモリ解放 */
    sequential_destroy(&seq);
    matrix_destroy(&x);
    matrix_destroy(&t);

    /* 各Layerのdestroy関数を呼び、内部リソースを解放 */
    linear_destroy(&linear1);
    linear_destroy(&linear2);
    activation_destroy(&activation1);
    activation_destroy(&activation2);

    return 0;
}
