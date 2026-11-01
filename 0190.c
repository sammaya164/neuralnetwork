/* C言語でニューラルネットワークを作る Lv.19 AddノードのForward・Backwardを完成させる */

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


/* ベクトルの和 */
void vector_add(Vector* pVector1, Vector* pVector2, Vector* pVector)
{
    for(int i = 0; i < pVector->size; i++)
        pVector->data[i] = pVector1->data[i] + pVector2->data[i];
}


/* ベクトルの複製 */
void vector_copy(Vector* pVector1, Vector* pVector)
{
    for(int i = 0; i < pVector->size; i++)
        pVector->data[i] = pVector1->data[i];
}


/* ベクトルを0にリセット */
void vector_reset(Vector* pVector)
{
    for(int i = 0; i < pVector->size; i++)
        pVector->data[i] = 0.0;
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

/* ===== Computation Graph ===== */

/* ノードの種類 */
typedef enum
{
    GRAPH_NODE_INPUT,
    GRAPH_NODE_LAYER,
    GRAPH_NODE_ADD
} GraphNodeType;


/* ノードの設定情報 */
typedef struct
{
    GraphNodeType type;
    Layer* layer;

    int input_nodes[2];
    int input_count;

    int input_index;
    int output_size;

} GraphNodeSpec;


/* ノード */
typedef struct
{
    /* ノードの種類 */
    GraphNodeType type;

    /* 計算を行うLayer */
    Layer* layer;

    /* 入力元ノードの番号 */
    int input_nodes[2];

    /* 入力元の数 */
    int input_count;

    /* GRAPH_NODE_INPUTで使用 */
    int input_index;

    /* このノードの出力のサイズ */
    int output_size;

    /* このノードの出力 */
    Vector output;

    /* このノードの出力に対する勾配 */
    Vector gradient;

} GraphNode;



/* 計算グラフ */
typedef struct
{
    GraphNode* nodes;
    int node_count;

} ComputationGraph;


/* ComputationGraphの生成 */
ComputationGraph graph_create(const GraphNodeSpec* specs, int node_count)
{
    ComputationGraph graph;

    graph.nodes = calloc(node_count, sizeof(GraphNode));

    if (graph.nodes == NULL)
        fprintf(stderr, "Memory allocation failed in graph_create.\n");
    
    graph.node_count = node_count;

    for (int i = 0; i < node_count; i++)
    {
        GraphNode* pNode = &graph.nodes[i];

        pNode->type = specs[i].type;
        pNode->layer = specs[i].layer;
        pNode->input_count = specs[i].input_count;
        pNode->input_index = specs[i].input_index;
        pNode->output_size = specs[i].output_size;
        
        for(int j = 0; j < 2; j++)
            pNode->input_nodes[j] = specs[i].input_nodes[j];

        pNode->output = vector_create(pNode->output_size);
        pNode->gradient = vector_create(pNode->output_size);
    }

    return graph;
}


/* ComputationGraphのメモリ解放 */
void graph_destroy(ComputationGraph* pGraph)
{
    for (int i = 0; i < pGraph->node_count; i++)
    {
        vector_destroy(&pGraph->nodes[i].output);
        vector_destroy(&pGraph->nodes[i].gradient);
    }

    free(pGraph->nodes);
    pGraph->nodes = NULL;
}


/* Forward */
void graph_forward(ComputationGraph* pGraph, Vector* inputs, Vector* pY)
{
    /* 各ノードの処理 */
    for (int i = 0; i < pGraph->node_count; i++)
    {
        GraphNode* pNode = &pGraph->nodes[i];

        switch(pNode->type)
        {
            case GRAPH_NODE_INPUT:
            {
                vector_copy(&inputs[pNode->input_index], &pNode->output);   
                break;
            }

            case GRAPH_NODE_ADD:
            {
                GraphNode* pNode1 = &pGraph->nodes[pNode->input_nodes[0]];
                GraphNode* pNode2 = &pGraph->nodes[pNode->input_nodes[1]];
                vector_add(&pNode1->output, &pNode2->output, &pNode->output);
                break;
            }

            case GRAPH_NODE_LAYER:
            {
                GraphNode* pParent = &pGraph->nodes[pNode->input_nodes[0]];
                pNode->layer->forward(pNode->layer, &pParent->output, &pNode->output);
                break;
            }
        }
    }

    /* 最終ノードの出力をコピー */
    GraphNode* pLast = &pGraph->nodes[pGraph->node_count - 1];
    vector_copy(&pLast->output, pY);
}


/* Backward */
void graph_backward(ComputationGraph* pGraph, Vector* pDy)
{
    int n = pGraph->node_count;

    /* 前回の勾配をリセット */
    for (int i = 0; i < pGraph->node_count; i++)
        vector_reset(&pGraph->nodes[i].gradient);
    
    /* 最終ノードに損失の勾配を設定 */
    vector_copy(pDy, &pGraph->nodes[n - 1].gradient);

    /* 出力側から入力側へ逆伝搬 */
    for (int i = n - 1; i >= 0; i--)
    {
        GraphNode* pNode = &pGraph->nodes[i];

        switch(pNode->type)
        {
            case GRAPH_NODE_ADD:
            {
                /* Addの勾配を両方の入力元へ加算 */
                for(int j = 0; j < 2; j++)
                {
                    GraphNode* pParent = &pGraph->nodes[pNode->input_nodes[j]];
                    vector_add(&pNode->gradient, &pParent->gradient, &pParent->gradient);
                }
                break;
            }

            case GRAPH_NODE_LAYER:
            {
                /* LayerのBackward */
                int input_size = pGraph->nodes[pNode->input_nodes[0]].output_size;
                Vector dInput = vector_create(input_size);
                pNode->layer->backward(pNode->layer, &pNode->gradient, &dInput);
        
                GraphNode* pParent = &pGraph->nodes[pNode->input_nodes[0]];
                vector_add(&dInput, &pParent->gradient, &pParent->gradient);
                vector_destroy(&dInput);
                break;
            }

            default:
                break;
        }
    }
}


/* Update */
void graph_update(ComputationGraph* pGraph, double lr)
{
    for (int i = 0; i < pGraph->node_count; i++)
        if (pGraph->nodes[i].layer != NULL && pGraph->nodes[i].layer->update != NULL)
            pGraph->nodes[i].layer->update(pGraph->nodes[i].layer, lr);
}

/* ===== Neural Network ===== */

void train(ComputationGraph* pNet, const Matrix* pX, const Matrix* pT, double lr, int epoch)
{
    for (int e = 0; e < epoch; e++)
    {
        for (int r = 0; r < pX->rows; r++)
        {
            Vector x = matrix_get_row(pX, r);
            Vector t = matrix_get_row(pT, r);

            int output_size = pNet->nodes[pNet->node_count - 1].output_size;

            Vector y = vector_create(output_size);
            Vector dy = vector_create(output_size);
            Vector dx = vector_create(x.size);

            /* Forward */
            graph_forward(pNet, &x, &y);

            /* Lossの微分 */
            sse_derivative(&y, &t, &dy);

            /* Backward */
            graph_backward(pNet, &dy);

            /* Update */
            graph_update(pNet, lr);

            vector_destroy(&x);
            vector_destroy(&t);
            vector_destroy(&y);
            vector_destroy(&dy);
            vector_destroy(&dx);
        }
    }
}


/* 結果を表示 */
void show_result(ComputationGraph* pNet, Matrix* pX, Matrix* pT)
{
    for(int r = 0; r < pX->rows; r++)
    {
        /* Matrixのr行目をVectorとして取得 */
        Vector x = matrix_get_row(pX, r);
        Vector t = matrix_get_row(pT, r);

        /* 出力 */
        Vector y = vector_create(pNet->nodes[pNet->node_count - 1].output_size);

        /* 推論 */
        graph_forward(pNet, &x, &y);

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

    GraphNodeSpec specs[] =
    {
        /* ノード0 : 外部入力a */
        {
            GRAPH_NODE_INPUT, 
            NULL,
            {-1, -1},
            0,
            0,
            2
        },

        /* ノード1 : 外部入力b */
        {
            GRAPH_NODE_INPUT, 
            NULL,
            {-1, -1},
            0,
            1,
            2
        },
        
        /* ノード2 : a + b */
        {
            GRAPH_NODE_ADD,
            NULL,
            {0, 1},
            2,
            -1,
            2
        },
        
        /* ノード3 : a + b */
        {
            GRAPH_NODE_ADD,
            NULL,
            {0, 1},
            2,
            -1,
            2
        },
        
        /* ノード4 : ノード2 + ノード3 */
        {
            GRAPH_NODE_ADD,
            NULL,
            {2, 3},
            2,
            -1,
            2
        }
    };

    /* 外部入力 */
    Vector inputs[2];

    inputs[0] = vector_create(2);
    inputs[1] = vector_create(2);

    inputs[0].data[0] = 1.0;
    inputs[0].data[1] = 2.0;

    inputs[1].data[0] = 3.0;
    inputs[1].data[1] = 4.0;

    /* 出力 */
    Vector y = vector_create(2);
    
    /* ComputationGraphを生成 */
    ComputationGraph  graph = graph_create(specs, 5);
    
    /* Forward */
    graph_forward(&graph, inputs, &y);

    /* 結果表示 */
    printf("y = (%f, %f)\n",
        y.data[0],
        y.data[1]
    );

    /* 出力の勾配 */
    Vector dy = vector_create(2);

    dy.data[0] = 1.0;
    dy.data[1] = 1.0;

    /* Backward */
    graph_backward(&graph, &dy);

    /* 入力aの勾配 */
    printf("da = (%f, %f)\n", graph.nodes[0].gradient.data[0], graph.nodes[0].gradient.data[1]);

    /* 入力bの勾配 */
    printf("db = (%f, %f)\n", graph.nodes[1].gradient.data[0], graph.nodes[1].gradient.data[1]);

    /* メモリ解放 */
    graph_destroy(&graph);
    vector_destroy(&inputs[0]);
    vector_destroy(&inputs[1]);
    vector_destroy(&y);
    vector_destroy(&dy);

    return 0;
}
