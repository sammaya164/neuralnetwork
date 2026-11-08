/* C言語でニューラルネットワークを作る Lv.21 LayerからTransformへ */

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
void vector_add(const Vector* vector1, Vector* vector2)
{
    for(int i = 0; i < vector2->size; i++)
        vector2->data[i] = vector1->data[i] + vector2->data[i];
}


/* ベクトルの複製 */
void vector_copy(const Vector* vector1, Vector* vector2)
{
    for(int i = 0; i < vector2->size; i++)
        vector2->data[i] = vector1->data[i];
}


/* ベクトルを0にリセット */
void vector_zero(Vector* vector)
{
    for(int i = 0; i < vector->size; i++)
        vector->data[i] = 0.0;
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

/* ===== Transform ===== */

typedef struct Transform Transform;

struct Transform
{
    /* 実際の変換処理への参照 */
    void* context;

    /* 入力の数 */
    int input_count;

    /* 出力のサイズ */
    int output_size;

    /* Forward */
    void (*forward)(
        Transform* transform,
        const Vector* inputs[],
        Vector* output
    );

    /* Backward */
    void (*backward)(
        Transform* transform,
        const Vector* inputs[],
        const Vector* doutput,
        Vector* dinputs[]
    );

    /* パラメータ更新 */
    void (*update)(Transform* transform, double lr);
};

/* ===== Linear Transform ===== */

/* Forward */
void linear_transform_forward(
    Transform* transform,
    const Vector* inputs[],
    Vector* output
)
{
    Linear* linear = transform->context;

    linear_forward(linear, inputs[0], output);
}

/* Backward */
void linear_transform_backward(
    Transform* transform,
    const Vector* inputs[],
    const Vector* doutput,
    Vector* dinputs[]
)
{
    Linear* linear = transform->context;

    linear_backward(linear, doutput, dinputs[0]);
}

/* Update */
void linear_transform_update(Transform* transform, double lr)
{
    Linear* linear = transform->context;

    linear_update(linear, lr);
}

/* LinearをTransformとして登録 */
Transform transform_from_linear(Linear* linear)
{
    Transform transform;

    transform.context = linear;
    transform.input_count = 1;
    transform.output_size = linear->weight.rows;
    transform.forward = linear_transform_forward;
    transform.backward = linear_transform_backward;
    transform.update = linear_transform_update;

    return transform;
}

/* ===== Activation Transform ===== */

/* Forward */
void activation_transform_forward(
    Transform* transform,
    const Vector* inputs[],
    Vector* output
)
{
    Activation* activation = transform->context;

    activation_forward(activation, inputs[0], output);
}

/* Backward */
void activation_transform_backward(
    Transform* transform,
    const Vector* inputs[],
    const Vector* doutput,
    Vector* dinputs[]
)
{
    Activation* pAct = transform->context;

    activation_backward(pAct, doutput, dinputs[0]);
}

/* ActivationをTransformとして登録 */
Transform transform_from_activation(Activation* activation)
{
    Transform transform;

    transform.context = activation;
    transform.input_count = 1;
    transform.output_size = activation->output.size;
    transform.forward = activation_transform_forward;
    transform.backward = activation_transform_backward;
    transform.update = NULL;

    return transform;
}

/* ===== Add Transform ===== */

/* Forward */
void add_transform_forward(
    Transform* transform,
    const Vector* inputs[],
    Vector* output
)
{
    vector_zero(output);

    int input_count = transform->input_count;

    for(int i = 0; i < input_count; i++)
        vector_add(inputs[i], output);
}

/* backward */
void add_transform_backward(
    Transform* transform,
    const Vector* inputs[],
    const Vector* doutput,
    Vector* dinputs[]
)
{
    int input_count = transform->input_count;

    for(int i = 0; i < input_count; i++)
        vector_copy(doutput, dinputs[i]);
}

/* AddをTransformとして登録 */
Transform transform_create_add(int input_count, int output_size)
{
    Transform transform;

    transform.context = NULL;
    transform.input_count = input_count;
    transform.output_size = output_size;
    transform.forward = add_transform_forward;
    transform.backward = add_transform_backward;
    transform.update = NULL;

    return transform;
}

/* ===== Computation Graph ===== */

/* ノードの種類 */
typedef enum
{
    GRAPH_NODE_INPUT,
    GRAPH_NODE_TRANSFORM
} GraphNodeType;

/* ノードの設定情報 */
typedef struct
{
    /* ノードの種類 */
    GraphNodeType type;

    /* Transformへのポインタ */
    Transform* transform;

    /* 入力元ノードの配列 */
    const int* input_nodes;

    /* 外部入力の番号 */
    int input_index;

    /* 出力のサイズ */
    int output_size;

} GraphNodeSpec;


/* ノード */
typedef struct
{
    /* ノードの種類 */
    GraphNodeType type;

    /* Transformへのポインタ */
    Transform* transform;

    /* 入力元ノードの配列 */
    int* input_nodes;

    /* 外部入力の番号 */
    int input_index;

    /* 出力のサイズ */
    int output_size;

    /* 出力 */
    Vector output;

    /* 出力の勾配 */
    Vector gradient;

} GraphNode;

/* 計算グラフ */
typedef struct
{
    GraphNode* nodes;
    int node_count;
    int input_port_count;

} ComputationGraph;

/* ComputationGraphの生成 */
ComputationGraph graph_create(const GraphNodeSpec* specs, int node_count)
{
    ComputationGraph graph;

    graph.nodes = calloc(node_count, sizeof(GraphNode));

    if (graph.nodes == NULL)
        fprintf(stderr, "Memory allocation failed in graph_create.\n");
    
    graph.node_count = node_count;

    int max_input_index = -1;

    for (int i = 0; i < node_count; i++)
    {
        GraphNode* pNode = &graph.nodes[i];
        
        pNode->type = specs[i].type;
        pNode->transform = specs[i].transform;
        pNode->input_index = specs[i].input_index;
        pNode->output_size = specs[i].output_size;

        switch(pNode->type)
        {
            case GRAPH_NODE_TRANSFORM:
            {
                int input_count = pNode->transform->input_count;
                //GRAPH_NODE_INPUTではtransform == NULLなので場合分けが必要

                if(input_count > 0)
                {
                    pNode->input_nodes = calloc(input_count, sizeof(int));

                    if(pNode->input_nodes == NULL)
                        fprintf(stderr, "Memory allocation failed in graph_create.\n");

                    for(int j = 0; j < input_count; j++)
                        pNode->input_nodes[j] = specs[i].input_nodes[j];
                }
                else
                {
                    pNode->input_nodes = NULL;
                }

                break;
            }
            case GRAPH_NODE_INPUT:
            {
                pNode->input_nodes = NULL;

                if(pNode->input_index > max_input_index)
                    max_input_index = pNode->input_index;

                break;
            }
        }
        
        pNode->output = vector_create(pNode->output_size);
        pNode->gradient = vector_create(pNode->output_size);
    }

    graph.input_port_count = max_input_index + 1;

    return graph;
}

/* ComputationGraphのメモリ解放 */
void graph_destroy(ComputationGraph* graph)
{
    for (int i = 0; i < graph->node_count; i++)
    {
        GraphNode* pNode = &graph->nodes[i];

        if(pNode->transform->input_count > 0)
        {
            free(pNode->input_nodes);
            pNode->input_nodes = NULL;
        }

        vector_destroy(&pNode->output);
        vector_destroy(&pNode->gradient);
    }

    free(graph->nodes);
    graph->nodes = NULL;
}

/* Forward 
    Transformの入力元ノードからVectorを集め、TransformのForwardに渡す */
void graph_forward(
    ComputationGraph* graph,
    Vector* inputs,
    Vector* output
)
{
    for(int i = 0; i < graph->node_count; i++)
    {
        GraphNode* pNode = &graph->nodes[i];

        if(pNode->type == GRAPH_NODE_INPUT)
            vector_copy(&inputs[pNode->input_index], &pNode->output);
        else
        {
            /* Transformへの入力を集める */
            const Vector* transform_inputs[pNode->transform->input_count]; //C99以降で使える可変長配列

            for(int j = 0; j < pNode->transform->input_count; j++)
            {
                GraphNode* pParent = &graph->nodes[pNode->input_nodes[j]];

                transform_inputs[j] = &pParent->output;
            }

            /* TransformのForward */
            pNode->transform->forward(pNode->transform, transform_inputs, &pNode->output);
        }
    }

    /* 最終ノードの出力 */
    GraphNode* pLast = &graph->nodes[graph->node_count - 1];

    vector_copy(&pLast->output, output);
}

/* Backward */
void graph_backward(ComputationGraph* graph, Vector* doutput)
{
    /* 勾配をリセット */
    for(int i = 0; i < graph->node_count; i++)
        vector_zero(&graph->nodes[i].gradient);

    /* 最終ノードに勾配を設定 */
    vector_copy(doutput, &graph->nodes[graph->node_count - 1].gradient);

    /* 逆順に処理 */
    for(int i = graph->node_count - 1; i >= 0; i--)
    {
        GraphNode* pNode = &graph->nodes[i];

        if(pNode->type == GRAPH_NODE_INPUT)
            continue;

        int num = pNode->transform->input_count;

        const Vector* transform_inputs[num]; //C99以降で使える可変長配列×3行
        Vector dinputs[num];
        Vector* dinput_ptrs[num];

        /* 入力と入力勾配のVectorを準備 */
        for(int j = 0; j < num; j++)
        {
            GraphNode* pParent = &graph->nodes[pNode->input_nodes[j]];

            transform_inputs[j] = &pParent->output;

            dinputs[j] = vector_create(pParent->output_size);
            dinput_ptrs[j] = &dinputs[j];
        }

        /* TransformのBackward */
        pNode->transform->backward(pNode->transform, transform_inputs, &pNode->gradient, dinput_ptrs);

        /* 各入力元へ勾配を加算 */
        for(int j = 0; j < num; j++)
        {
            GraphNode* pParent = &graph->nodes[pNode->input_nodes[j]];

            vector_add(&dinputs[j], &pParent->gradient);

            vector_destroy(&dinputs[j]);
        }
    }
}

/* Update */
void graph_update(ComputationGraph* graph, double lr)
{
    for(int i = 0; i < graph->node_count; i++)
    {
        Transform* pTransform = graph->nodes[i].transform;

        if(pTransform != NULL && pTransform->update != NULL)
            pTransform->update(pTransform, lr);
    }
}

/* ===== Neural Network ===== */

void train(ComputationGraph* pNet, const Matrix x[], const Matrix* pT, double lr, int epoch)
{
    int input_count = pNet->input_port_count;
    int output_size = pNet->nodes[pNet->node_count - 1].output_size;

    for (int e = 0; e < epoch; e++)
    {
        for (int r = 0; r < pT->rows; r++)
        {
            /* 入力ポート数に応じてVector配列を準備する */
            Vector* inputs = calloc(input_count, sizeof(Vector));

            for(int i = 0; i < input_count; i++)
                inputs[i] = matrix_get_row(&x[i], r);
            
            /* 教師データと出力 */
            Vector t = matrix_get_row(pT, r);
            Vector y = vector_create(output_size);
            Vector dy = vector_create(output_size);

            /* Forward */
            graph_forward(pNet, inputs, &y);

            /* Lossの微分 */
            sse_derivative(&y, &t, &dy);

            /* Backward */
            graph_backward(pNet, &dy);

            /* Update */
            graph_update(pNet, lr);

            /* メモリ解放 */
            vector_destroy(&t);
            vector_destroy(&y);
            vector_destroy(&dy);

            for(int i = 0; i < input_count; i++)
                vector_destroy(&inputs[i]);

            free(inputs);
        }
    }
}


/* 結果を表示 */
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


void show_result(ComputationGraph* pNet, Matrix x[], const Matrix* pT)
{
    int input_count = pNet->input_port_count;
    int output_size = pNet->nodes[pNet->node_count - 1].output_size;

    for(int r = 0; r < pT->rows; r++)
    {
        /* 入力ポート数に応じてVector配列を準備する */
        Vector* inputs = calloc(input_count, sizeof(Vector));

        for(int i = 0; i < input_count; i++)
            inputs[i] = matrix_get_row(&x[i], r);

        /* 教師データと出力 */
        Vector t = matrix_get_row(pT, r);        
        Vector y = vector_create(output_size);
        
        /* 推論 */
        graph_forward(pNet, inputs, &y);
        
        /* 表示 */
        for(int i = 0; i < input_count; i++)
       {
            printf("Input %d: ", i);
            vector_print(&inputs[i]);
        }
        printf(" -> ");
        vector_print(&y);
        printf(" target = ");
        vector_print(&t);
        printf("\n");

        /* メモリ解放 */
        vector_destroy(&t);
        vector_destroy(&y);

        for(int i = 0; i < input_count; i++)
            vector_destroy(&inputs[i]);

        free(inputs);
    }
}


/* 開始 */
int main(void)
{
    srand(1);

    /* 入力データ（XORゲート） */
    double data_x[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Matrix x[1];
    x[0] = matrix_create(4, 2);
    
    for(int i = 0; i < x[0].rows; i++)
        for(int j = 0; j < x[0].cols; j++)
            matrix_set(&x[0], i, j, data_x[i][j]);

    /* 教師データ（XORゲート） */
    double data_t[][1] = {{0}, {1}, {1}, {0}};
    
    Matrix t = matrix_create(4, 1);

    for(int i = 0; i < t.rows; i++)
        for(int j = 0; j < t.cols; j++)
            matrix_set(&t, i, j, data_t[i][j]);

    /* Linear */
    Linear linear1 = linear_create(2, 3);
    Linear linear2 = linear_create(3, 1);

    /* 重みの初期化 */
    linear_randomize(&linear1);
    linear_randomize(&linear2);

    /* Activation */
    Activation act1 = activation_create(3, sigmoid, sigmoid_backward);
    Activation act2 = activation_create(1, sigmoid, sigmoid_backward);

    /* Transformとして登録 */
    Transform linear_transform1 = transform_from_linear(&linear1);
    Transform act_transform1 = transform_from_activation(&act1);

    Transform linear_transform2 = transform_from_linear(&linear2);
    Transform act_transform2 = transform_from_activation(&act2);

    const int node_inputs1[] = {0};
    const int node_inputs2[] = {1};
    const int node_inputs3[] = {2};
    const int node_inputs4[] = {3};

    GraphNodeSpec specs[] =
    {
        /* ノード0 : 外部入力a */
        {
            GRAPH_NODE_INPUT, 
            NULL,
            NULL,
            0,
            2
        },

        /* ノード1 : Linear1 */
        {
            GRAPH_NODE_TRANSFORM, 
            &linear_transform1,
            node_inputs1,
            -1,
            3
        },

        /* ノード2 : Activation1 */
        {
            GRAPH_NODE_TRANSFORM,
            &act_transform1,
            node_inputs2,
            -1,
            3
        },

        /* ノード3 : Linear2 */
        {
            GRAPH_NODE_TRANSFORM,
            &linear_transform2,
            node_inputs3,
            -1,
            1
        },
        
        /* ノード4 : Activation2 */
        {
            GRAPH_NODE_TRANSFORM,
            &act_transform2,
            node_inputs4,
            -1,
            1
        }
    };
    
    /* 出力 */
    Vector y = vector_create(1);
    Vector dy = vector_create(1);
    
    /* ComputationGraphを生成 */
    ComputationGraph  graph = graph_create(specs, 5);

    /* 学習 */
    double lr = 0.5;
    int epoch = 2000;

    train(&graph, x, &t, lr, epoch);

    /* 結果表示 */
    printf("=== Result ===\n");
    show_result(&graph, x, &t);

    /* メモリ解放 */
    graph_destroy(&graph);
    linear_destroy(&linear1);
    linear_destroy(&linear2);
    activation_destroy(&act1);
    activation_destroy(&act2);
    matrix_destroy(&x[0]);
    matrix_destroy(&t);

    return 0;
}
