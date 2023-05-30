#include "alg_bp_networks.h"

/* define the BP networks*/
#define NETWORKS_INPUT_SIZE (2)   /* input data feature dimension */
#define NETWORKS_HIDDEN_SIZE (10) /* hidden layer neuron number */
#define NETWORKS_OUTPUT_SIZE (2)  /* class number */

/* weight: reshape the original [row_num, column_num] matrix into [row_num*column_num, 1] along columns */
typedef struct NetwoksLayer
{
    uint16_t row_num;
    uint16_t column_num;
    const float *weight;
    const float *bias;
} NetwoksLayer;
typedef struct ConvLayer{
    uint16_t cov_num;
    uint16_t cov_size;
    const float *weight;
    const float *bias;
}
typedef struct BpNetworks
{
    NetwoksLayer input_layer;
    NetwoksLayer hidden_layer;
} BpNetworks;

static BpNetworks bp_networks;

//* changed weight
const float input_layer_weight[NETWORKS_INPUT_SIZE * NETWORKS_HIDDEN_SIZE] = {0.9028494358f, -0.4522040486f, -3.2800829411f, -1.3664128780f,
                                                                              -2.4513528347f, -3.7540333271f, 1.6335811615f, -0.8627996445f,
                                                                              0.7127078176f, -0.4096514881f, 0.9569021463f, 0.2273463607f,
                                                                              1.3702820539f, -0.6932174563f, 0.5542764664f, 0.8269275427f,
                                                                              -2.8319714069f, -1.8603184223f, -1.8588626385f, -0.6099621058f};

const float input_layer_bias[NETWORKS_HIDDEN_SIZE] = {0.4247229993f, -0.2404207140f, 3.8230521679f, 0.7308659554f,
                                                      0.2908494771f, 0.1764592379f, 0.6388018727f, 0.3470892310f,
                                                      -0.3197288215, 4.2836852074f};

const float hidden_layer_weight[NETWORKS_HIDDEN_SIZE * NETWORKS_OUTPUT_SIZE] = {-0.4589923620f, 2.3280203342f, -4.1144347191f, -1.1487400532f,
                                                                                -0.4644607008f, 0.5328536034f, -0.9638527632f, 0.7887286544f,
                                                                                2.4604980946f, 3.2788965702f, 1.0151129961f, -2.6160552502f,
                                                                                4.1275353432f, 1.6911562681f, 0.7115666866f, -0.3596055508f,
                                                                                1.3174898624f, -0.9191195965f, -2.2845132351f, -3.5295078754f};

const float hidden_layer_bias[NETWORKS_OUTPUT_SIZE] = {1.4795401096f, -1.3724881411f};

static int linear_calculation(NetwoksLayer *layer, LayerData *input_data, LayerData *output_data)
{
    uint16_t i = 0, j = 0;
    float tmp_sum = 0.0f;

    if (!layer || !input_data || !output_data)
    {
        return ALGO_ERR_GENERIC;
    }

    if (input_data->size != layer->row_num || output_data->size != layer->column_num)
    {
        return ALGO_ERR_GENERIC;
    }

    for (i = 0; i < output_data->size; i++)
    {
        tmp_sum = 0.0f;
        for (j = 0; j < layer->row_num; j++)
        {
            tmp_sum += input_data->data[j] * layer->weight[i * layer->row_num + j];
        }
        tmp_sum += layer->bias[i];
        output_data->data[i] = tmp_sum;
    }

    return ALGO_NORMAL;
}

static int relu(LayerData *input_data)
{
    uint16_t i = 0;
    if (!input_data)
    {
        return ALGO_ERR_GENERIC;
    }
    for (i = 0; i < input_data->size; i++)
    {
        if (input_data->data[i] < 0.0f)
        {
            input_data->data[i] = 0.0f;
        }
    }
    return ALGO_NORMAL;
}

int networks_init(void)
{
    AlgoError ret;
    memset(&bp_networks, 0, sizeof(BpNetworks));

    bp_networks.input_layer.row_num = NETWORKS_INPUT_SIZE;
    bp_networks.input_layer.column_num = NETWORKS_HIDDEN_SIZE;
    bp_networks.input_layer.weight = input_layer_weight;
    bp_networks.input_layer.bias = input_layer_bias;

    bp_networks.hidden_layer.row_num = NETWORKS_HIDDEN_SIZE;
    bp_networks.hidden_layer.column_num = NETWORKS_OUTPUT_SIZE;
    bp_networks.hidden_layer.weight = hidden_layer_weight;
    bp_networks.hidden_layer.bias = hidden_layer_bias;

    return ALGO_NORMAL;
}

static ClassResult result_classification(LayerData *hidden_layer_output)
{
    uint16_t i = 0, j = 0;
    for (i = 0; i < hidden_layer_output->size; i++)
    {
        if (hidden_layer_output->data[i] > hidden_layer_output->data[j])
        {
            j = i;
        }
    }
    return (ClassResult)j;
}

int foward_process(LayerData *input_data, int *class)
{
    float input_layer_output_data[NETWORKS_HIDDEN_SIZE] = {0.0f};
    LayerData input_layer_output;
    input_layer_output.size = NETWORKS_HIDDEN_SIZE;
    input_layer_output.data = input_layer_output_data;
    float hidden_layer_output_data[NETWORKS_OUTPUT_SIZE] = {0.0f};
    LayerData hidden_layer_output;
    hidden_layer_output.size = NETWORKS_OUTPUT_SIZE;
    hidden_layer_output.data = hidden_layer_output_data;

    AlgoError ret;

    ret = linear_calculation(&bp_networks.input_layer, input_data, &input_layer_output);
    if (ret == ALGO_NORMAL)
    {
        ret = relu(&input_layer_output);
        if (ret == ALGO_NORMAL)
        {
            ret = linear_calculation(&bp_networks.hidden_layer, &input_layer_output, &hidden_layer_output);
            if (ret == ALGO_NORMAL)
            {
                *class = result_classification(&hidden_layer_output);
            }
        }
    }
    return ret;
}