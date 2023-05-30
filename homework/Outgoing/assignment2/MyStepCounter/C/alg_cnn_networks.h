#ifndef __CNN_NETWORKS_H_
#define __CNN_NETWORKS_H_

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum AlgoError
{
    ALGO_ERR_GENERIC = -1,
    ALGO_NORMAL = 0
} AlgoError;

typedef enum ClassResult
{
    CLASS_0 = 0,
    CLASS_1 = 1
} ClassResult;

typedef struct LayerData
{
    uint16_t size;
    float *data;
} LayerData;

int networks_init(void);

/* input_data size should be 2 */
int foward_process(LayerData *input_data, int *class);

#endif