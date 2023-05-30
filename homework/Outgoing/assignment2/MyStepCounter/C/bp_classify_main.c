#include <stdio.h>
#include "alg_bp_networks.h"
float data[2];
int label[200];

int main()
{
    uint16_t size = 2;
    LayerData input_data;
    int res, num = 0;

    FILE *fp = fopen("label.csv", "r");
    char str[101];
    while (fgets(str, 100, fp) != NULL)
    {
        label[num] = atoi(str);
        num++;
    }
    fclose(fp);

    networks_init();

    FILE *fp2 = fopen("feature.csv", "r");
    num = 0;
    while (fgets(str, 100, fp2) != NULL)
    {

        char *save_ptr;
        data[0] = atof(strtok_s(str, ",", &save_ptr));
        data[1] = atof(strtok_s(NULL, ",", &save_ptr));
        printf("%f,%f,", data[0], data[1]);
        input_data.data = data;
        input_data.size = size;

        foward_process(&input_data, &res);

        printf("pred:%d, label:%d\n", res, label[num]);
        num++;
    }
    fclose(fp2);
    return 0;
}
