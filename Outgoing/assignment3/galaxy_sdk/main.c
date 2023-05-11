/**
 * Copyright (C) 2022 VeriSilicon Holdings Co., Ltd.
 * All rights reserved.
 *
 * @file main.c
 * @brief Functions for main
 * @author Shaobo Tu <Shaobo.Tu@verisilicon.com>
 */

#include <stdint.h>
#include <stdlib.h>
#include "vs_conf.h"
#include "soc_init.h"
#include "bsp.h"
#include "uart_printf.h"
#include "board.h"
#include "osal_heap_api.h"
#include "osal_task_api.h"
#include "vpi_error.h"
#include "task_msg_reader_writer.h"
//static void *sample_task;
////static BoardDevice board_dev;
//static void *init_task;

//void task_sample(void *param)
//{
////    int count = 0;
////
////    while (count < 10) {
////        count++;
////        uart_printf("Sample task count %d\r\n", count);
////        osal_sleep(1000);
////    }
//
//    uart_printf("Finish sample task!\r\n");
//    osal_delete_task(sample_task);
//}
//void task_init_app(void *param)
//{
//    /* Initialize board */
////    board_register(board_get_ops());
////    board_init((void *)&board_dev);
////    if (board_dev.name)
////        uart_printf("Board: %s", board_dev.name);
////
////    uart_printf("Hello VeriHealth!\r\n");
//
//    sample_task = osal_create_task(task_msg_reader, "task_sample", 512, 4, NULL);
//    sample_task = osal_create_task(task_msg_reader, "task_sample1", 512, 4, NULL);
//    osal_delete_task(init_task);
//}
int main(void)
{
    int ret;

    /* Initialize soc */
    ret = soc_init();
    if (vsd_to_vpi(ret) != VPI_SUCCESS) {
        if (vsd_to_vpi(ret) == VPI_ERR_LICENSE)
            uart_printf("Invalid SDK license!\r\n");
        else
            uart_printf("Soc Init Error!\r\n");
        return vsd_to_vpi(ret);
    }
    /* Initialize uart */
    uart_debug_init();
    /* Initialize bsp */
    bsp_init();
    /* Create init task */
    tasks_init();
    /* Start os */
    osal_start_scheduler();

    return 0;
}
