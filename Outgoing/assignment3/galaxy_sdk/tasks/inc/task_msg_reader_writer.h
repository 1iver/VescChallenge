/*
 * @author 1iver
 * @file task_msg_reader_writer.h
 * @brief reader task and writer task with semaphore for syncs
 * 
 */

#ifndef _TASK_MSG_READER_WRITER_H_
#define _TASK_MSG_READER_WRITER_H_
#include <stdint.h>           //uint32 definition 
#include <inttypes.h>
#include <stdio.h>
#include "osal_semaphore_api.h"
#include "osal_task_api.h"
#include "osal_heap_api.h"
#include "osal_time_api.h"
#include "uart_printf.h"
#include "vpi_event.h"


void tasks_init();
//void task_msg_reader(void* params);
//void task_msg_writer(void* params);
#endif /* _TASK_MSG_READER_WRITER_H_ */
