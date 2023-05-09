#include "task_Hello_VeriSilicon.h"
static void *task_Hello_VeriSilicon;
static void task_Hello_VeriSilicon_exec(void *param)
{
	uart_printf("Hello VeriSilicon\r\n");
	osal_delete_task(task_Hello_VeriSilicon);
}
void task_Hello_VeriSilicon_create(void *param)
{
	task_Hello_VeriSilicon = osal_create_task(task_Hello_VeriSilicon_exec, "Hello_VeriSilicon", 512, 4, NULL);
}

