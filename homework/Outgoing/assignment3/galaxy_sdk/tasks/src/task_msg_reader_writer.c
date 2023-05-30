/*
 * @author 1iver
 * @file task_msg_reader_writer.c
 * @brief reader task and writer task with semaphore for sync
 *
 */
#include "task_msg_reader_writer.h"
typedef struct TimeEventMsg{
    uint64_t time;
    uint32_t msg_index;
}TimeEventMsg;
enum CustomEvent{
    CUSTOM_EVENT_DEFAULT=0x08000100,  //0x08 user defined event, 0x0001 class 1
    CUSTOM_EVENT_TIME, //reader writer event
};
static void *init_task_handle = NULL;
static void *reader_task_handle = NULL;
static void *writer_task_handle = NULL;
static int reader_msg_handler(void *cobj, uint32_t event_id, void *msg)
{
    switch(event_id)
    {
        case CUSTOM_EVENT_TIME:
            //llu not support
            uart_printf("msg_index:%" PRIu32,((TimeEventMsg*)msg)->msg_index);
            uart_printf(", send:%" PRIu32 ", receive:%" PRIu32 "\r\n",
                        (uint32_t)(((TimeEventMsg*)msg)->time),
                        (uint32_t)osal_get_uptime());
            osal_free(msg);
            break;
        default:
            uart_printf("%08" PRIx32 "\r\n",event_id);
    }
    return EVENT_OK;
}
static void task_msg_reader(void* sync)
{
    void* manager=vpi_event_new_manager(COBJ_CUSTOM_MGR,reader_msg_handler);
    int event_code = vpi_event_register(CUSTOM_EVENT_TIME,manager);
    if(event_code==EVENT_OK)
    {
        osal_sem_post(sync);
        while(1){
            if(vpi_event_listen(manager)==EVENT_ERROR)
                break;
        }
    }else
    {
        uart_printf("reader's event register failed!\r\n");
    }
    vpi_event_unregister(CUSTOM_EVENT_TIME,manager);
    osal_delete_task(reader_task_handle);
}

static void task_msg_writer(void* sync)
{
    int return_code = osal_sem_wait(sync,1000);
    osal_free(sync);
    if(return_code==OSAL_TRUE)
    {
        vpi_event_new_manager(COBJ_CUSTOM_MGR,NULL);
        uint32_t msg_index = 0;
        while(1){
            TimeEventMsg* msg = osal_malloc(sizeof(TimeEventMsg));
            msg->msg_index = msg_index++;
            msg->time = osal_get_uptime();
            if(vpi_event_notify(CUSTOM_EVENT_TIME,msg)==EVENT_ERROR)
                break;
            osal_sleep(10);
        }
    }else
    {
        uart_printf("sync error\r\n");
    }
    osal_delete_task(writer_task_handle);
}
static void reader_writer_task_init(void* params)
{
    OsalSemaphore* sync = osal_malloc(sizeof(OsalSemaphore)); //try malloc
    if(sync == NULL)
    {
        uart_printf("task reader_writer init failed, no enough space\r\n");
    }else
    {
        int return_code = osal_init_sem(sync);
        if(return_code != OSAL_TRUE)
        {
            uart_printf("sync semaphore init failed!\r\n");
        }else{
            reader_task_handle=osal_create_task(task_msg_reader,"reader_task",
                                                 256,2,sync);
            writer_task_handle=osal_create_task(task_msg_writer,"writer_task",
                                                 256,2,sync);
        }
    }
    osal_delete_task(init_task_handle);
}
void tasks_init()
{
    init_task_handle = osal_create_task(reader_writer_task_init,"init task",256,
                                        1,NULL);
}




