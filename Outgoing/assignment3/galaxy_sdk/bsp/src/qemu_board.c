/**
 * Copyright (C) 2023 VeriSilicon Holdings Co., Ltd.
 * All rights reserved.
 *
 * @file qemu_board.c
 * @brief Functions for qemu qemu board
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vs_conf.h"
#include "qemu_board.h"
#include "sys_common.h"
#include "vsd_error.h"
#include "hal_imu.h"
#include "imu_simulator.h"

static void *device_list[MAX_DEVICE_ID];

static const I2cConfig imu_i2c_cfg = {
    .address_width = I2C_HAL_ADDRESS_WIDTH_7BIT,
    .freq          = I2C_BUS_SPEED_400K,
    .work_mode     = I2C_MODE_MASTER,
    .xfer_mode     = XFER_MODE_POLLING,
    .dev_addr      = 0x68,
    .reg_width     = 1,
};

static const ImuConfig imu_sensor_config = {
    .power_pin = NULL,
    .data_pin  = NULL,
    .wake_pin  = NULL,
};

static inline int create_imu_device(void)
{
    int ret;
    /*init the bmi160 sensor driver*/
    ImuDevice *imu_dev = malloc(sizeof(ImuDevice));
    if (!imu_dev) {
        return VSD_ERR_NO_MEMORY;
    }
    memset(imu_dev, 0, sizeof(ImuDevice));
    imu_dev->bus_device.type    = BUS_TYPE_I2C;
    imu_dev->bus_device.port_id = 0;
    imu_dev->bus_device.i2c = hal_i2c_get_device(imu_dev->bus_device.port_id);
    imu_dev->bus_config.i2c = &imu_i2c_cfg;
    ret = imu_simulator_device_init(imu_dev, &imu_sensor_config);
    if (ret) {
        device_list[IMU_SENSOR_ID] = NULL;
        free(imu_dev);
        return VSD_ERR_HW;
    }
    device_list[IMU_SENSOR_ID] = imu_dev;
    return ret;
}

NON_XIP_TEXT
static int qemu_board_init(BoardDevice *board)
{
    create_imu_device();

    return VSD_SUCCESS;
}

APP_SECTION
void *qemu_board_find_device(uint8_t device_id)
{
    return device_list[device_id];
}

const BoardOperations qemu_board_ops = {
    .init        = qemu_board_init,
    .find_device = qemu_board_find_device,
};

APP_SECTION
const BoardOperations *board_get_ops(void)
{
    return &qemu_board_ops;
}
