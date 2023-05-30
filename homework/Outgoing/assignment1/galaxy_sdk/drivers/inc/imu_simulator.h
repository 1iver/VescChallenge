/**
 * Copyright (C) 2023 VeriSilicon Holdings Co., Ltd.
 * All rights reserved.
 *
 * @file imu_simulator.h
 * @brief public head file of imu simulator
 * @author Shaobo Tu <Shaobo.Tu@verisilicon.com>
 */

#ifndef _IMU_SIMULATOR_H
#define _IMU_SIMULATOR_H

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup SIM_IMU
 *  @brief IMU simulator code.
 *  @ingroup DRIVER
 *  @{
 */

#include "hal_imu.h"

/**
 * @brief Init the device for imu simulator
 *
 * @param imu_device handle of imu device
 * @param imu_config imu sensor configuration
 * @return 0 for succeed, others for failure
 */
int imu_simulator_device_init(ImuDevice *imu_device,
                              const ImuConfig *imu_config);

/**
 * @brief Deinit the device for imu simulator
 *
 * @param imu_device handle of imu device
 * @return 0 for succeed, others for failure
 */
int imu_simulator_device_deinit(ImuDevice *imu_device);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* _IMU_SIMULATOR_H */
