################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/tasks/src/task_Hello_VeriSilicon.c 

OBJS += \
./galaxy_sdk/tasks/src/task_Hello_VeriSilicon.o 

C_DEPS += \
./galaxy_sdk/tasks/src/task_Hello_VeriSilicon.d 


# Each subdirectory must supply rules for building sources it contributes
galaxy_sdk/tasks/src/task_Hello_VeriSilicon.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/tasks/src/task_Hello_VeriSilicon.c
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross C Compiler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\tasks\inc" -std=gnu11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


