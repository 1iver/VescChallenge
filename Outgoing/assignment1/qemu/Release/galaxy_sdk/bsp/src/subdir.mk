################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/newlib.c \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/qemu_board.c 

S_UPPER_SRCS += \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/intexc_riscv.S \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/portasm.S \
D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/startup_riscv.S 

OBJS += \
./galaxy_sdk/bsp/src/intexc_riscv.o \
./galaxy_sdk/bsp/src/newlib.o \
./galaxy_sdk/bsp/src/portasm.o \
./galaxy_sdk/bsp/src/qemu_board.o \
./galaxy_sdk/bsp/src/startup_riscv.o 

S_UPPER_DEPS += \
./galaxy_sdk/bsp/src/intexc_riscv.d \
./galaxy_sdk/bsp/src/portasm.d \
./galaxy_sdk/bsp/src/startup_riscv.d 

C_DEPS += \
./galaxy_sdk/bsp/src/newlib.d \
./galaxy_sdk/bsp/src/qemu_board.d 


# Each subdirectory must supply rules for building sources it contributes
galaxy_sdk/bsp/src/intexc_riscv.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/intexc_riscv.S
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross Assembler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -x assembler-with-cpp -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

galaxy_sdk/bsp/src/newlib.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/newlib.c
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross C Compiler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\tasks\inc" -std=gnu11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

galaxy_sdk/bsp/src/portasm.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/portasm.S
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross Assembler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -x assembler-with-cpp -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

galaxy_sdk/bsp/src/qemu_board.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/qemu_board.c
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross C Compiler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\tasks\inc" -std=gnu11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

galaxy_sdk/bsp/src/startup_riscv.o: D:/work/VescChallenge/Outgoing/assignment1/galaxy_sdk/bsp/src/startup_riscv.S
	@echo 'Building file: $<'
	@echo 'Invoking: GNU RISC-V Cross Assembler'
	riscv-nuclei-elf-gcc -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -x assembler-with-cpp -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\config\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\inc" -I"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\inc" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


