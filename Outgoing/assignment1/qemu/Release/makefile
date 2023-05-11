################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include galaxy_sdk/tasks/src/subdir.mk
-include galaxy_sdk/drivers/src/subdir.mk
-include galaxy_sdk/bsp/src/subdir.mk
-include galaxy_sdk/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
SECONDARY_FLASH += \
assignment1.hex \

SECONDARY_LIST += \
assignment1.lst \

SECONDARY_SIZE += \
assignment1.siz \


# All Target
all: assignment1.out secondary-outputs

# Tool invocations
assignment1.out: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GNU RISC-V Cross C++ Linker'
	riscv-nuclei-elf-g++ -march=rv32imafc -mabi=ilp32f -mtune=nuclei-300-series -mcmodel=medlow -mno-save-restore -O2 -ffunction-sections -fdata-sections -fno-common -Werror -Wall  -g -T "D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\n309_iot_qemu.ld" -nostartfiles -Xlinker --gc-sections -L"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\bsp\lib" -L"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\drivers\lib" -L"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\modules\lib" -L"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\os\lib" -L"D:\work\VescChallenge\Outgoing\assignment1\galaxy_sdk\osal\lib" -Wl,-Map,"assignment1.map" --specs=nano.specs --specs=nosys.specs -o "assignment1.out" -Wl,--start-group $(OBJS) $(USER_OBJS) $(LIBS) -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

assignment1.hex: assignment1.out
	@echo 'Invoking: GNU RISC-V Cross Create Flash Image'
	riscv-nuclei-elf-objcopy -O ihex "assignment1.out"  "assignment1.hex"
	@echo 'Finished building: $@'
	@echo ' '

assignment1.lst: assignment1.out
	@echo 'Invoking: GNU RISC-V Cross Create Listing'
	riscv-nuclei-elf-objdump --source --all-headers --demangle --line-numbers --wide "assignment1.out" > "assignment1.lst"
	@echo 'Finished building: $@'
	@echo ' '

assignment1.siz: assignment1.out
	@echo 'Invoking: GNU RISC-V Cross Print Size'
	riscv-nuclei-elf-size --format=berkeley "assignment1.out"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(SECONDARY_FLASH)$(SECONDARY_LIST)$(SECONDARY_SIZE)$(ASM_DEPS)$(S_UPPER_DEPS)$(C_DEPS)$(CPP_DEPS) assignment1.out
	-@echo ' '

secondary-outputs: $(SECONDARY_FLASH) $(SECONDARY_LIST) $(SECONDARY_SIZE)

.PHONY: all clean dependents

-include ../makefile.targets
