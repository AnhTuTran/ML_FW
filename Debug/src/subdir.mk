################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/DataManagement.cpp \
../src/Log.cpp \
../src/ML_FW.cpp \
../src/NetEvaluation.cpp \
../src/NetworkManipulation.cpp \
../src/NeuralNetwork.cpp \
../src/ParamBlock.cpp \
../src/TrainingExample.cpp 

OBJS += \
./src/DataManagement.o \
./src/Log.o \
./src/ML_FW.o \
./src/NetEvaluation.o \
./src/NetworkManipulation.o \
./src/NeuralNetwork.o \
./src/ParamBlock.o \
./src/TrainingExample.o 

CPP_DEPS += \
./src/DataManagement.d \
./src/Log.d \
./src/ML_FW.d \
./src/NetEvaluation.d \
./src/NetworkManipulation.d \
./src/NeuralNetwork.d \
./src/ParamBlock.d \
./src/TrainingExample.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++-mp-6 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


