/*
 * Log.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */
#include "../header/Log.h"
#include "../header/ML_FW.h"
#include <iostream>

Log::Log(int epochs) {
	this->epochs = epochs;
	this->cost_func_num = new double[epochs];
#ifdef DEBUG
	std::cout << "Epochs " << this->epochs << std::endl;
#endif
}

double Log::get_cost_num(int epoch_num) {
	if (epoch_num >= this->epochs)
		return -1;
	return this->cost_func_num[epoch_num];
}

void Log::set_cost_num(int epoch_num, double cost_func) {
	if (epoch_num >= this->epochs)
			return;
	this->cost_func_num[epoch_num] = cost_func;
}

Log::~Log() {
	delete[] this->cost_func_num;
#ifdef DEBUG
	std::cout << "Done in Log\n";
#endif
}

