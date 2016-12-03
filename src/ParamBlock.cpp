/*
 * ParamBlock.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */

#include "../header/ParamBlock.h"
#include "../header/ML_FW.h"
#include <iostream>
using namespace std;

ParamBlock::ParamBlock() {

}

void ParamBlock::allo_ParamBlock(int dim_x, int dim_y) {
	if (!this->params) {
		this->dim_x = dim_x;
		this->dim_y = dim_y;

		if (dim_x == 1) {
			this->params = new double*;
			*this->params = new double[dim_y];
		} else {
			this->params = new double*[dim_x];
			for (int i = 0; i < dim_x; i++)
				this->params[i] = new double[dim_y];
		}
	} else {
		cout << "Block already allocated";
	}
}

ParamBlock::ParamBlock(int dim_x, int dim_y) {
	this->dim_x = dim_x;
	this->dim_y = dim_y;

	if (dim_x == 1) {
		this->params = new double*;
		*this->params = new double[dim_y];
	} else {
		this->params = new double*[dim_x];
		for (int i = 0; i < dim_x; i++)
			this->params[i] = new double[dim_y];
	}

#ifdef DEBUG
	cout << "dim_x " << this->dim_x << " dim_y " << this->dim_y << endl;
#endif
}
double ParamBlock::getParam(int pos_x, int pos_y) {

	if (this->dim_x == 1 && this->dim_y == 1)
		return **this->params;
	else if (this->dim_x == 1)
		return (*this->params)[pos_y];
	else if (this->dim_y == 1)
		return *(this->params[pos_x]);
	else
		return this->params[pos_x][pos_y];

}
void ParamBlock::setParam(int pos_x, int pos_y, double param) {

	if (this->dim_x == 1 && this->dim_y == 1)
		**this->params = param;
	else if (this->dim_x == 1)
		(*this->params)[pos_y] = param;
	else if (this->dim_y == 1)
		*(this->params[pos_x]) = param;
	else
		this->params[pos_x][pos_y] = param;

}
ParamBlock::~ParamBlock() {
	if (dim_x == 1) {
		delete[] *this->params;
		delete this->params;
	} else {
		for (int i = 0; i < this->dim_x; i++)
			delete[] this->params[i];
		delete[] this->params;
	}

#ifdef DEBUG
	cout << "Done in ParamBlock" << endl;
#endif
}

int ParamBlock::get_dim_x() {
	return this->dim_x;
}

int ParamBlock::get_dim_y() {
	return this->dim_y;
}
