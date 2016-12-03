/*
 * NetworkManipulation.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */

#include "../header/NetworkManipulation.h"
#include "../header/ML_FW.h"
#include <math.h>
#include <iostream>

#define a_off_lay -1
#define z_off_lay -2
#define z_off_size -1
#define weights_off_lay -1
#define delta_off_lay -2

using namespace std;

NetworkManipulation::NetworkManipulation(int input_size, int output_size,
		int hidden_size, int num_layers) {

	this->activations = new ParamBlock[num_layers];
	this->gradients = new ParamBlock[num_layers - 1];
	this->deltas = new ParamBlock[num_layers - 1];
	this->z = new ParamBlock[num_layers - 1];

	this->hidden_size = hidden_size;
	this->input_size = input_size;
	this->num_layers = num_layers;
	this->output_size = output_size;

	for (int i = 0; i < num_layers; i++) {
		if (i == 0)
			this->activations[i].allo_ParamBlock(input_size + 1, 1);
		else if (i == num_layers - 1)
			this->activations[i].allo_ParamBlock(output_size, 1);
		else
			this->activations[i].allo_ParamBlock(hidden_size + 1, 1);
	}

	for (int i = 0; i < num_layers - 1; i++) {
		if (i == num_layers - 2) {
			this->deltas[i].allo_ParamBlock(output_size, 1);
			this->z[i].allo_ParamBlock(output_size, 1);
		} else {
			this->deltas[i].allo_ParamBlock(hidden_size, 1);
			this->z[i].allo_ParamBlock(hidden_size, 1);
		}
	}

	for (int i = 0; i < num_layers - 1; i++) {
		if (i == 0)
			this->gradients[i].allo_ParamBlock(hidden_size, input_size + 1);
		else if (i == num_layers - 2)
			this->gradients[i].allo_ParamBlock(output_size, hidden_size + 1);
		else
			this->gradients[i].allo_ParamBlock(hidden_size, hidden_size + 1);
	}
	/*
	 #ifdef DEBUG
	 double a[] = { 1, -0.5, 0, 0.5, 1 };
	 for (int i = 0; i < 5; i++)
	 std::cout << sigmoidGradient(a[i]) << ' ';
	 std::cout << std::endl;
	 #endif
	 */
}

NetworkManipulation::~NetworkManipulation() {
	if (!this->activations)
		delete[] this->activations;
	if (!this->deltas)
		delete[] this->deltas;
	if (!this->gradients)
		delete[] this->gradients;
	if (!this->z)
		delete[] this->z;
#ifdef DEBUG
	std::cout << "Done in NetworkManipulation\n";
#endif
}

double NetworkManipulation::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double NetworkManipulation::sigmoidGradient(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

ParamBlock* NetworkManipulation::forwardProp(ParamBlock* weights,
		double* tr_exp_input) {

	for (int i = 0; i <= this->input_size; i++) {
		if (i == 0)
			this->activations[1 + a_off_lay].setParam(i, 0, 1);
		else
			this->activations[1 + a_off_lay].setParam(i, 0,
					tr_exp_input[i - 1]);
	}

	for (int i = 2; i <= this->num_layers; i++) {
		mul_matrices(&weights[i - 1 + weights_off_lay],
				&this->activations[i - 1 + a_off_lay], &this->z[i + z_off_lay]);

		if (i != this->num_layers) {
			for (int j = 0; j < this->activations[i + a_off_lay].get_dim_x();
					j++) {
				if (j == 0)
					this->activations[i + a_off_lay].setParam(j, 0, 1);
				else
					this->activations[i + a_off_lay].setParam(j, 0,
							sigmoid(this->z[i + z_off_lay].getParam(j - 1, 0)));
			}
		} else {
			for (int j = 0; j < this->activations[i + a_off_lay].get_dim_x();
					j++) {
				this->activations[i + a_off_lay].setParam(j, 0,
						sigmoid(this->z[i + z_off_lay].getParam(j, 0)));
			}
		}

	}

	return &this->activations[this->num_layers + a_off_lay];
}

void NetworkManipulation::mul_matrices(ParamBlock* weights,
		ParamBlock* activation, ParamBlock* z) {

	for (int i = 0; i < z->get_dim_x(); i++) {
		for (int j = 0; j < z->get_dim_y(); j++) {
			z->setParam(i, j, 0);
		}
	}

	for (int i = 0; i < weights->get_dim_x(); i++) {
		for (int j = 0; j < activation->get_dim_y(); j++) {
			for (int k = 0; k < weights->get_dim_y(); k++) {
				z->setParam(i, j,
						z->getParam(i, j)
								+ weights->getParam(i, k)
										* activation->getParam(k, j));
			}
		}
	}
}

ParamBlock* NetworkManipulation::backProp(ParamBlock* weights,
		double tr_exp_output) {

	int out = int(tr_exp_output) % int(this->output_size);
	for (int i = 0;
			i < this->activations[this->num_layers + a_off_lay].get_dim_x();
			i++) {
		int tmp = (out == i) ? 1 : 0;
		this->deltas[this->num_layers + delta_off_lay].setParam(i, 0,
				this->activations[this->num_layers + a_off_lay].getParam(i, 0)
						- double(tmp));
	}

	for (int i = this->num_layers - 1; i >= 2; i--) {

		ParamBlock temp(weights[i + weights_off_lay].get_dim_y(),
				this->deltas[i + 1 + delta_off_lay].get_dim_y());
		mul_matrices(tranpose_mat(&weights[i + weights_off_lay]),
				&this->deltas[i + 1 + delta_off_lay], &temp);
		//std::cout << "AA2\n";
		for (int j = 0; j < this->z[i + z_off_lay].get_dim_x(); j++) {
			this->deltas[i + delta_off_lay].setParam(j, 0,
					temp.getParam(j + 1, 0)
							* sigmoidGradient(
									this->z[i + z_off_lay].getParam(j, 0)));
		}

	}
	return this->deltas;
}

ParamBlock* NetworkManipulation::tranpose_mat(ParamBlock* mat) {
	ParamBlock *tran_mat = new ParamBlock(mat->get_dim_y(), mat->get_dim_x());

	for (int i = 0; i < mat->get_dim_x(); i++) {
		for (int j = 0; j < mat->get_dim_y(); j++) {
			tran_mat->setParam(j, i, mat->getParam(i, j));
		}
	}
	return tran_mat;
}

int NetworkManipulation::get_input_size() {
	return this->input_size;
}

int NetworkManipulation::get_output_size() {
	return this->output_size;
}

int NetworkManipulation::get_hidden_size() {
	return this->hidden_size;
}

int NetworkManipulation::get_num_layers() {
	return this->num_layers;
}

ParamBlock* NetworkManipulation::computeGradient(ParamBlock* weights,
		TrainingExample* training_set, int batch_size, int lambda) {

	ParamBlock* deltas;
	ParamBlock temp[this->num_layers - 1];

	for (int i = 0; i < this->num_layers - 1; i++)
		temp[i].allo_ParamBlock(weights[i].get_dim_x(), weights[i].get_dim_y());

	for (int i = 0; i < this->num_layers - 1; i++)
		for (int j = 0; j < this->gradients[i].get_dim_x(); j++)
			for (int k = 0; k < this->gradients[i].get_dim_y(); k++)
				this->gradients[i].setParam(j, k, 0);

	for (int i = 0; i < batch_size; i++) {
		forwardProp(weights, training_set[i].get_input());
		deltas = backProp(weights, training_set[i].get_output());
		//
//		for (int m = 0; m < 10; m++)
//			std::cout << deltas[1].getParam(m,0) << ' ';
//		std::cout << '\n';
		//

		for (int i = 0; i < this->num_layers - 1; i++)
			mul_matrices(&deltas[i], tranpose_mat(&this->activations[i]),
					&temp[i]);

		for (int j = 0; j < this->num_layers - 1; j++)
			for (int k = 0; k < this->gradients[j].get_dim_x(); k++)
				for (int l = 0; l < this->gradients[j].get_dim_y(); l++) {
					this->gradients[j].setParam(k, l,
							this->gradients[j].getParam(k, l)
									+ temp[j].getParam(k, l));
				}
	}

//	for (int j = 0; j < this->gradients[1].get_dim_x(); j++)
//		for (int k = 0; k < this->gradients[1].get_dim_y(); k++)
//			std::cout << this->gradients[1].getParam(j, k) << ' ';
//	std::cout << "\n";

	for (int i = 0; i < this->num_layers - 1; i++)
		for (int j = 0; j < this->gradients[i].get_dim_x(); j++)
			for (int k = 0; k < this->gradients[i].get_dim_y(); k++) {
				if (k == 0) {
					this->gradients[i].setParam(j, k,
							this->gradients[i].getParam(j, k) / batch_size);
				} else {
					this->gradients[i].setParam(j, k,
							(this->gradients[i].getParam(j, k)
									+ lambda * weights[i].getParam(j, k))
									/ batch_size);
				}
			}

	return this->gradients;
}

