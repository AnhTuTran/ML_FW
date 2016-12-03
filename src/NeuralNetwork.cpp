/*
 * NeuralNetwork.cpp
 *
 *  Created on: Nov 30, 2016
 *      Author: AnhTu
 */

#include "../header/NeuralNetwork.h"
#include "../header/ML_FW.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

NeuralNetwork::NeuralNetwork(DataManagement* data_management,
		NetworkManipulation* net_manipulation) {
	this->data_management = data_management;
	this->net_manipulation = net_manipulation;
	this->gradients = new ParamBlock[net_manipulation->get_num_layers() - 1];
	this->weights = new ParamBlock[net_manipulation->get_num_layers() - 1];

	for (int i = 0; i < net_manipulation->get_num_layers() - 1; i++) {
		if (i == 0) {
			this->gradients[i].allo_ParamBlock(
					net_manipulation->get_hidden_size(),
					net_manipulation->get_input_size() + 1);
			this->weights[i].allo_ParamBlock(
					net_manipulation->get_hidden_size(),
					net_manipulation->get_input_size() + 1);
		} else if (i == net_manipulation->get_num_layers() - 2) {
			this->gradients[i].allo_ParamBlock(
					net_manipulation->get_output_size(),
					net_manipulation->get_hidden_size() + 1);
			this->weights[i].allo_ParamBlock(
					net_manipulation->get_output_size(),
					net_manipulation->get_hidden_size() + 1);
		} else {
			this->gradients[i].allo_ParamBlock(
					net_manipulation->get_hidden_size(),
					net_manipulation->get_hidden_size() + 1);
			this->weights[i].allo_ParamBlock(
					net_manipulation->get_hidden_size(),
					net_manipulation->get_hidden_size() + 1);
		}
	}
	// Initialize weights
	double epsilon_init = 0.12;
	for (int i = 0; i < this->net_manipulation->get_num_layers() - 1; i++) {
		for (int j = 0; j < this->weights[i].get_dim_x(); j++)
			for (int k = 0; k < this->weights[i].get_dim_y(); k++)
				this->weights[i].setParam(j, k,
						double(rand()) / double(RAND_MAX) * 2 * epsilon_init
								- epsilon_init);
	}
}

NeuralNetwork::~NeuralNetwork() {
	if (!this->gradients)
		delete[] this->gradients;
	if (!this->weights)
		delete[] this->weights;
	if (!this->log)
		delete[] this->log;
#ifdef DEBUG
	std::cout << "Done in NeuralNetwork\n";
#endif
}

Log * NeuralNetwork::get_log() {
	if (!this->log)
		std::cout << "Empty log\n";
	return this->log;
}

double NeuralNetwork::getCostFunc() {
	double costFunc = 0;
	TrainingExample* training_set;
	ParamBlock* activation;
	double tr_exp_output;

	if (!this->data_management->is_data_ready())
		this->data_management->get_data_from_file();
	training_set = this->data_management->get_training_set();

	int batch_size = this->data_management->get_batch_size();
	int output_size = this->net_manipulation->get_output_size();
	int num_layers = this->net_manipulation->get_num_layers();

	for (int i = 0; i < batch_size; i++) {
		activation = this->net_manipulation->forwardProp(this->weights,
				training_set[i].get_input());

		for (int j = 0; j < output_size; j++) {
			tr_exp_output =
					((int(training_set[i].get_output()) % int(output_size)) == j) ?
							1 : 0;
			costFunc += tr_exp_output * std::log(activation->getParam(j, 0))
					+ (1 - tr_exp_output)
							* std::log(1 - activation->getParam(j, 0));
		}
	}
	costFunc *= -(1.0 / batch_size);

	double temp = 0;
	for (int i = 0; i < num_layers - 1; i++) {
		for (int j = 0; j < this->weights[i].get_dim_x(); j++)
			for (int k = 1; k < this->weights[i].get_dim_y(); k++)
				temp += pow(this->weights[i].getParam(j, k), 2.0);
	}
	temp *= (this->lambda / (2.0 * batch_size));

	costFunc += temp;
	return costFunc;
}

void NeuralNetwork::training(int epochs) {
	ParamBlock* gradients;
	int num_layers = this->net_manipulation->get_num_layers();
	this->log = new Log(epochs);

	for (int i = 0; i < epochs; i++) {
		this->data_management->get_data_from_file();

		gradients = this->net_manipulation->computeGradient(this->weights,
				this->data_management->get_training_set(),
				this->data_management->get_batch_size(), this->lambda);

		for (int i = 0; i < num_layers - 1; i++) {
			for (int j = 0; j < this->weights[i].get_dim_x(); j++)
				for (int k = 1; k < this->weights[i].get_dim_y(); k++)
					this->weights[i].setParam(j, k,
							this->weights[i].getParam(j, k)
									- this->alpha
											* gradients[i].getParam(j, k));
		}
		this->log->set_cost_num(i, getCostFunc());

		cout << "Epoch: " << i << " , cost function: "
				<< this->log->get_cost_num(i) << endl;

	}
}

ParamBlock* NeuralNetwork::get_weights() {
	return this->weights;
}
