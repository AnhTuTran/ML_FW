/*
 * NetworkManipulation.h
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */

#ifndef _NETWORKMANIPULATION_H
#define _NETWORKMANIPULATION_H 1

#include "../header/ParamBlock.h"
#include "../header/TrainingExample.h"

class NetworkManipulation {
private:
	ParamBlock* activations = 0;
	ParamBlock* z = 0;
	ParamBlock* deltas = 0;
	ParamBlock* gradients = 0;
	int input_size;
	int output_size;
	int hidden_size;
	int num_layers;
public:
	NetworkManipulation(int input_size, int output_size, int hidden_size,
			int num_layers);
	~NetworkManipulation();
	ParamBlock* forwardProp(ParamBlock* weights, double* tr_exp_input);
	ParamBlock* backProp(ParamBlock* weights, double tr_exp_output);
	ParamBlock* computeGradient(ParamBlock* weights,
			TrainingExample* training_set, int batch_size, int lambda);
	int get_input_size();
	int get_output_size();
	int get_hidden_size();
	int get_num_layers();

private:
	double sigmoid(double x);
	double sigmoidGradient(double x);
	void mul_matrices(ParamBlock* weights, ParamBlock* activation,
			ParamBlock* z);
	ParamBlock* tranpose_mat(ParamBlock* mat);
};

#endif /* HEADER_NETWORKMANIPULATION_H_ */
