/*
 * NeuralNetwork.h
 *
 *  Created on: Nov 30, 2016
 *      Author: AnhTu
 */

#ifndef _NEURALNETWORK_H
#define _NEURALNETWORK_H 1

#include "../header/DataManagement.h"
#include "../header/ParamBlock.h"
#include "../header/Log.h"
#include "../header/NetworkManipulation.h"

class NeuralNetwork {
	Log* log = 0;
	ParamBlock* weights = 0;
	ParamBlock* gradients = 0;
	DataManagement* data_management = 0;
	NetworkManipulation* net_manipulation = 0;
	double lambda = 10; // regularization parameter
	double alpha = 1.5; // learning rate
public:
	NeuralNetwork(DataManagement* data_management,
			NetworkManipulation* net_manipulation);
	~NeuralNetwork();
	double getCostFunc();
	ParamBlock* get_weights();
	ParamBlock* getNumericalGradient();
	//ParamBlock* computeGradient();
	void training(int epochs);
	Log* get_log();
};

#endif /* HEADER_NEURALNETWORK_H_ */
