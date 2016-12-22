//============================================================================
// Name        : ML_FW.cpp
// Author      : AnhTuTran
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "../header/TrainingExample.h"
#include "../header/DataManagement.h"
#include "../header/ParamBlock.h"
#include "../header/Log.h"
#include "../header/NetworkManipulation.h"
#include "../header/NeuralNetwork.h"
#include "../header/NetEvaluation.h"
#include "../header/ML_FW.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void testTrainingExample();
void testDataManagement();
void testParamBlock();
void testLog();
void testNetworkManipulation();
void testNeuralNetwork();
void testGradient_NetworkMani();
void test_mul_mat_parallel();

int main() {
	//testTrainingExample();
	//testDataManagement();
	//testParamBlock();
	//testLog();
	//testNetworkManipulation();
	testNeuralNetwork();
	//testGradient_NetworkMani();
	//test_mul_mat_parallel();

}

void testTrainingExample() {
	double* in = new double[10];
	for (int i = 0; i < 10; i++)
		in[i] = i;
	double out = 10;

	TrainingExample tr_exp(in, out);
#ifdef DEBUG
	cout << "Output " << tr_exp.get_output() << endl;
	cout << "Input " << in << " " << tr_exp.get_input() << endl;
#endif
	tr_exp.set_tr_exp(in, 5);
#ifdef DEBUG
	cout << "Output " << tr_exp.get_output() << endl;
	cout << "Input " << in << " " << tr_exp.get_input() << endl;
#endif
	delete[] in;
}

void testDataManagement() {
	int batch_size = 5000;
	string data_file_name = "../data/data_set1.csv";
	DataManagement dataManagement(batch_size, data_file_name);

	dataManagement.get_data_from_file();

	TrainingExample* data = dataManagement.get_training_set();

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < 400; j++) {
			cout << data[i].get_input()[j] << " ";
		}
		cout << data[i].get_output() << endl;
	}
}

void testParamBlock() {
	int dim_x = 5;
	int dim_y = 6;
	ParamBlock param_block(dim_x, dim_y);
	cout << "MAT\n";
	for (int i = 0; i < dim_x; i++) {
		for (int j = 0; j < dim_y; j++) {
			param_block.setParam(i, j, i + j);
		}
	}
	for (int i = 0; i < dim_x; i++) {
		for (int j = 0; j < dim_y; j++) {
			cout << param_block.getParam(i, j) << " ";
		}
		cout << endl;
	}

	cout << "One\n";
	ParamBlock one(1, 1);
	for (int i = 0; i < one.get_dim_x(); i++) {
		for (int j = 0; j < one.get_dim_y(); j++) {
			one.setParam(i, j, 1);
		}
	}
	for (int i = 0; i < one.get_dim_x(); i++) {
		for (int j = 0; j < one.get_dim_y(); j++) {
			cout << one.getParam(i, j) << " ";
		}
		cout << endl;
	}

	cout << "Vec\n";
	ParamBlock vec(3, 1);
	for (int i = 0; i < vec.get_dim_x(); i++) {
		for (int j = 0; j < vec.get_dim_y(); j++) {
			vec.setParam(i, j, i);
		}
	}
	for (int i = 0; i < vec.get_dim_x(); i++) {
		for (int j = 0; j < vec.get_dim_y(); j++) {
			cout << vec.getParam(i, j) << " ";
		}
		cout << endl;
	}

	cout << "Vec tranpose\n";
	ParamBlock vec_tran(1, 3);
	for (int i = 0; i < vec_tran.get_dim_x(); i++) {
		for (int j = 0; j < vec_tran.get_dim_y(); j++) {
			vec_tran.setParam(i, j, j);
		}
	}
	for (int i = 0; i < vec_tran.get_dim_x(); i++) {
		for (int j = 0; j < vec_tran.get_dim_y(); j++) {
			cout << vec_tran.getParam(i, j) << " ";
		}
		cout << endl;
	}
}

void testLog() {
	int epochs = 5;
	Log log(epochs);

	for (int i = 0; i < epochs; i++)
		log.set_cost_num(i, 2 * i);

	for (int i = 0; i < epochs; i++)
		cout << log.get_cost_num(i) << ' ';
	cout << endl;
}

void testNetworkManipulation() {
	int input_size = 4;
	int output_size = 2;
	int hidden_size = 3;
	int num_layers = 3;
	NetworkManipulation netMan(input_size, output_size, hidden_size,
			num_layers);

	ParamBlock weights[2];
	ParamBlock *activation;
	double in[] = { 1, 1, 1, 1 };
	weights[0].allo_ParamBlock(3, 5);
	weights[1].allo_ParamBlock(2, 4);

	for (int i = 0; i < weights[0].get_dim_x(); i++) {
		for (int j = 0; j < weights[0].get_dim_y(); j++) {
			weights[0].setParam(i, j, 2);
		}
	}

	for (int i = 0; i < weights[1].get_dim_x(); i++) {
		for (int j = 0; j < weights[1].get_dim_y(); j++) {
			weights[1].setParam(i, j, 3);
		}
	}

	activation = netMan.forwardProp(weights, in);
	cout << "Forward\n";
	for (int i = 0; i < activation->get_dim_x(); i++) {
		cout << activation->getParam(i, 0) << endl;
	}
	cout << endl;

//	ParamBlock* backward = netMan.backProp(weights, 2);
//	cout << "Backward\n";
//	for (int i = 0; i < 2; i++) {
//		for (int j = 0; j < backward[i].get_dim_x(); j++) {
//			cout << backward[i].getParam(j, 0) << ' ';
//		}
//		cout << endl;
//	}
}

void testNeuralNetwork() {
	int num_threads = 1;
	omp_set_num_threads(num_threads);
	int batch_size = 5000;
	string data_file_name = "../data/data_set1.csv";
	DataManagement dataManagement(batch_size, data_file_name);
	double time = omp_get_wtime();

	// int input_size = 400; output_size = 10;
	// hidden_size = 25; num_layers = 3;

	int input_size = 400;
	int output_size = 10;
	int hidden_size = 25;
	int num_layers = 3;
	NetworkManipulation netMan(input_size, output_size, hidden_size,
			num_layers);

	NeuralNetwork neuralNet(&dataManagement, &netMan);
	cout << "Cost function: ";
	cout << neuralNet.getCostFunc() << endl;

	int epochs = 5;
	cout << "Training\n";
	neuralNet.training(epochs);

	NetEvaluation net_evaluation;
	cout << "Accuracy: "
			<< net_evaluation.mean_net_accuracy(&dataManagement,
					neuralNet.get_weights(), &netMan) << "%" << endl;
	time = omp_get_wtime() - time;
	printf("Execution time: %f \n", time);
}

void testGradient_NetworkMani() {

	int batch_size = 5000;
	string data_file_name = "../data/data_set1.csv";
	DataManagement dataManagement(batch_size, data_file_name);
	dataManagement.get_data_from_file();

	int input_size = 400;
	int output_size = 10;
	int hidden_size = 25;
	int num_layers = 3;

	NetworkManipulation netMan(input_size, output_size, hidden_size,
			num_layers);

	TrainingExample* training_set = dataManagement.get_training_set();
	double lambda = 0;
	ParamBlock weights[2];
	weights[0].allo_ParamBlock(25, 401);
	weights[1].allo_ParamBlock(10, 26);

	for (int i = 0; i < weights[0].get_dim_x(); i++) {
		for (int j = 0; j < weights[0].get_dim_y(); j++) {
			weights[0].setParam(i, j, 10);
		}
	}

	for (int i = 0; i < weights[1].get_dim_x(); i++) {
		for (int j = 0; j < weights[1].get_dim_y(); j++) {
			weights[1].setParam(i, j, 20);
		}
	}

//	ParamBlock* activation;
//	activation = netMan.forwardProp(weights, training_set[0].get_input());
//	cout << "Forward\n";
//	for (int i = 0; i < activation->get_dim_x(); i++) {
//		cout << activation->getParam(i, 0) << endl;
//	}
//	cout << "out "<<training_set[0].get_output() << endl;
//
//	ParamBlock* deltas = netMan.backProp(weights, training_set[0].get_output());
//	for (int m = 0; m < 10; m++)
//		std::cout << deltas[1].getParam(m, 0) << ' ';
//	std::cout << '\n';

	ParamBlock *gradients;
	gradients = netMan.computeGradient(weights, training_set, batch_size,
			lambda);

//	/*
//	for (int i = 0; i < gradients[0].get_dim_x(); i++) {
//		for (int j = 0; j < gradients[0].get_dim_y(); j++) {
//			cout << gradients[0].getParam(i, j) << ' ';
//		}
//		cout << endl;
//	}
//	*/
	for (int i = 0; i < gradients[1].get_dim_x(); i++) {
		for (int j = 0; j < gradients[1].get_dim_y(); j++) {
			cout << gradients[1].getParam(i, j) << ' ';
		}
		cout << endl;
	}
}

void test_mul_mat_parallel() {
	int th = 1;
	omp_set_num_threads(th);

	NetworkManipulation net_nam(2, 2, 2, 2);

	int n = 500;
	ParamBlock a(n, n), b(n, n), c(n, n);
	double time = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			a.setParam(i, j, 1);
			b.setParam(i, j, 2);
		}


	//net_nam.mul_matrices(&a, &b, &c);
	time = omp_get_wtime() - time;

	cout << c.getParam(0, 0) << " " << c.getParam(n / 2, n / 2) << " "
			<< c.getParam(n-1, n-1) << endl;

	printf("Execution time: %f \n", time);
}
