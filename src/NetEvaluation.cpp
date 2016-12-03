/*
 * NetEvaluation.cpp
 *
 *  Created on: Dec 3, 2016
 *      Author: AnhTu
 */

#include "../header/NetEvaluation.h"
#include <iostream>

using namespace std;

double NetEvaluation::mean_net_accuracy(DataManagement* data_man,
		ParamBlock* weights, NetworkManipulation* net_man) {

	double mean = 0;
	int num_tr_exps = data_man->get_num_tr_exps();
	int batch_size = data_man->get_batch_size();
	TrainingExample* data_set;
	ParamBlock* expected_out;
	int ratio =
			(num_tr_exps % batch_size == 0) ?
					num_tr_exps / batch_size : num_tr_exps / batch_size + 1;
	int out;

	for (int i = 0; i < ratio; i++) {
		data_man->get_data_from_file();
		data_set = data_man->get_training_set();
		for (int j = 0; j < batch_size; j++) {
			if (j + i * batch_size < num_tr_exps) {
				expected_out = net_man->forwardProp(weights,
						data_set[j].get_input());

				out = get_pos_max_out(expected_out);
				mean += (out
						== (int(data_set[j].get_output())
								% net_man->get_output_size())) ? 1 : 0;

//				cout << "i " << i << " j " << j << " out "
//						<< data_set[j].get_output() << " expted " << out
//						<< endl;
//				for (int k = 0; k < 10; k++)
//					cout << expected_out->getParam(k, 0) << ' ';
//				cout << endl;

			}
		}
	}
	mean = mean / double(num_tr_exps) * 100.0;
	return mean;

}

int NetEvaluation::get_pos_max_out(ParamBlock* params) {
	int pos = 0;
	double max = params->getParam(0, 0);
	for (int i = 1; i < params->get_dim_x(); i++) {
		if (params->getParam(i, 0) > max) {
			max = params->getParam(i, 0);
			pos = i;
		}
	}
	return pos;
}

