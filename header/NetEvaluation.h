/*
 * NetEvaluation.h
 *
 *  Created on: Dec 3, 2016
 *      Author: AnhTu
 */

#ifndef _NETEVALUATION_H
#define _NETEVALUATION_H 1

#include "../header/DataManagement.h"
#include "../header/NetworkManipulation.h"

class NetEvaluation  {
public:
	double mean_net_accuracy(DataManagement* data_man, ParamBlock* weights, NetworkManipulation* net_man);
private:
	int get_pos_max_out(ParamBlock* params);
};

#endif /* HEADER_NETEVALUATION_H_ */
