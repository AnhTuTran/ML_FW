/*
 * Log.h
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */

#ifndef _LOG_H
#define _LOG_H 1

class Log {
private:
	int epochs;
	double* cost_func_num;
public:
	Log(int epochs);
	double get_cost_num(int epoch_num);
	void set_cost_num(int epoch_num, double cost_func);
	~Log();
};

#endif /* HEADER_LOG_H_ */
