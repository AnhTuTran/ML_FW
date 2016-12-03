#ifndef _DATAMANAGEMENT_H
#define _DATAMANAGEMENT_H 1

#include <string>
#include "../header/TrainingExample.h"

using namespace std;


class DataManagement {
private:
	bool data_ready = false;
	int batch_size;
	int num_tr_exps;
	int tr_exp_num;
	int tr_exp_length;
	string data_file_name;
	TrainingExample* training_set;

public:
	void get_data_from_file();
	TrainingExample* get_training_set();
	DataManagement(int batch_size, string data_file_name);
	~DataManagement();
	int get_batch_size();
	int get_num_tr_exps();
	bool is_data_ready();
private:
	double str_to_double(string str);
	double* string_to_arr_num(string str, int size);
};

#endif
