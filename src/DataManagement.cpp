#include "../header/DataManagement.h"
#include "../header/ML_FW.h"
#include <fstream>
#include <iostream>

using namespace std;

DataManagement::DataManagement(int batch_size, string data_file_name) {
	this->batch_size = batch_size;
	this->data_file_name = data_file_name;
	this->training_set = new TrainingExample[batch_size];
	this->tr_exp_num = 0;
	this->num_tr_exps = 0;
	this->tr_exp_length = 0;

	string line;
	ifstream data_file(this->data_file_name);
	//data_file.open(this->data_file_name, ios::in);

	if (data_file.is_open()) {
		while (getline(data_file, line)) {
			this->num_tr_exps++;
		}
		data_file.close();
	} else {
		cout << "Can not open data file\n";
	}

#ifdef DEBUG
	cout << "Batch size " << this->batch_size << endl;
	cout << "Data file name " << this->data_file_name << endl;
	cout << "Num_tr_exps " << this->num_tr_exps << endl;
	cout << "tr_exp_num " << this->tr_exp_num << endl;
#endif
}

void DataManagement::get_data_from_file() {
	if (this->data_ready && this->batch_size == this->num_tr_exps)
		return;

	if (this->tr_exp_num == this->batch_size)
		this->tr_exp_num = 0;

	string line;
	int i = 0;
	ifstream data_file(this->data_file_name);

	if (data_file.is_open()) {
		while (getline(data_file, line) && i < this->batch_size) {
			if (i == 0) {
				int size = 1;
				for (int i = 0; i < line.length(); i++)
					if (line[i] == ',')
						size++;
				this->tr_exp_length = size;
				//cout << "Size " << this->tr_exp_length << endl;
			}
			double* arr = string_to_arr_num(line, this->tr_exp_length);


			double* in = new double[this->tr_exp_length - 1];
			double out = arr[this->tr_exp_length - 1];
			for (int i = 0; i < this->tr_exp_length - 1; i++) {
				in[i] = arr[i];
			}

			this->training_set[i].set_tr_exp(in, out);
			this->tr_exp_num++;
			i++;

		}
		data_file.close();
		this->data_ready = true;
	} else {
		cout << "Can not open data file\n";
	}
}

TrainingExample* DataManagement::get_training_set() {
	return this->training_set;
}

double* DataManagement::string_to_arr_num(string str, int size) {
	int* space_pos = new int[size - 1];
	int k = 0;
	for (int j = 0; j < str.length(); j++) {
		if (str[j] == ',') {
			space_pos[k] = j;
			k++;
		}
	}

	string* sub_str = new string[size];
	for (int i = 0; i < size; i++) {
		if (i == 0)
			sub_str[i] = str.substr(0, space_pos[i]);
		else if (i == size - 1)
			sub_str[i] = str.substr(space_pos[i - 1] + 1,
					str.length() - space_pos[i - 1] - 1);
		else
			sub_str[i] = str.substr(space_pos[i - 1] + 1,
					space_pos[i] - space_pos[i - 1] - 1);
	}

	double* arr = new double[size];
	for (int i = 0; i < size; i++) {
		arr[i] = str_to_double(sub_str[i]);
		//cout << sub_str[i] << endl;
	}

	delete[] space_pos;
	delete[] sub_str;
	return arr;

}

double DataManagement::str_to_double(string str) {
	double num = 0;
	int length = str.length();

	int neg_pos = (str[0] == '-') ? 0 : string::npos;
	int point_pos = str.find('.');
	int e_pos = str.find('e');

	e_pos = (e_pos == string::npos) ? length : e_pos;
	int i = (neg_pos == string::npos) ? 0 : 1;
	for (; i < e_pos; i++) {
		if (i != point_pos) {
			num += str[i] - 48;
			if (i != e_pos - 1)
				num *= 10.0;
		}
	}
	for (int j = 0; (j < e_pos - 1 - point_pos) && (point_pos != string::npos);
			j++)
		num /= 10.0;

	if (e_pos != length) {
		int expo = 0;
		for (int i = e_pos + 2; i < length; i++) {
			expo += str[i] - 48;
			if (i != length - 1) {
				expo *= 10.0;
			}
		}

		if (str[e_pos + 1] == '-') {
			for (int i = 0; i < expo; i++)
				num /= 10.0;
		} else {
			for (int i = 0; i < expo; i++)
				num *= 10.0;
		}
	}

	return (neg_pos == string::npos) ? num : -num;
}

DataManagement::~DataManagement() {
	delete[] this->training_set;
#ifdef DEBUG
	std::cout << "Done in DataManagement\n";
#endif
}

int DataManagement::get_batch_size() {
	return this->batch_size;
}

bool DataManagement::is_data_ready() {
	return this->data_ready;
}

int DataManagement::get_num_tr_exps() {
	return this->num_tr_exps;
}
