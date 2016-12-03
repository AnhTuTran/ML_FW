#include "../header/TrainingExample.h"

TrainingExample::TrainingExample(double *in, double out) {
	this->in = in;
	this->out = out;
	//this->in_size = in_size;
	//this->out_size = out_size;
}

TrainingExample::TrainingExample() {
	this->in = 0;
	this->out = 0;
}


double* TrainingExample::get_input() {
	return this->in;
}

double TrainingExample::get_output() {
	return this->out;
}

void TrainingExample::set_tr_exp(double* in, double out) {
	if (this->in != 0 && this->in != in)
		delete[] this->in;
	this->in = in;
	this->out = out;
}

TrainingExample::~TrainingExample() {
	if (this->in != 0)
		delete[] in;
#ifdef DEBUG
	std::cout << "Done in TrainingExample\n";
#endif
}

