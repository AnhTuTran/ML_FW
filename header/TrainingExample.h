#ifndef _TRAININGEXAMPLE_H
#define _TRAININGEXAMPLE_H 1

class TrainingExample {
private:
	double* in;
	double out;
	//int in_size;
	//int out_size;
public:
	double* get_input();
	double get_output();
	TrainingExample(double* in, double out);
	TrainingExample();
	void set_tr_exp(double* in, double out);
	~TrainingExample();
};

#endif
