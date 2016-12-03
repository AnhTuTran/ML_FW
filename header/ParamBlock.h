/*
 * ParamBlock.h
 *
 *  Created on: Nov 26, 2016
 *      Author: AnhTu
 */

#ifndef _PARAMBLOCK_H
#define _PARAMBLOCK_H 1

class ParamBlock {
private:
	double** params = 0;
	int dim_x = 0;
	int dim_y = 0;
public:
	ParamBlock();
	void allo_ParamBlock(int dim_x, int dim_y);
	ParamBlock(int dim_x, int dim_y);
	double getParam(int pos_x, int pos_y);
	void setParam(int pos_x, int pos_y, double param);
	// random symmetric breaking
	int get_dim_x();
	int get_dim_y();
	~ParamBlock();
};



#endif /* HEADER_PARAMBLOCK_H_ */
