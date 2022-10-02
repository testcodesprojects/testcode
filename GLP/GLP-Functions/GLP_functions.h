
#ifndef _GLPfunctions_
#define _GLPfunctions_

#include "../GLP-Libraries/GLP_libraries.h"

template <typename T> void print(T x){std::cout << x << std::endl;}
template <typename T> void printmat(T x)
{for(size_t i=0;i<x.rows();i++){for(size_t j=0;j<x.columns();j++)std::cout << std::setprecision(3) << x(i,j) << ",";std::cout << std::endl;}}
inline double sign (double x) {if(x>0) return 1.0; else return -1.0;}

double log_det(sym_matrix &Q);

void get_txt_size_t(string s,size_t &info);
void get_txt_bool(string s,bool &info);
void get_txt_column(string s,col_vector &info);
void get_txt_string_column(string s,blaze::DynamicVector<string> &info);
void get_txt_string(string s,string &info);
void get_txt_string_p(string s,string &info);
void get_txt_double(string s,double &info);
void get_txt_int(string s,int &info);

void ginv(sym_matrix &Q, size_t &rankdef, sym_matrix &invQ);
void ginv_sym(sym_matrix &Q, size_t &rankdef, sym_matrix &invQ);
void ginv_sym_with_Qx(sym_matrix &Q, size_t &rankdef, sym_matrix &invQ);
void ginv_sym_with_eigenvals_bym2(sym_matrix &Q, size_t &rankdef,ptr_bowl &param,size_t r, sym_matrix &invQ);

void MGS_orthonormalization(col_matrix &G);

void myeign(row_matrix &hessian_theta_star, col_vector &eigenvalues,row_matrix  &eigenvectors);
void z2theta(col_vector &z,col_vector &theta, col_vector &theta_mode, col_vector eigen_values, row_matrix eigen_vectors);
void theta2z(col_vector &theta,col_vector &z, col_vector &theta_mode, col_vector eigen_values, row_matrix eigen_vectors);
void export_density(col_vector &x, col_vector &y, size_t &index, string T);
void export_margx_density(col_vector &x, col_vector &y, size_t &index, string T);

void normalize_simpson_321_p(col_vector &y);
double factorial(double n);
double logfactorial(double n);

void normalize_simpson_321_p_dic(col_vector &y, col_vector &d);
void log_fac(col_vector &logfac);

void seasoning(int num_tasks, int num_workers, int worker_index, bool include_boss, boost::ptr_vector<int> &tasks);
void seasoning_margx(int num_tasks, int num_workers, int worker_index, bool include_boss, boost::ptr_vector<int> &tasks);
int gradient_majic_spread(bool &central, int &pos, bool &left, bool &right, size_t &n);
bool hessian_majic_spread(int &task_number, int &index_i, int &step_i, int &index_j, int &step_j, size_t &n, bool safe_mode);
void task_position_connection(int task_number, row_matrix &hessian, double *value, size_t &n, bool safe_mode,row_matrix &savehihj);

void oven_gloves(int &task_number, double &value, size_t &n, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg);
int oven_majic_temperature(int &task_number, int &pos, size_t &n);
int check_oven(int &task_number, size_t &n);

void cupboard(ptr_bowl & param);
void drawer1(ptr_bowl & param);
void drawer2(ptr_bowl & param);
void increase_temp(ptr_bowl & param);
void decrease_temp(ptr_bowl & param);

void monitor_food_oven(ptr_bowl & param, size_t n);
void turnoff_oven(ptr_bowl & param, size_t n);
void export_vector(col_vector &x, string T);
void polyfit(	const std::vector<double> &t,
		const std::vector<double> &v,
		std::vector<double> &coeff,
		int order

	     );

/*
class GLP_functions
{	
	
};
*/

#endif
