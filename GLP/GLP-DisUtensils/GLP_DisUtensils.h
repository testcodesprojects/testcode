
#ifndef _GLPDisUtensils_
#define _GLPDisUtensils_


#include "../GLP-Libraries/GLP_libraries.h"
#include "../GLP-Functions/GLP_functions.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//typedef void (* opt_fun1)(ptr &fx, col_vector &theta,size_t xsize, ptr_bowl &param,sym_matrix &Qx, sym_matrix &Qstar, col_vector &xstar,col_vector &mu, col_vector &grad_x_like,col_vector &b_vec,col_vector &x_intial,blaze::DiagonalMatrix< blaze::DynamicMatrix<double> > &D,boost::scoped_ptr<double> &y, col_vector &Ax);
typedef double (* opt_fun2)(const VectorXd& x, VectorXd& grad, ptr_bowl &param, col_vector &xstar);
//typedef void (* fun_type1)(ptr &fx, col_vector &theta,ptr_bowl &param, sym_matrix &Qx, sym_matrix &Qstar, col_vector &xstar,col_vector &mu, col_vector &grad_x_like,col_vector &b_vec,col_vector &x_intial,blaze::DiagonalMatrix< blaze::DynamicMatrix<double> > &D,boost::scoped_ptr<double> &y,col_vector &Ax);
//typedef void (* fun_type2)(col_vector_D &for_gradient, double &c, ptr_bowl &param);
//typedef void (* fun_type3)(ptr &fx, col_vector &x, col_vector &theta,ptr_bowl &param,sym_matrix &Qx, col_vector &mu);
//typedef void (* fun_type4)(col_vector &theta, blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &x,sym_matrix &Qlike, ptr_bowl &param, col_vector &gradient,col_vector &Ax, bool G, bool H);
//typedef void (* fun_type5)(ptr &fx, col_vector &x,col_vector &theta, ptr_bowl &param);
//typedef void (* fun_type6)(ptr &y1, col_vector &x, col_vector &theta,blaze::CompressedMatrix<double> &A,col_vector &ysim,ptr_bowl &param,blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D,sym_matrix2 &Qlike);
typedef void (* fun_type7)(ptr &fx,col_vector &theta, col_vector &eta, ptr_bowl &param);
typedef void (* fun_type8)(fun_type7 fun, ptr &fx,col_vector &theta, col_vector &eta, ptr_bowl &param);
typedef void (* fun_type8)(fun_type7 fun, ptr &fx,col_vector &theta, col_vector &eta, ptr_bowl &param);
typedef void (* fun_type9)(ptr_bowl &param, col_vector &pts, int comp_case);
typedef void (* fun_type10)(fun_type9 fun, col_vector &theta, ptr_bowl &param);

typedef void (* fun_type_eta1)(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param);
typedef void (* fun_type_eta2)(fun_type9 recipe, ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param);




void ldmvnorm(ptr &fx, col_vector &x,col_vector &mu, sym_matrix &Q);
void like_Poisson_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param);
void like_Gaussian_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param);
void like_Binomial_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param);


void hessian_theta_given_y(fun_type7 fun, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central);
void trans_hessian_theta_given_y(fun_type7 fun1,fun_type8 fun2, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central);
void smart_hessian_theta_given_y(fun_type7 fun1,fun_type8 fun2, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central);

void hessian_p_y_given_x_theta(fun_type7 fun, col_vector &x, row_matrix &m, double stepsize, size_t xsize, ptr_bowl &param, col_vector &theta);
void optimize_p_theta_given_y(opt_fun2 fun, col_vector &thetastar,col_vector &etastar, ptr_bowl &param);



void loglikelihood_fun(ptr_bowl &param, size_t &ind, col_vector &eta,col_vector &loglike, col_vector &theta);

void update_invQx_by_theta(col_vector &theta, ptr_bowl &param);
void update_Qx_by_theta(col_vector &theta, ptr_bowl &param);

/*
void like_Poisson(ptr &fx, col_vector &x,col_vector &theta, ptr_bowl &param);
void like_Binomial(ptr &fx, col_vector &x,col_vector &theta, ptr_bowl &param);
void like_Gaussian(ptr &fx, col_vector &x,col_vector &theta, ptr_bowl &param);


void Qlike_Poisson(col_vector &theta, blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &x,sym_matrix &Qlike, ptr_bowl &param, col_vector &gradient,col_vector &Ax, bool G, bool H);
void Qlike_Binomial(col_vector &theta, blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &x,sym_matrix &Qlike, ptr_bowl &param, col_vector &gradient,col_vector &Ax, bool G, bool H);
void Qlike_Gaussian(col_vector &theta, blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &x,sym_matrix &Qlike, ptr_bowl &param, col_vector &gradient,col_vector &Ax, bool G, bool H);

void myModel_Poisson(ptr &y1, col_vector &x, col_vector &theta,blaze::CompressedMatrix<double> &A,col_vector &ysim,ptr_bowl &param,blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D,sym_matrix2 &Qlike);
void myModel_Binomial(ptr &y1, col_vector &x, col_vector &theta,blaze::CompressedMatrix<double> &A,col_vector &ysim,ptr_bowl &param,blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D,sym_matrix2 &Qlike);
//void gradient_theta_given_y(opt_fun1 fun, col_vector &x, VectorXd &gradx, double stepsize, size_t xsize, ptr_bowl &param);
*/




/*
class GLP_DisUtensils
{	
	
};
*/

#endif
