
#ifndef _GLPRecipes_
#define _GLPRecipes_


#include "../GLP-Libraries/GLP_libraries.h"
#include "../GLP-Functions/GLP_functions.h"
#include "../GLP-DisUtensils/GLP_DisUtensils.h"

void poisson_recipe(ptr_bowl &param, col_vector &pts, int comp_case);
void binomail_recipe(ptr_bowl &param, col_vector &pts, int comp_case);
void approx_NoNGaussian_RD_eta_Poisson(fun_type9 recipe, ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param);
void approx_NoNGaussian_RD_eta_Binomial(fun_type9 recipe, ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param);
void approx_Gaussian_RD_eta(fun_type9 recipe, ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param);
void correction_Gaussian(fun_type9 recipe, col_vector &theta, ptr_bowl &param); 
void correction_Non_Gaussian(fun_type9 recipe, col_vector &theta, ptr_bowl &param);
void stdev_corr(fun_type7 fun, row_matrix &COV, ptr_bowl &param, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, ptr &ldense_theta_star,col_vector &eigenvalues, row_matrix  &eigenvectors, col_vector &eta_star_k);
void stdev_corr_position(fun_type7 fun, ptr_bowl &param, col_vector &theta_star, ptr &ldense_theta_star,col_vector &eigenvalues, row_matrix  &eigenvectors, col_vector &eta_star_k, int &index, int &position, double &value);


/*


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef void (* opt_fun1)(ptr &fx, col_vector &theta,size_t xsize);
typedef double (* opt_fun2)(const VectorXd& x, VectorXd& grad,size_t xsize);
typedef double (* fun_type1)(ptr &fx, col_vector &theta, col_vector &xstar, sym_matrix &Qstar);


void ldmvnorm(ptr &fx, col_vector &x,col_vector &mu, sym_matrix &Q);

void like_Poisson(ptr &fx, col_vector &x,col_vector &theta,blaze::CompressedMatrix<double> &A,col_vector &ysim);
void like_Binomial(ptr &fx, col_vector &x,col_vector &theta,blaze::CompressedMatrix<double> &A,col_vector &ysim, size_t bin_size);

void Qlike_Poisson(blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &x,sym_matrix &Qlike,blaze::CompressedMatrix<double> &A);
void Qlike_Binomial(blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> &D, col_vector &Ax, col_vector &x,sym_matrix &Qlike,blaze::CompressedMatrix<double> &A, size_t bin_size);

void gradient_theta_given_y(opt_fun1 fun, col_vector &x, VectorXd &gradx, double stepsize, size_t xsize);
void hessian_theta_given_y(opt_fun1 fun, col_vector &x, row_matrix &m, double stepsize, size_t xsize);


void optimize_p_theta_given_y(opt_fun2 fun, col_vector &thetastar, size_t xsize);

void stdev_corr(fun_type1 fun, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, ptr &ldense_theta_star,col_vector &eigenvalues, row_matrix  &eigenvectors, size_t xsize);
*/



/*
class GLP_Recipes
{	
	
};
*/

#endif
