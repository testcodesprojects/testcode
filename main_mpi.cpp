
#include <mpi.h>
#include <iostream>

#include "GLP/GLP-Libraries/GLP_libraries.h"
#include "GLP/GLP-Data/GLP_Data.h"
#include "GLP/GLP-Priors/GLP_Priors.h"
#include "GLP/GLP-DisUtensils/GLP_DisUtensils.h"
#include "GLP/GLP-Recipes/GLP_Recipes.h"
#include "GLP/GLP-Libraries/polynomials/polynomial.h"
#include "GLP/GLP-Libraries/polynomials/polynomials.h"

bool multipleSends = true;

/*
//Improvements:

(*parameters).optim->opt_satisfied = true if I need more accurate optimum




*/

/*
mpicxx -std=c++14 -O3 -DNDEBUG -mavx -pthread -fopenmp main.cpp GLP/GLP-Libraries/GLP_libraries.cpp GLP/GLP-Functions/GLP_functions.cpp GLP/GLP-Data/GLP_Data.cpp GLP/GLP-Libraries/GLP_splines.cpp GLP/GLP-DisUtensils/GLP_DisUtensils.cpp GLP/GLP-Recipes/GLP_Recipes.cpp -L/opt/intel/mkl/lib/intel64 -Wl,--start-group /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_gnu_thread.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl -o output -lstdc++  -lmpi -I./blaze -L./R/x86_64-pc-linux-gnu-library/4.1/BH/include/boost/mpi -lboost_serialization -lboost_mpi
*/
//ysim_size is changed to y_size
//for bym2 model update theta in libraries differently
// i chanaged Splines to splines (this when adding the posterior marginals)
///////////////////////////////////////////////////////////////---Given

//fix the fixes: search for !
//search for inla1234
//VARBAYES
//fixes

//To add models, add these:
fun_type_eta1 like_Model_eta;
fun_type_eta2 approx_NoNGaussian_RD_eta;
fun_type9 recipe;
fun_type10 correction;

void setmodel(ptr_bowl &param);
void check_Model(ptr_bowl &param);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 struct storage_type2
{
   boost::ptr_vector<double> mues;
   boost::ptr_vector<double> sds;
   boost::ptr_vector<tkp::polynomials> polynomials;
};


void p_theta(ptr &fx, col_vector &theta,ptr_bowl &param);
void p_y_given_eta_theta(ptr &fx, col_vector &eta,col_vector &theta,ptr_bowl &param);
void GA_p_eta_given_y_theta(ptr &y, col_vector &theta,col_vector &eta_star, ptr_bowl &param);

void opt_fun_p_theta_given_y(ptr &fx, col_vector &theta, col_vector &eta, ptr_bowl &param);
void topt_fun_p_theta_given_y(fun_type7 fun, ptr &fx, col_vector &phi, col_vector &etastar, ptr_bowl &param);
void gradient_theta_given_y(VectorXd &gradx, col_vector &theta, col_vector &etastar, ptr_bowl &param);
void trans_gradient_theta_given_y(VectorXd &gradx, col_vector &theta, col_vector &etastar, ptr_bowl &param);
double wrapper_p_theta_given_y(const VectorXd& th, VectorXd& grad, ptr_bowl &param, col_vector &etastar);

double p_thetaj(col_vector &theta,col_vector &theta_mode, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg);
void p_thetaj_given_y(col_vector &theta_star, row_matrix &COV, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg);
void p_thetaj_given_y_index(int &index, col_vector &theta_star, row_matrix &COV, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg);

void get_CDD_Design(size_t &theta_size, row_matrix &CCD_design_mat, col_vector &log_weights, ptr &fvalue);
void CCD_Stratetgy(col_vector &etastar, ptr_bowl &param, row_matrix &thetas,col_vector &weights, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3);
void CCD_Stratetgy_shelf1(col_vector &log_weights, row_matrix &all_CCD_points, size_t &theta_size, ptr_bowl &param, row_matrix &thetas, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3);
void CCD_Stratetgy_shelf2(int &task, double &value, row_matrix &all_CCD_points, ptr_bowl &param, row_matrix &thetas,col_vector &weights, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3);
void CCD_Stratetgy_shelf3(col_vector &log_weights, col_vector &weights);


//GA
void for_cooking_GA(col_vector &theta, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors);
void GA_xi_marginal_s(ptr_bowl &param, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage);
void GA_xi_marginal_pot1(col_matrix &a, col_matrix &b, col_matrix &c, size_t &xsize, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage, col_vector &weighted_mean, col_vector &weighted_stdev);
void GA_xi_marginal_pot2(bool &RTheta, int &ind, col_vector &a, col_vector &b, col_vector &c, col_vector &weighted_mean, col_vector &weighted_stdev, col_vector &weights);

void cooking_GA(int &task, double &value, col_vector &theta, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors);
void GA_is_cooked(int id_bowl, int &task, col_vector &weights, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors);

void cooking_table(col_vector &theta_star, col_vector &eta_star_star, ptr_bowl &param);
void compute_marginal_likelihood(ptr_bowl &param, col_vector &eigen_values, col_vector &weights, col_vector & stdev_corr_pos, col_vector & stdev_corr_neg);
void compute_DIC(ptr_bowl &param, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage, row_matrix &thetas,col_vector &eta_star,col_vector &theta_star);

void send_vector(int to_worker, col_vector &value, int tag){
   size_t n = value.size();
   double *mesg;
   mesg = new double[n];
   for(size_t i=0; i<n; i++) mesg[i] = value[i];
   MPI_Send(mesg,n,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD);
}

void isend_vector(int to_worker, col_vector &value, int tag){
   MPI_Request req;
   size_t n = value.size();
   double *mesg;
   mesg = new double[n];
   for(size_t i=0; i<n; i++) mesg[i] = value[i];
   MPI_Isend(mesg,n,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void isend_2vector(int to_worker, col_vector &value1, col_vector &value2, int tag){
   MPI_Request req;
   size_t n1 = value1.size(), n2 = value1.size() + value2.size();
   double *mesg;
   mesg = new double[n2];
   
   size_t j1 = 0, j2 = 0;
   for(size_t i=0; i<n1; i++) {mesg[i] = value1[j1];  j1++;}
   for(size_t i=n1; i<n2; i++) {mesg[i] = value2[j2]; j2++;}

   MPI_Isend(mesg,n2,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void isend_2vector_d(int to_worker, col_vector &value1, col_vector &value2, double &value, int tag){
   MPI_Request req;
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = n2 + 1;
   double *mesg;
   mesg = new double[n3];
   
   size_t j1 = 0, j2 = 0;
   for(size_t i=0; i<n1; i++) {mesg[i] = value1[j1];  j1++;}
   for(size_t i=n1; i<n2; i++) {mesg[i] = value2[j2]; j2++;}
   for(size_t i=n2; i<n3; i++) {mesg[i] = value;}

   MPI_Isend(mesg,n3,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void isend_3vector(int to_worker, col_vector &value1, col_vector &value2, col_vector &value3, int tag){
   MPI_Request req;
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = value1.size() + value2.size() + value3.size();
   double *mesg;
   mesg = new double[n3];
   
   size_t j1 = 0, j2 = 0, j3 = 0;
   for(size_t i=0; i<n1; i++) {mesg[i] = value1[j1];  j1++;}
   for(size_t i=n1; i<n2; i++) {mesg[i] = value2[j2]; j2++;}
   for(size_t i=n2; i<n3; i++) {mesg[i] = value3[j3]; j3++;}

   MPI_Isend(mesg,n3,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void isend_4vector(int to_worker, col_vector &value1, col_vector &value2, col_vector &value3, col_vector &value4, double &value, int tag){
   MPI_Request req;
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = value1.size() + value2.size() + value3.size(), n4 = value1.size() + value2.size() + value3.size() + value4.size(), n5 = n4+1;
   double *mesg;
   mesg = new double[n5];
   
   size_t j1 = 0, j2 = 0, j3 = 0, j4 = 0;
   for(size_t i=0; i<n1; i++)  {mesg[i] = value1[j1]; j1++;}
   for(size_t i=n1; i<n2; i++) {mesg[i] = value2[j2]; j2++;}
   for(size_t i=n2; i<n3; i++) {mesg[i] = value3[j3]; j3++;}
   for(size_t i=n3; i<n4; i++) {mesg[i] = value4[j4]; j4++;}
   for(size_t i=n4; i<n5; i++) mesg[i] = value;

   MPI_Isend(mesg,n5,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void receive_vector(int from_worker, col_vector &value, int tag){
   size_t n = value.size();
   double *mesg;
   mesg = new double[n];
   MPI_Recv(mesg,n,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
   for(size_t i=0; i<n; i++) value[i] = mesg[i];
}

void receive_2vector(int from_worker, col_vector &value1, col_vector &value2, int tag){
   
   size_t n1 = value1.size(), n2 = value1.size() + value2.size();
   double *mesg;
   mesg = new double[n2];

   MPI_Recv(mesg,n2,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

   size_t j1 = 0, j2 = 0;
   for(size_t i=0; i<n1; i++) {value1[j1] = mesg[i]; j1++;}
   for(size_t i=n1; i<n2; i++) {value2[j2] = mesg[i]; j2++;}
}

void receive_2vector_d(int from_worker, col_vector &value1, col_vector &value2, double &value, int tag){
   
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = n2 + 1;
   double *mesg;
   mesg = new double[n3];

   MPI_Recv(mesg,n3,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

   size_t j1 = 0, j2 = 0;
   for(size_t i=0; i<n1; i++) {value1[j1] = mesg[i]; j1++;}
   for(size_t i=n1; i<n2; i++) {value2[j2] = mesg[i]; j2++;}
   for(size_t i=n2; i<n3; i++) {value = mesg[i];}

}

void receive_3vector(int from_worker, col_vector &value1, col_vector &value2, col_vector &value3, int tag){
   
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = value1.size() + value2.size() + value3.size();
   double *mesg;
   mesg = new double[n3];

   MPI_Recv(mesg,n3,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

   size_t j1 = 0, j2 = 0, j3 = 0;
   for(size_t i=0; i<n1; i++) {value1[j1] = mesg[i]; j1++;}
   for(size_t i=n1; i<n2; i++) {value2[j2] = mesg[i]; j2++;}
   for(size_t i=n2; i<n3; i++) {value3[j3] = mesg[i]; j3++;}
}

void receive_4vector(int from_worker, col_vector &value1, col_vector &value2, col_vector &value3, col_vector &value4, double &value, int tag){
   
   size_t n1 = value1.size(), n2 = value1.size() + value2.size(), n3 = value1.size() + value2.size() + value3.size(), n4 = value1.size() + value2.size() + value3.size() + value4.size(), n5 = n4+1;
   double *mesg;
   mesg = new double[n5];

   MPI_Recv(mesg,n5,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

   size_t j1 = 0, j2 = 0, j3 = 0, j4 = 0;
   for(size_t i=0; i<n1; i++) {value1[j1] = mesg[i]; j1++;}
   for(size_t i=n1; i<n2; i++) {value2[j2] = mesg[i]; j2++;}
   for(size_t i=n2; i<n3; i++) {value3[j3] = mesg[i]; j3++;}
   for(size_t i=n3; i<n4; i++) {value4[j4] = mesg[i]; j4++;}
   for(size_t i=n4; i<n5; i++) value = mesg[i];
}

void isend_double(int to_worker, double &value, int tag){
   MPI_Request req;
   double *mesg; mesg = new double[1]; mesg[0] = value;
   MPI_Isend(mesg,1,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD,&req);
}

void send_double(int to_worker, double &value, int tag){
   double *mesg; mesg = new double[1]; mesg[0] = value;
   MPI_Send(mesg,1,MPI_DOUBLE,to_worker,tag,MPI_COMM_WORLD);
}

void receive_double(int from_worker, double &value, int tag){
   double *mesg; mesg = new double[1]; 
   MPI_Recv(mesg,1,MPI_DOUBLE,from_worker,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
   value = mesg[0];
}

void testfunction(ptr_bowl &param)
{
   col_vector theta(6,0);
   theta[0] = -1.2; theta[1] = -1.3; theta[2] = -1.4; theta[3] = -1.5; theta[4] = -1.6; theta[5] = -1.1;

   theta[0] = 0.03178631817; theta[1] = -4.312608567; theta[2] = 0.4968412996; theta[3] = 0.4284958044; theta[4] = -0.01739381674; theta[5] = 1.26023963;


   //col_vector eta_star = log((*param).y_response+0.0001);
   col_vector eta_star =(*param).y_response+0.0001;

   ptr y{new double{0.0}};
  // GA_p_eta_given_y_theta(y,theta,eta_star,param);

   opt_fun_p_theta_given_y(y,theta,eta_star,param);

   VectorXd grad = VectorXd::Ones(theta.size());

   gradient_theta_given_y(grad,theta,eta_star,param); 
   print(grad);
}  

void p_theta(ptr &fx, col_vector &theta,ptr_bowl &param){
  *fx = 0.0;

   if(!(*param).pcjoint){
      std::cout << "I should not be here!" << std::endl;

      if((*param).RTheta)
      {
         for(size_t i = 0; i < theta.size(); i++)
         {
            if((*param).pr[i].type=="pc"){(*fx) += (*param).pr[i].Sp->get_value(theta[i]); 
            } else  {(*fx) += (*param).pr[i].f(theta[i],(*param).pr[i].p1,(*param).pr[i].p2); } 
         }

      } 
      else *fx = 0.0;
   }else{
     // std::cout << "I am pc.joint" << std::endl;
      *fx =  pc_joint(theta);
   }
   
}

void p_y_given_eta_theta(ptr &fx, col_vector &eta,col_vector &theta,ptr_bowl &param){
   *fx = 0.0;
   like_Model_eta(fx,eta,theta,param);
}

void topt_fun_p_theta_given_y(fun_type7 fun, ptr &fx, col_vector &phi, col_vector &etastar, ptr_bowl &param)
{
   col_vector theta = (*param).optim->curr_x + (*param).optim->G*phi;
   fun(fx,theta,etastar,param);
}

void GA_p_eta_given_y_theta(ptr &y, col_vector &theta,col_vector &eta_star, ptr_bowl &param){

   if(isnan(theta[0]) || abs(theta[0])>1000000) {  
      //if((*param).internalverbose) cout << "" << std::endl;
      throw "<<seven pepper is added>>";
   }  
   if(!isfinite(theta)) { 
      throw "<<seven pepper is added>>";
   }  

   if((*param).Model!="Gaussian")
   {
      if((*param).RD_system){ 
         approx_NoNGaussian_RD_eta(recipe,y,eta_star,theta,param); //print("always here");
      } else{ }
   } else { 
      if((*param).RD_system){ 
         approx_Gaussian_RD_eta(recipe,y,eta_star,theta,param); //print("always here");
      } else{ }
   }

}

void opt_fun_p_theta_given_y(ptr &fx, col_vector &theta, col_vector &eta, ptr_bowl &param)
{ 
   if(!(*param).optim->hess_tick) print(trans(theta));
   ptr y{new double{0.0}}; *fx = 0.0;  
   GA_p_eta_given_y_theta(y,theta,eta,param); (*fx) -= (*y); //print("5"); print(-(*y));
   p_y_given_eta_theta(y,eta,theta,param); (*fx) -= (*y); //print("4"); print(-(*y));
   p_theta(y,theta,param); (*fx) -= (*y);  //print("7"); print(-(*y));

   //if((*param).optim->optfunctioncall==0) (*param).max_subtract = *fx;
   //*fx -= (*param).max_subtract;
   //print("8"); print((*fx));
   (*param).optim->optfunctioncall = (*param).optim->optfunctioncall + 1;

}

void gradient_theta_given_y(VectorXd &gradx, col_vector &theta, col_vector &etastar, ptr_bowl &param)
{
   boost::scoped_ptr<double> h{new double{(*param).optim->grad_stepsize}};
   size_t i; //col_vector xxx;

   (*param).optim->hess_tick = true;
   if((*param).optim->central)
   {
      for (i = 0; i < (theta.size()); i++)
      {
         col_vector xcopy = theta;
         boost::scoped_ptr<double> y1{new double{0.0}}, y2{new double{0.0}};
         xcopy[i] += (*h);
         opt_fun_p_theta_given_y(y1,xcopy,etastar,param);
         xcopy[i] -= 2*(*h);
         opt_fun_p_theta_given_y(y2,xcopy,etastar,param);
         gradx[i] = ((*y1) - (*y2)) / (2.0 * (*h));
      }
   } else{

      boost::scoped_ptr<double> y2{new double{(*param).optim->y}};
      for (i = 0; i < (theta.size()); i++)
      {
         col_vector xcopy = theta;
         boost::scoped_ptr<double> y1{new double{0.0}};
         xcopy[i] += (*h);
         opt_fun_p_theta_given_y(y1,xcopy,etastar,param);
         gradx[i] = ((*y1) - (*y2)) / (*h);
      }
   }

      (*param).optim->hess_tick = false;
      //print(gradx);
}

void trans_gradient_theta_given_y(VectorXd &gradx, col_vector &theta, col_vector &etastar, ptr_bowl &param)
{
  (*param).optim->smart_update(theta);

   boost::scoped_ptr<double> h{new double{(*param).optim->grad_stepsize}};
   size_t i;
   col_vector tran_grad(theta.size()), xcopy(theta.size());
   (*param).optim->hess_tick = true;
   if((*param).optim->central)
   {
      for (i = 0; i < (theta.size()); i++)
      {
         xcopy = 0.0;
         boost::scoped_ptr<double> y1{new double{0.0}}, y2{new double{0.0}};
         xcopy[i] += (*h);
         topt_fun_p_theta_given_y(opt_fun_p_theta_given_y,y1,xcopy,etastar,param);
         xcopy[i] -= 2*(*h);
         topt_fun_p_theta_given_y(opt_fun_p_theta_given_y,y2,xcopy,etastar,param);
         tran_grad[i] = ((*y1) - (*y2)) / (2.0 * (*h));
      }
   } else{
      boost::scoped_ptr<double> y2{new double{(*param).optim->y}};
      for (i = 0; i < (theta.size()); i++)
      {
         xcopy = 0.0;
         boost::scoped_ptr<double> y1{new double{0.0}};
         xcopy[i] += (*h);
         topt_fun_p_theta_given_y(opt_fun_p_theta_given_y,y1,xcopy,etastar,param);
         tran_grad[i] = ((*y1) - (*y2)) / (*h);
      }
   }

   //print(trans((*param).optim->G));
   tran_grad = solve(trans((*param).optim->G),tran_grad);
   for (i = 0; i < (theta.size()); i++) gradx[i] = tran_grad[i];

}

void gradient_i(double &gradx, col_vector &theta, col_vector &etastar, ptr_bowl &param, int i, bool left, bool right)
{
   boost::scoped_ptr<double> h{new double{(*param).optim->grad_stepsize}};
   (*param).optim->hess_tick = true;
   col_vector xcopy = theta;
   boost::scoped_ptr<double> y1{new double{0.0}};
   if(right) xcopy[i] += (*h);
   else if(left) xcopy[i] -= (*h);
   
   //std::cout << (*param).id_bowl << " sum(etastar): "<< sum(etastar) << std::endl;
   opt_fun_p_theta_given_y(y1,xcopy,etastar,param);
   gradx = (*y1);
   (*param).optim->hess_tick = false;
   //std::cout << (*param).id_bowl << " computed grad: "<< (*y1) << " at theta: " <<  xcopy[0] << "," << xcopy[1] << "," << xcopy[2] << "," << xcopy[3] << "," << xcopy[4] << std::endl;
}

void linesearch_i(double &value, col_vector &theta, col_vector &etastar, ptr_bowl &param)
{
   (*param).optim->hess_tick = true;
   boost::scoped_ptr<double> y1{new double{0.0}};
   opt_fun_p_theta_given_y(y1,theta,etastar,param);
   value = (*y1);
   (*param).optim->hess_tick = false;
   //std::cout << (*param).id_bowl << " computed grad: "<< (*y1) << " at theta: " <<  xcopy[0] << "," << xcopy[1] << "," << xcopy[2] << "," << xcopy[3] << "," << xcopy[4] << std::endl;
}

double wrapper_p_theta_given_y(const VectorXd& th, VectorXd& grad, ptr_bowl &param, col_vector &etastar)
{
   bool warning = false;
   if((*param).size_bowl>1){

      col_vector theta(th.size()), mpi_grad_right(th.size()),mpi_grad_left(th.size());
      for(size_t i =0; i<th.size(); i++) theta[i] = th[i];
      std::cout << "theta: ";
      print(trans(theta));

      ptr fx{new double{0.0}};  
      int id_process = 0;

      if(multipleSends){
         int tag1 = 123, tag2 =999;
         double *contin; contin = new double[1]; contin[0] = 1;

         for(int id_worker=1; id_worker<(*param).grad_workers; id_worker++){
            MPI_Request req1; MPI_Isend(contin,1,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req1);
            isend_2vector(id_worker, theta,(*param).update->eta_moon,tag1);}

         for(int id_worker=1; id_worker<(*param).grad_workers; id_worker++){
            int fix_tag = 621;
            double *main_value;
            main_value = new double[1];
            MPI_Recv(main_value,1,MPI_DOUBLE,id_worker,fix_tag,MPI_COMM_WORLD,MPI_STATUSES_IGNORE); 
         }

      }else{
         int tag1 = 123, tag2 =999, tag3 = 321;
         double *mesg1, *mesg2, *contin;
         mesg1 = new double[(*param).theta_size]; mesg2 = new double[(*param).y_size]; contin = new double[1]; contin[0] = 1;
         
         for(size_t i=0; i<(*param).theta_size; i++)
            mesg1[i] = theta[i];
      
         for(size_t i=0; i<(*param).y_size; i++)
            mesg2[i] = (*param).update->eta_moon[i];

         for(int id_worker=1; id_worker<(*param).grad_workers; id_worker++){
            MPI_Request req1; MPI_Isend(contin,1,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req1);
            MPI_Request req2; MPI_Isend(mesg1,(*param).theta_size,MPI_DOUBLE,id_worker,tag1,MPI_COMM_WORLD,&req2);
            MPI_Request req3; MPI_Isend(mesg2,(*param).y_size,MPI_DOUBLE,id_worker,tag3,MPI_COMM_WORLD,&req3);}
      }
  
      (*param).optim->hess_tick = true;
      opt_fun_p_theta_given_y(fx,theta,etastar,param);
      if(!((*param).optim->central)) (*param).optim->y = *fx;

      //gradient_theta_given_y(grad,theta,etastar,param); 
      //print(grad);

      for(int id_bowl = 0; id_bowl <(*param).grad_workers; id_bowl++){
         boost::ptr_vector<int> tasks;
         seasoning((*param).num_tasks_grad, (*param).grad_workers, id_bowl, true,tasks);
         int *ptr_tasks;
         ptr_tasks = new int[tasks.size()];

         size_t i=0;
         for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
            
         for (size_t i=0;i < tasks.size(); i++){
            bool left, right; int task = ptr_tasks[i];

            if(id_bowl==0){
               
               int index_theta = gradient_majic_spread((*param).optim->central,task,left,right,(*param).theta_size);
               if(index_theta!=-999){
                  double gradx = 0;
                  gradient_i(gradx,theta,etastar, param, index_theta,left, right);
                  if(right) mpi_grad_right[index_theta] = gradx;
                  else if(left) mpi_grad_left[index_theta] = gradx;
               }
            }else{
               int tag3 = 3;
               double *value;
               value = new double[1];

               MPI_Recv(value,1,MPI_DOUBLE,id_bowl,task,MPI_COMM_WORLD,MPI_STATUSES_IGNORE); //3 is id of bowl

               if(value[0]==10000000) {warning = true; print("warning!!!!"); }
               int index_theta = gradient_majic_spread((*param).optim->central,task,left,right,(*param).theta_size);

               //std::cout << index_theta << " || " << task << std::endl;
               if(right) mpi_grad_right[index_theta] = value[0];
               else if(left) mpi_grad_left[index_theta] = value[0];
            }
            
         }
      }
      
   
      if((*param).optim->central){

         mpi_grad_right -=  mpi_grad_left;
         mpi_grad_right = mpi_grad_right/(2*(*param).optim->grad_stepsize);


      }else{
      	//for(size_t i =0; i<5; i++)
      	//	std::cout << "I am " << (*param).id_bowl << " and getting this: grad_i: " << mpi_grad_right[i] << " - fx: " << (*fx) << " - h: " << (*param).optim->grad_stepsize << std::endl;
        
        mpi_grad_right = (mpi_grad_right - (*fx))/(*param).optim->grad_stepsize;

      }

      for(size_t i=0; i<(*param).theta_size;i++) grad[i] = mpi_grad_right[i];

      if((*param).optim->peppers>0) for(size_t i=0; i<(*param).theta_size;i++) if(std::abs((*param).optim->save_grad[i] - grad[i])>100) warning = true;

      if(warning) {
         //print((*param).optim->peppers);
         throw (*param).optim->peppers;
      }

      //print(grad);

      //std::cout << "gnorm: " << l1Norm(gg) << "theta: " << l1Norm(theta - (*param).optim->theta_old) << " f(x): " << (*param).optim->f_old - (*fx) << std::endl;
      //if(l1Norm(theta - (*param).optim->theta_old)<1e-3 && ((*param).optim->f_old - (*fx))<1e-2) (*param).optim->central = true;

      if((*param).optim->f_old==0) (*param).optim->f_old = (*fx);
      else{
        if((*param).optim->f_old>(*fx)){
           (*param).optim->f_old = (*fx);
           (*param).optim->theta_old = theta;
        }
      }
      
      //if((*param).internalverbose) print(grad);
      for(size_t i =0; i< (*param).theta_size ; i++) (*param).optim->save_grad[i] = grad[i];

      return (*fx);
   //----------------------------------------------------------------------------------------------------
   }else{
      col_vector theta(th.size());
      for(size_t i =0; i<th.size(); i++) theta[i] = th[i];

      ptr fx{new double{0.0}};
      opt_fun_p_theta_given_y(fx,theta,etastar,param);
      if(!((*param).optim->central)) (*param).optim->y = *fx;

      if((*param).optim->smartGrad && theta.size()>1) trans_gradient_theta_given_y(grad,theta,etastar,param); 
      else gradient_theta_given_y(grad,theta,etastar,param); 
      

      //col_vector gg(5);
      //for(size_t i=0; i<5;i++) gg[i] = grad[i];
      //std::cout << "gnorm: " << l1Norm(gg) << "theta: " << l1Norm(theta - (*param).optim->theta_old) << " f(x): " << (*param).optim->f_old - (*fx) << std::endl;
   
      if(l1Norm(theta - (*param).optim->theta_old)<1e-3 && ((*param).optim->f_old - (*fx))<1e-2) (*param).optim->central = true;
      (*param).optim->f_old = (*fx);
      (*param).optim->theta_old = theta;  
      
      return (*fx);
   }
   
}

void hessian_ij(fun_type7 fun, col_vector &etastar, ptr_bowl &param, ptr &y, int &index_i, int & step_i, int & index_j, int & step_j)
{
   (*param).optim->smartGrad = false;
   boost::scoped_ptr<double> h{new double{sqrt((*param).optim->grad_stepsize)}};
   col_vector thetacopy = (*param).theta_star;

   thetacopy[index_i] += (step_i)*(*h);
   if(index_j>=0) thetacopy[index_j] += (step_j)*(*h); 
   fun(y,thetacopy,etastar,param);
   thetacopy[index_i] -= (step_i)*(*h);
   if(index_j>=0) thetacopy[index_j] -= (step_j)*(*h);  
}

void chef_cooking_table(ptr_bowl &param, col_vector &eigenvalues, row_matrix &eigenvectors, row_matrix &hessian){
   
   bool burnt = false, go = false;
   if((*param).go_to_oven) burnt = true;
   size_t n = (*param).theta_size;
   (*param).optim->hess_tick = true;
   if(!(*param).go_to_oven) drawer1(param);
   else drawer2(param);

   if((*param).id_bowl < (*param).hess_workers){
   if((*param).id_bowl==0){

      (*param).update->eta_moon = (*param).update->update_eta_star;
      double fx=0.0;
      (*param).optim->get_ldens_theta_star_equal(fx);

      //(*param).eta_star.resize((*param).y_size);
      //(*param).eta_star = (*param).update->eta_moon;
      
      if(multipleSends){
         int tag = 111; 
         for(int id_worker=1; id_worker<(*param).hess_workers; id_worker++)
            isend_2vector_d(id_worker,(*param).theta_star, (*param).update->eta_moon, fx, tag);

      }else{
         int tag0 = 111, tag1 = 123, tag2 = 321;
         double *mesg0, *mesg1, *mesg2; mesg0 = new double[1]; mesg0[0] = fx;
         mesg1 = new double[(*param).theta_size]; mesg2 = new double[(*param).y_size];
      
         for(size_t i=0; i<(*param).theta_size; i++)
            mesg1[i] = (*param).theta_star[i];
         
         for(size_t i=0; i<(*param).y_size; i++)
            mesg2[i] = (*param).update->eta_moon[i];

         for(int id_worker=1; id_worker<(*param).hess_workers; id_worker++){
         MPI_Request req1; MPI_Isend(mesg0,1,MPI_DOUBLE,id_worker,tag0,MPI_COMM_WORLD,&req1);
         MPI_Request req2; MPI_Isend(mesg1,(*param).theta_size,MPI_DOUBLE,id_worker,tag1,MPI_COMM_WORLD,&req2);
         MPI_Request req3; MPI_Isend(mesg2,(*param).y_size,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req3);
         }
      }
      
   }else{

      (*param).theta_star.resize((*param).theta_size);

      if(multipleSends){
         int tag = 111; double fx = 0;
         receive_2vector_d(0,(*param).theta_star,(*param).update->eta_moon,fx,tag);
         (*param).optim->set_ldens_theta_star_equal(fx);

      }else{
         int tag0 = 111, tag1 = 123, tag2 = 321;
         double *mesg0, *mesg1,*mesg2; mesg0 = new double[1];
         mesg1 = new double[(*param).theta_size]; mesg2 = new double[(*param).y_size]; 

         MPI_Recv(mesg0,1,MPI_DOUBLE,0,tag0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
         MPI_Recv(mesg1,(*param).theta_size,MPI_DOUBLE,0,tag1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         MPI_Recv(mesg2,(*param).y_size,MPI_DOUBLE,0,tag2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

         double fx = mesg0[0];
         (*param).optim->set_ldens_theta_star_equal(fx);

         for(size_t i=0; i<(*param).theta_size; i++){
            (*param).theta_star[i] = mesg1[i];
         }

         for(size_t i=0; i<(*param).y_size; i++){
            (*param).update->eta_moon[i] = mesg2[i];
         }
      }
      
   }

   if((*param).id_bowl==0){

      hessian.resize((*param).theta_size,(*param).theta_size);
      if(!(*param).go_to_oven){
         (*param).save_hihj.resize((*param).theta_size,(*param).theta_size);
         double fx=0.0;
         (*param).optim->get_ldens_theta_star_equal(fx);
         hessian = -fx;
         band(hessian,0) = +2*fx;
      }else{
         hessian = (*param).save_hihj;
      }

      col_vector eta_star((*param).y_size);
      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_hess, (*param).hess_workers, (*param).id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];

      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
      
      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i]; bool left, right; 
         int index_i = -1, step_i = -1, index_j = -1, step_j = -1;

         ptr y{new double{0.0}};
         hessian_majic_spread(task, index_i, step_i, index_j, step_j,(*param).theta_size,(*param).go_to_oven);
         hessian_ij(opt_fun_p_theta_given_y, eta_star, param, y, index_i,step_i, index_j,step_j);

         double *value; value = new double[1]; value[0] = *y;
         task_position_connection(task, hessian, value,n,(*param).go_to_oven,(*param).save_hihj);
      }
      
   } else{

      col_vector eta_star((*param).y_size);
      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_hess, (*param).hess_workers, (*param).id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];

      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
      
      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i]; bool left, right; 
         int index_i = -1, step_i = -1, index_j = -1, step_j = -1;

         ptr y{new double{0.0}};
         hessian_majic_spread(task, index_i, step_i, index_j, step_j,(*param).theta_size,(*param).go_to_oven);
         hessian_ij(opt_fun_p_theta_given_y, eta_star, param, y, index_i,step_i, index_j,step_j);
         double *value; value = new double[1]; value[0] = *y;
         MPI_Request req; MPI_Isend(value,1,MPI_DOUBLE,0,task,MPI_COMM_WORLD,&req);
      }
   }
   
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if((*param).id_bowl==0){
      for(int id_bowl = 1; id_bowl <(*param).hess_workers; id_bowl++){
      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_hess, (*param).hess_workers, id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];

      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
         
      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i];

         double *value;
         value = new double[1];
         //std::cout << "id bowl: " << (*param).id_bowl << " and task: " << task << std::endl;
         MPI_Recv(value,1,MPI_DOUBLE,id_bowl,task,MPI_COMM_WORLD,MPI_STATUSES_IGNORE); 
         task_position_connection(task, hessian, value, n,(*param).go_to_oven,(*param).save_hihj);
         }}

      for(size_t ii=0; ii< n; ii++)
      {
         if(!(*param).go_to_oven){
            for(size_t jj = ii+1; jj < (*param).theta_size; jj++) {hessian(ii,jj) = hessian(ii,jj)/((*param).optim->grad_stepsize); hessian(jj,ii) = hessian(ii,jj);}
            hessian(ii,ii) = hessian(ii,ii)/(4*(*param).optim->grad_stepsize);
         }else{
            for(size_t jj = ii+1; jj < (*param).theta_size; jj++)  {hessian(ii,jj) = hessian(ii,jj)/(4*(*param).optim->grad_stepsize); hessian(jj,ii) = hessian(ii,jj);}

         }
        
      }

      if(!(*param).go_to_oven) band((*param).save_hihj,0) = band(hessian,0);

      bool last_option = false;
      myeign(hessian,eigenvalues,eigenvectors); 
      for(size_t p=0;p<eigenvalues.size();p++){
         if(eigenvalues[p]<0 && !(*param).go_to_oven) (*param).go_to_oven = true;
         else if(eigenvalues[p]<0 && (*param).go_to_oven){
            std::cout << "Central Difference Method for hessian did not work" << std::endl;
            col_vector keep = band(hessian,0);
            hessian = 0;
            band(hessian,0) = keep;
            (*param).go_to_oven = false;
         }
      }

      if(burnt) (*param).go_to_oven = false;

      if((*param).go_to_oven){

         int tag = 989;
         double *mesg; mesg = new double[1]; mesg[0] = 11111;
         for(int id_worker=1; id_worker<(*param).size_bowl; id_worker++)
            {MPI_Request req; MPI_Isend(mesg,1,MPI_DOUBLE,id_worker,tag,MPI_COMM_WORLD, &req); }
         
      }else if(!(*param).go_to_oven || burnt){
         
         int tag = 989;
         double *mesg; mesg = new double[1]; mesg[0] = 10001;
         for(int id_worker=1; id_worker<(*param).size_bowl; id_worker++)
            {MPI_Request req; MPI_Isend(mesg,1,MPI_DOUBLE,id_worker,tag,MPI_COMM_WORLD, &req); }

         double fx=0.0;
         (*param).optim->get_ldens_theta_star_equal(fx);
         
         int m = ((*param).theta_size*(*param).theta_size + (*param).theta_size)/2;
         int tag0 = 111, tag1 = 123, tag2 = 321, tag3 = 222;
         double *mesg0, *mesg1, *mesg2, *mesg3; 
         mesg0 = new double[1]; mesg0[0] = fx;
         mesg1 = new double[(*param).theta_size]; mesg2 = new double[(*param).y_size]; mesg3 = new double[m];
      
         for(size_t i=0; i<(*param).theta_size; i++)
            mesg1[i] = (*param).theta_star[i];
         
         for(size_t i=0; i<(*param).y_size; i++)
            mesg2[i] = (*param).update->eta_moon[i];

         size_t k = 0;
         for(size_t ii=0; ii< n; ii++)
            for(size_t jj = ii; jj < n; jj++) {mesg3[k] = hessian(ii,jj); k++;}
            
         for(int id_worker=1; id_worker<(*param).size_bowl; id_worker++){

            MPI_Request req1; MPI_Isend(mesg0,1,MPI_DOUBLE,id_worker,tag0,MPI_COMM_WORLD,&req1);
            MPI_Request req2; MPI_Isend(mesg1,(*param).theta_size,MPI_DOUBLE,id_worker,tag1,MPI_COMM_WORLD,&req2);
            MPI_Request req3; MPI_Isend(mesg2,(*param).y_size,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req3);
            MPI_Request req4; MPI_Isend(mesg3,m,MPI_DOUBLE,id_worker,tag3,MPI_COMM_WORLD,&req4);
         }
         go = true;
      
      }

   }else{
      int tag = 989;
      double *mesg; mesg = new double[1];
      MPI_Recv(mesg,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
      hessian.resize((*param).theta_size,(*param).theta_size);

      if(mesg[0]==11111) (*param).go_to_oven = true;
      else (*param).go_to_oven = false;

      if(!(*param).go_to_oven){
         
         int m = ((*param).theta_size*(*param).theta_size + (*param).theta_size)/2;
         int tag0 = 111, tag1 = 123, tag2 = 321, tag3 = 222;
         double *mesg0, *mesg1, *mesg2, *mesg3; 
         mesg0 = new double[1]; mesg1 = new double[(*param).theta_size]; 
         mesg2 = new double[(*param).y_size]; mesg3 = new double[m];
      
         MPI_Recv(mesg0,1,MPI_DOUBLE,0,tag0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
         MPI_Recv(mesg1,(*param).theta_size,MPI_DOUBLE,0,tag1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         MPI_Recv(mesg2,(*param).y_size,MPI_DOUBLE,0,tag2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         MPI_Recv(mesg3,m,MPI_DOUBLE,0,tag3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

         double fx = mesg0[0];
         (*param).optim->set_ldens_theta_star_equal(fx);
      
         for(size_t i=0; i<(*param).theta_size; i++){
            (*param).theta_star[i] = mesg1[i];
         }

         for(size_t i=0; i<(*param).y_size; i++){
            (*param).update->eta_moon[i] = mesg2[i];
         }

         size_t k = 0;
         for(size_t ii=0; ii< (*param).theta_size; ii++)
            for(size_t jj = ii; jj < (*param).theta_size; jj++) {hessian(ii,jj) = mesg3[k]; hessian(jj,ii) = mesg3[k]; k++;}
         
         go = true;
         //std::cout << (*param).id_bowl << std::endl;
         //print(hessian);
      

      } 
   }

   MPI_Barrier(MPI_COMM_WORLD);
   if(go){

      myeign(hessian,eigenvalues,eigenvectors); 

      if((*param).id_bowl==0){
         std::cout <<  std::endl;
         std::cout << "Hessian: " << std::endl;
         std::cout << hessian << std::endl;

         std::cout << "Eigenvalues: " << std::endl;
         std::cout << eigenvalues << std::endl;
         std::cout << "Eigenvectors: " << std::endl;
         std::cout << eigenvectors << std::endl;
      }
   }
}

void oven(ptr_bowl &param, col_vector &eigenvalues, row_matrix &eigenvectors, row_matrix &hessian){

   col_vector stdev_corr_pos((*param).theta_size,0.0),stdev_corr_neg((*param).theta_size,0.0);
   row_matrix Sigma = inv(hessian);
   ptr ldens_theta_star{new double{0.0}};
   (*param).optim->get_ldens_theta_star_equal(*ldens_theta_star); 

   increase_temp(param);

   if((*param).id_bowl < (*param).std_workers){

      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_std, (*param).std_workers, (*param).id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];
      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i], position = -1; double value = 0.0;
         int theta_index = oven_majic_temperature(task,position,(*param).theta_size);
         if(theta_index< (*param).theta_size) stdev_corr_position(opt_fun_p_theta_given_y,param, (*param).theta_star,ldens_theta_star,eigenvalues,eigenvectors,(*param).update->eta_moon, theta_index, position, value);
         
         if((*param).id_bowl>0) send_double(0,value,task);
         else oven_gloves(task,value,(*param).theta_size, stdev_corr_pos,stdev_corr_neg);
      }

      if((*param).id_bowl==0){

         for(size_t id = 1; id < (*param).std_workers; id++){
            boost::ptr_vector<int> tasks;
            seasoning((*param).num_tasks_std, (*param).std_workers, id, true,tasks);
            int *ptr_tasks;
            ptr_tasks = new int[tasks.size()];
            size_t i=0;
            for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

            for (size_t i=0;i < tasks.size(); i++){
               int task = ptr_tasks[i], position = -1; double value = 0.0;
               receive_double(id,value,task);
               oven_gloves(task,value,(*param).theta_size, stdev_corr_pos,stdev_corr_neg);
            }
         }
         

         for(size_t id = 1; id < (*param).size_bowl; id++){
            send_vector(id,stdev_corr_pos, 123);
            send_vector(id,stdev_corr_neg, 321);
         }

         std::cout << "corrections: " << std::endl;
         for(size_t i =0; i< (*param).theta_size;i++)
            std::cout <<"negative: " << stdev_corr_neg[i] << ", positive: " << stdev_corr_pos[i] << std::endl;
      }
   }

   if((*param).id_bowl>0){
      receive_vector(0, stdev_corr_pos, 123);
      receive_vector(0, stdev_corr_neg, 321);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   decrease_temp(param);

   if((*param).id_bowl < (*param).margtheta_workers){
      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_margtheta, (*param).margtheta_workers, (*param).id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];
      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i];
         int theta_index = check_oven(task,(*param).theta_size);
         p_thetaj_given_y_index(theta_index,(*param).theta_star,Sigma,eigenvalues, eigenvectors,stdev_corr_pos,stdev_corr_neg);
      }

   }

   MPI_Barrier(MPI_COMM_WORLD);

   /*
   if((*param).id_bowl==0){
      for(int id=1 ;id<(*param).size_bowl; id++)
         isend_vector(id,(*param).eta_star, 1010);
   }else{
      (*param).eta_star.resize((*param).y_size);
      receive_vector(0,(*param).eta_star,1010);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   */

   (*param).set_utensils_x();

   if((*param).Model=="Gaussian") {(*param).correction->Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
   }else if((*param).Model=="Poisson") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
   }else if((*param).Model=="Binomial") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);}

   boost::scoped_ptr<storage_type1> storage1{new storage_type1{(*param).x_size,(*param).theta_size}};
   boost::scoped_array<storage_type2> storage2{new storage_type2[(*param).x_size]}; 
   boost::scoped_ptr<storage_type3> storage3{new storage_type3{(*param).x_size,(*param).theta_size}};


   col_vector weights,log_weights; row_matrix thetas, all_CCD_points; 
   if((*param).theta_size==1) {}//{Grid_Stratetgy_1D(param,theta_star,x_star,ldens_theta_star,eigenvalues,eigenvectors,weights,thetas);}//Grid_Stratetgy_1D(theta_star,eigenvalues,eigenvectors,weights,thetas,x_size,Qx,Qstar,xstar,mu,grad_x_like,b_vec,x_initial,D,y);
   else if((*param).theta_size==2) {} //Grid_Stratetgy_2D(theta_star,eigenvalues,eigenvectors,weights,thetas,x_size,Qx,Qstar,xstar,mu,grad_x_like,b_vec,x_initial,D,y);
   else if((*param).theta_size>=3) {CCD_Stratetgy_shelf1(log_weights, all_CCD_points, (*param).theta_size, param, thetas,stdev_corr_pos,stdev_corr_neg,eigenvalues,eigenvectors,storage1,storage2, storage3);}


   monitor_food_oven(param,all_CCD_points.rows());
   if((*param).id_bowl < (*param).ccd_workers){

      weights.resize(all_CCD_points.rows());
      boost::ptr_vector<int> tasks;
      seasoning((*param).num_tasks_ccd, (*param).ccd_workers, (*param).id_bowl, true,tasks);
      int *ptr_tasks;
      ptr_tasks = new int[tasks.size()];
      size_t i=0;
      for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

      for (size_t i=0;i < tasks.size(); i++){
         int task = ptr_tasks[i]; double value = 0.0;
         CCD_Stratetgy_shelf2(task,value, all_CCD_points, param, thetas,weights,stdev_corr_pos,stdev_corr_neg,eigenvalues,eigenvectors,storage1,storage2,storage3);         
      }
      
      
      if((*param).id_bowl==0){
         
         size_t work_by_task = (*param).num_tasks_ccd/(*param).ccd_workers+1;
         int_row_matrix summary_tasks((*param).ccd_workers-1,work_by_task,-1);
         
         for(size_t id = 1; id < (*param).ccd_workers; id++){
            boost::ptr_vector<int> tasks;
            seasoning((*param).num_tasks_ccd, (*param).ccd_workers, id, true,tasks);
            int *ptr_tasks;
            ptr_tasks = new int[tasks.size()];
            size_t i=0;
            for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

            for (size_t i=0;i < tasks.size(); i++){
               int task = ptr_tasks[i]; double value = 0.0;
               summary_tasks(id-1,i) = task;
            }
         }
         
         for(size_t j=0; j<summary_tasks.columns();j++){
            for(size_t i=0; i<summary_tasks.rows();i++){
            int task = summary_tasks(i,j);
            int id = i +1;
            //if(task>=0) std::cout << "same: " << task << " -- " << id << std::endl;
            if(task>=0) GA_is_cooked(id,task,weights,param, storage1);
            }
         }

         print("");
         print("Cooked Successfully");

         /*
         for(size_t id = 1; id < (*param).ccd_workers; id++){
            boost::ptr_vector<int> tasks;
            seasoning((*param).num_tasks_ccd, (*param).ccd_workers, id, true,tasks);
            int *ptr_tasks;
            ptr_tasks = new int[tasks.size()];
            size_t i=0;
            for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

            for (size_t i=0;i < tasks.size(); i++){
               int task = ptr_tasks[i]; double value = 0.0;
               //std::cout << "task: " << task << " -- " << id << std::endl;
               GA_is_cooked(id,task,weights,param, storage1);
            }
         }*/
         
            
      }

   

   

   size_t xsize = (*param).x_size;
   col_vector weighted_mean(xsize,0), weighted_stdev(xsize,0);

   bool test = true;
   turnoff_oven(param,(*param).x_size);



   if((*param).id_bowl==0){
         //esma
      //std::cout << "weights: " << sum(weights) << std::endl;
      //print(weights); //esma

      CCD_Stratetgy_shelf3(log_weights,weights);
      
      if(!test){
         if((*param).GApp) GA_xi_marginal_s(param,weights,storage1);
         //else if((*param).SLApp) SLA_xi_marginal((*param).RTheta,weights,storage1);
         // else if(LApp) LA_xi_marginal(weights,storage2,storage3,x_size,(*param).theta_size);
      }else{
         col_matrix A(xsize,weights.size()),B(xsize,weights.size()),C(xsize,weights.size());
         if((*param).GApp) GA_xi_marginal_pot1(A,B,C,xsize,weights,storage1,weighted_mean,weighted_stdev);
         //else if((*param).SLApp) SLA_xi_marginal((*param).RTheta,weights,storage1);
         // else if(LApp) LA_xi_marginal(weights,storage2,storage3,x_size,(*param).theta_size);

         export_vector(weighted_mean,"x_star.txt");


         for(int id=1 ;id<(*param).size_bowl; id++){

            if(multipleSends){
               isend_3vector(id, weighted_mean,weighted_stdev,weights,500);
            }else{
               isend_vector(id,weighted_mean, 500);
               isend_vector(id,weighted_stdev, 501);
               isend_vector(id,weights, 502);
            }
            

         }
     

         for(size_t id = 1; id < (*param).size_bowl; id++){
            
            boost::ptr_vector<int> tasks;
            seasoning_margx((*param).num_tasks_margx, (*param).margx_workers, id, false,tasks);

            //for (auto task = tasks.begin(); task != tasks.end(); task++) {
            //   std::cout << "id: " << id << "---> " << *task << std::endl;}   

            int *ptr_tasks;
            ptr_tasks = new int[tasks.size()];
            size_t i=0;
            for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

            for (size_t i=0;i < tasks.size(); i++){
               int task = ptr_tasks[i], position = -1; double value = 0.0;
               int tag = task;
               
               col_vector a(weights.size(),0.0), b(weights.size(),0.0), c(weights.size(),0.0);

               for(size_t i=0;i<weights.size(); i++) a[i] = A(task,i);
               for(size_t i=0;i<weights.size(); i++) b[i] = B(task,i);
               for(size_t i=0;i<weights.size(); i++) c[i] = C(task,i);

               double label = (double)task;
               isend_double(id,label,700); 

               if(multipleSends){
                  tag += 1;
                  isend_3vector(id,a,b,c,tag);
               }else{
                  tag += 1; isend_vector(id,a, tag);
                  tag += 1; isend_vector(id,b, tag);
                  tag += 1; isend_vector(id,c, tag);
               }

               

            }
            double label = -99;
            //print(label);
            isend_double(id,label,700); 
         }
      }
      
      compute_marginal_likelihood(param, eigenvalues, weights,stdev_corr_pos,stdev_corr_neg);


   }else{

      if(test){

         if(multipleSends){
            receive_3vector(0,weighted_mean,weighted_stdev,weights,500);
         }else{
            receive_vector(0,weighted_mean,500);
            receive_vector(0,weighted_stdev,501);
            receive_vector(0,weights,502);
         }
         

         col_vector a(weights.size(),0.0), b(weights.size(),0.0), c(weights.size(),0.0);

         while(true){

            double label = 0.0; int index = 0; int tag =0;
            receive_double(0,label, 700);
            if(label<0) break;
            else index = (int)label;
            tag = index;

            if(multipleSends){
               tag += 1;
               receive_3vector(0,a,b,c,tag);
            }else{
               tag += 1; receive_vector(0,a,tag);
               tag += 1; receive_vector(0,b,tag);
               tag += 1; receive_vector(0,c,tag);
            }

            //double start_omp1,end_omp1; 
            //start_omp1 = omp_get_wtime(); 
            GA_xi_marginal_pot2((*param).RTheta,index,a,b,c,weighted_mean,weighted_stdev,weights);

            //end_omp1 = omp_get_wtime(); 
            //std::cout << end_omp1 - start_omp1  << '\n';


         }
      }
   }

   }//ccd workers




   /*
   for(size_t id = 1; id < (*param).std_workers; id++){
   boost::ptr_vector<int> tasks;
   seasoning((*param).num_tasks_std, (*param).std_workers, id, true,tasks);
   int *ptr_tasks;
   ptr_tasks = new int[tasks.size()];
   size_t i=0;
   for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}

   for (size_t i=0;i < tasks.size(); i++){
      int task = ptr_tasks[i], position = -1; double value = 0.0;
      receive_double(id,value,task);
      oven_gloves(task,value,(*param).theta_size, stdev_corr_pos,stdev_corr_neg);
   }
}



   */



   if((*param).id_bowl==0){
      compute_DIC(param,weights,storage1,thetas,(*param).update->eta_moon,(*param).theta_star);
   }else{
      //std::cout << "I am not working: " << (*param).id_bowl << std::endl;
   }
   
}


int main(int argc, char *argv[])
{
   double start_omp,mid_omp,end_omp; 

   int id_bowl, size_bowl, provided;
   MPI_Status status;
   //MPI_Init(&argc, &argv);
   MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE, &provided);
   if(provided < MPI_THREAD_MULTIPLE) print("The threading support level is lesser than that demanded.\n");

   MPI_Comm_size(MPI_COMM_WORLD, &size_bowl);
   MPI_Comm_rank(MPI_COMM_WORLD, &id_bowl);

   start_omp = omp_get_wtime(); 
   ptr_bowl param{new Bowl{}};
   (*param).id_bowl = id_bowl;
   (*param).size_bowl = size_bowl;

   if(size_bowl > 1) {
      
      //boost::scoped_ptr<Bowl<tkp::splines>> param{new Bowl<tkp::splines>{}};

      getdata(param); 
      set_priors(param);
      setmodel(param);
      (*param).pour_the_Bowl();
      check_Model(param);

      //blaze::setNumThreads((*param).optim->num_threads); //export BLAZE_NUM_THREADS=6
      //std::cout << "3. Number of threads: " << (*param).optim->num_threads << std::endl;
      cupboard(param);
      if((*param).RTheta)
      {
         if(id_bowl==0){

            col_vector theta_star((*param).theta_size,0.0),eta_star((*param).y_size);  
            optimize_p_theta_given_y(wrapper_p_theta_given_y, theta_star,eta_star,param);

            int tag2 =999;;
            double *contin;
            contin = new double[1]; contin[0] = 0;

            for(int id_worker=1; id_worker<(*param).grad_workers; id_worker++)
               {MPI_Request req; MPI_Isend(contin,1,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req);}

            (*param).theta_star.resize((*param).theta_size);
            (*param).theta_star = theta_star;

            //linesearch
           // for(int id_worker=(*param).grad_workers; id_worker<(*param).size_bowl; id_worker++)
           //    {MPI_Request req; MPI_Isend(contin,1,MPI_DOUBLE,id_worker,tag2,MPI_COMM_WORLD,&req);}


         }else if(id_bowl < (*param).grad_workers){

            int tag1 = 123, tag2 = 999;
            col_vector theta((*param).theta_size), etastar((*param).y_size);
            double *contin; contin = new double[1];

            while(true){

               MPI_Recv(contin,1,MPI_DOUBLE,0,tag2,MPI_COMM_WORLD,&status); if(contin[0]==0) break;

               if(multipleSends){
                  receive_2vector(0,theta,(*param).update->eta_moon,tag1);

                  int fix_tag = 621;
                  double *main_value; main_value = new double[1]; main_value[0] = 621;
                  MPI_Request req; MPI_Isend(main_value,1,MPI_DOUBLE,0,fix_tag,MPI_COMM_WORLD,&req);

               }else{
                  int tag1 = 123, tag2 = 999, tag3 = 321;
                  double *mesg1,*mesg2;
                  mesg1 = new double[(*param).theta_size]; mesg2 = new double[(*param).y_size];
                  MPI_Recv(mesg1,(*param).theta_size,MPI_DOUBLE,0,tag1,MPI_COMM_WORLD,&status);
                  MPI_Recv(mesg2,(*param).y_size,MPI_DOUBLE,0,tag3,MPI_COMM_WORLD,&status);

                  for(size_t i=0; i<(*param).theta_size; i++){
                  theta[i] = mesg1[i];}

                  for(size_t i=0; i<(*param).y_size; i++){
                     (*param).update->eta_moon[i] = mesg2[i];}
               }
               
               boost::ptr_vector<int> tasks;
               seasoning((*param).num_tasks_grad, (*param).grad_workers, (*param).id_bowl, true,tasks);
               int *ptr_tasks;
               ptr_tasks = new int[tasks.size()];

               size_t i=0;
               for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
                  
               for (size_t i=0;i < tasks.size(); i++){
                  bool left, right; int task = ptr_tasks[i];
                  int index_theta = gradient_majic_spread((*param).optim->central,task,left,right,(*param).theta_size);

                  //std::cout << task << " is my task " << index_theta << " ---- " << (*param).id_bowl << " should work on " << left << " | " << right << std::endl;
                  
                  double gradx = 0;
                  etastar = (*param).update->eta_moon;

                  try {
                     gradient_i(gradx,theta,etastar, param, index_theta,left, right);
                     double *value; value = new double[1]; value[0] = gradx;
                     MPI_Request req; MPI_Isend(value,1,MPI_DOUBLE,0,task,MPI_COMM_WORLD,&req);
                  } 

                  catch(const char *error){
                     if((*param).internalverbose) cout << "seven peppers seasoning is added" << std::endl;
                     double *value; value = new double[1]; value[0] = 10000000;
                     MPI_Request req; MPI_Isend(value,1,MPI_DOUBLE,0,task,MPI_COMM_WORLD,&req);
                  }

                  catch (const std::exception &ex){
                     if((*param).internalverbose) cout << "seven peppers seasoning is added" << std::endl;
                     double *value; value = new double[1]; value[0] = 10000000;
                     MPI_Request req; MPI_Isend(value,1,MPI_DOUBLE,0,task,MPI_COMM_WORLD,&req);
                  }
               }
            }
            
         }else{
            
            //help in line search!
            //id_bowl >= (*param).grad_workers
/*
            int tag1 = 123, tag2 = 999;
            col_vector theta((*param).theta_size), etastar((*param).y_size);
            double *contin; contin = new double[1];

            while(true){

               MPI_Recv(contin,1,MPI_DOUBLE,0,tag2,MPI_COMM_WORLD,&status); if(contin[0]==0) break;
               receive_2vector(0,theta,(*param).update->eta_moon,tag1);

               int fix_tag = 621;
               double *main_value; main_value = new double[1]; main_value[0] = 621;
               MPI_Request req; MPI_Isend(main_value,1,MPI_DOUBLE,0,fix_tag,MPI_COMM_WORLD,&req);
               
               boost::ptr_vector<int> tasks;
               seasoning((*param).num_tasks_linesearch, (*param).linesearch_workers, (*param).id_bowl, true,tasks);
               int *ptr_tasks;
               ptr_tasks = new int[tasks.size()];

               size_t i=0;
               for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
                  
               for (size_t i=0;i < tasks.size(); i++){
                  bool left, right; int task = ptr_tasks[i];
                  int index_theta = gradient_majic_spread((*param).optim->central,task,left,right,(*param).theta_size);
                  
                  etastar = (*param).update->eta_moon;
                  double getvalue = 0;
                  linesearch_i(getvalue,theta,etastar,param);
                  double *value; value = new double[1]; value[0] = getvalue;
                  MPI_Request req; MPI_Isend(value,1,MPI_DOUBLE,0,task,MPI_COMM_WORLD,&req);
               }
            }

            std::cout << "Worker #: " << id_bowl << " is not useless anymore" << std::endl;

            */
         }
                  
      }

   }else{
   
   //boost::scoped_ptr<Bowl<tkp::splines>> param{new Bowl<tkp::splines>{}};

   getdata(param); 
   set_priors(param);
   setmodel(param);

   //blaze::setNumThreads((*param).optim->num_threads); //export BLAZE_NUM_THREADS=6
   (*param).optim->central = false;

   std::cout << "1. The model is " << (*param).Model << std::endl;
   std::cout << "2. The Qx_type is " << (*param).Qx_type << std::endl;
   //std::cout << "3. Number of threads: " << (*param).optim->num_threads << std::endl;
   std::cout << "3. Number of threads: " << blaze::getNumThreads() << std::endl;
   if((*param).optim->smartGrad && (*param).theta_size>1) std::cout << "4. Gradient is smart" << std::endl;
   else std::cout << "4: Gradient is stupid"  << std::endl;
   //if((*param).optim->central) std::cout << "4.2: IT is central" << std::endl;
   //else std::cout << "4.2: Gradient is forward" << std::endl;

   std::cout << "5: x size: " << (*param).x_size << ", y size: " << (*param).y_size  << " and theta size: " << (*param).theta_size << std::endl;
   std::cout << "6: Bowl contains: " << std::endl;
   (*param).pour_the_Bowl();

   check_Model(param);

   size_t tsize = 1; col_vector test_time(tsize);
   for(size_t i=0; i<tsize;i++)
   {
      auto start = std::chrono::system_clock::now();
      if((*param).RTheta)
      {
         col_vector theta_star((*param).theta_size,0.0),eta_star((*param).y_size);

         //make this on!!!!!!!!!!!!!!!!!!!!
         //print((*param).bin_size);
         //testfunction(param);
         //test_pivot_cholesky(param);
         
         optimize_p_theta_given_y(wrapper_p_theta_given_y,theta_star,eta_star,param);
         cooking_table(theta_star,eta_star,param);

         /*
         (*param).set_utensils_x();
         //Search Strategies

         if((*param).Model=="Gaussian") {(*param).correction->Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
         }else if((*param).Model=="Poisson") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
         }else if((*param).Model=="Binomial") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);}

         correction(recipe,theta_star,param);
         */
         
      }
      //else
      //meal_is_ready((*param).x_size);
 
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;
      //std::cout << "elapsed time: " << elapsed_seconds.count() << std::endl;

      test_time[i] = elapsed_seconds.count();
   }

   std::cout << "elapsed time: " << mean(test_time) << std::endl;
   std::cout << "Function Calls: " << (*param).optim->optfunctioncall << std::endl;

   //auto start = std::chrono::system_clock::now();
   //auto end = std::chrono::system_clock::now();
   //std::chrono::duration<double> elapsed_seconds = end-start;
   //std::cout << elapsed_seconds.count() << std::endl;

   //std::cout << (*param).addnormcnst << std::endl;

   }


   MPI_Barrier(MPI_COMM_WORLD);
   mid_omp = omp_get_wtime(); 
   if((*param).id_bowl==0) std::cout << "To get theta star: " << mid_omp - start_omp  << '\n';


   if(size_bowl > 1){
      col_vector eigenvalues((*param).theta_size,0.0); row_matrix eigenvectors((*param).theta_size, (*param).theta_size,0.0); row_matrix hessian;
      chef_cooking_table(param,eigenvalues,eigenvectors,hessian);
      if((*param).go_to_oven) chef_cooking_table(param,eigenvalues,eigenvectors,hessian);
      oven(param,eigenvalues,eigenvectors,hessian);

   }

   MPI_Barrier(MPI_COMM_WORLD);
   end_omp = omp_get_wtime(); 
   if((*param).id_bowl==0){
      std::cout << "To get inference: " << end_omp - start_omp  << std::endl;
      std::cout << std::endl;
      //std::cout << "Size of x: " << (*param).x_size << " and Size of y" << (*param).y_size << std::endl;
      std::cout << "Number of Constraints: " << (*param).num_Con << std::endl;
      std::cout << "Number of threads: " << blaze::getNumThreads() << std::endl;
      std::cout << "Number of processes: " << (*param).size_bowl << std::endl;
   }


   MPI_Finalize();

}

void setmodel(ptr_bowl &param)
{

   if((*param).Model=="Poisson")
   {
      like_Model_eta = &like_Poisson_eta; 
      approx_NoNGaussian_RD_eta = &approx_NoNGaussian_RD_eta_Poisson;
      (*param).update->eta_moon = blaze::log((*param).y_response+0.0001);
      recipe = &poisson_recipe;
      correction = &correction_Non_Gaussian;
      //get_txt_column("././Data/x_initial.txt",(*param).x_moon);
      //Qlike_Model = &Qlike_Poisson;
      //LA_get_gammas_itern = &SLA_get_gammas_Poisson;
      //like_Model = &like_Poisson; 
   
      //myModel =&myModel_Poisson; //for LA
      
      (*param).logfac.resize((max((*param).y_response))+1);
      log_fac((*param).logfac);
      for(size_t i =0; i< (*param).y_size; i++) (*param).addnormcnst -= (*param).logfac[(*param).y_response[i]];

   } 
   else if((*param).Model=="Binomial")
   {
      like_Model_eta = &like_Binomial_eta; 
      approx_NoNGaussian_RD_eta = &approx_NoNGaussian_RD_eta_Binomial;
      col_vector num = ((*param).y_response + 0.001);
      col_vector denom = ((*param).Ntrials-(*param).y_response + 0.001);

      (*param).update->eta_moon = log(num/denom + 0.0001);
      recipe = &binomail_recipe;
      correction = &correction_Non_Gaussian;
      //Qlike_Model = &Qlike_Binomial;
      //SLA_get_gammas_itern = &SLA_get_gammas_Binomial;
      //like_Model = &like_Binomial;
      //myModel =&myModel_Binomial; //for LA
      //(*param).x_moon[0] = blaze::log(blaze::mean((*param).y_response))/(1.0-blaze::log(blaze::mean((*param).y_response)));
   }
   else if((*param).Model=="Gaussian")
   {
      like_Model_eta = &like_Gaussian_eta; 
      (*param).update->eta_moon = (*param).y_response+0.0001;
      correction = &correction_Gaussian;

      //if((*param).SLApp) {(*param).SLApp = false; (*param).GApp = true; print("--> SLA and GA for Gaussian is the same!");}
      //Qlike_Model = &Qlike_Gaussian;
      //like_Model = &like_Gaussian;
      //SLA_get_gammas_itern = &SLA_get_gammas_Gaussian;
      //(*param).x_moon[0] = blaze::mean((*param).y_response);
   }
}

void check_Model(ptr_bowl &param)
{
   blaze::DynamicVector<string> effi, priors_types; //make it private?
   blaze::DynamicVector<int> rankdef;
   blaze::DynamicVector<double> xpts;
   col_vector mu, y_response;
   sym_matrix Qx_fixed, invQx_fixed;
   size_t x_size, y_size, theta_size, theta_size_Qx;
   string Model, Qx_type, strategy;
   blaze::CompressedMatrix<double> A;

   if((*param).id_bowl==0){
   std::cout << "priors types" << std::endl;
   std::cout << (*param).priors_types << std::endl;
   std::cout << "effi" << std::endl;
   std::cout << (*param).effi << std::endl;
   std::cout << "rankdef" << std::endl;
   std::cout << (*param).rankdef << std::endl;
   std::cout << "xpts" << std::endl;
   std::cout << (*param).xpts << std::endl;
   std::cout << "theta_size_Qx" << std::endl;
   std::cout << (*param).theta_size_Qx << std::endl;
   std::cout << "theta_size" << std::endl;
   std::cout << (*param).theta_size << std::endl;
   std::cout << "intercept" << std::endl;
   std::cout << (*param).x_mu << std::endl;
   std::cout << "zcov" << std::endl;
   std::cout << (*param).zcov << std::endl;}
}

double p_thetaj(col_vector &theta,col_vector &theta_mode, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg)
{
   col_vector z(theta_mode.size(),0.0);
   theta2z(theta,z,theta_mode,eigen_values,eigen_vectors);
   ptr sd{new double{0.0}};
   ptr value{new double{0.0}};

   for (size_t i = 0; i < theta_mode.size(); i++) 
   {
			*sd = (z[i] > 0 ? stdev_corr_pos[i] : stdev_corr_neg[i]);
			*value = *value -(0.5 * pow(z[i]/(*sd),2) );
	}

   return *value;
}

void p_thetaj_given_y(col_vector &theta_star, row_matrix &COV, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg)
{
   for(size_t index = 0; index < theta_star.size(); index ++)
   {
      ptr theta_fixed{new double{0.0}};
      ptr dthetaj{new double{0.0}};

      col_vector seq_thetaj = linspace<columnVector>(200, -10, 10);
      ptr dtheta{new double{seq_thetaj[1]-seq_thetaj[0]}};
      seq_thetaj = theta_star[index] + sqrt(COV(index,index))*seq_thetaj;

      col_vector theta(theta_star.size(),0.0);
      col_vector ldens_thetaj(seq_thetaj.size(),0.0);
      for (size_t i = 0; i < seq_thetaj.size(); i++) 
      {
         *theta_fixed =  seq_thetaj[i];
         for (size_t j = 0; j < theta_star.size(); j++) 
         {
            if (j != index) theta[j] = theta_star[j] + (COV(index, j) / COV(index, index)) * ((*theta_fixed) - theta_star[index]);
            else theta[j] = *theta_fixed;
         }
         ldens_thetaj[i] = p_thetaj(theta,theta_star,eigen_values,eigen_vectors,stdev_corr_pos,stdev_corr_neg);
      }

      *dthetaj = (*dtheta)*(sqrt(COV(index,index)));
      ldens_thetaj = blaze::softmax(ldens_thetaj);
      ldens_thetaj = ldens_thetaj/(*dthetaj);

      export_density(seq_thetaj,ldens_thetaj,index,"HP");
   }
   
}

void p_thetaj_given_y_index(int &index, col_vector &theta_star, row_matrix &COV, col_vector &eigen_values, row_matrix &eigen_vectors,col_vector &stdev_corr_pos,col_vector &stdev_corr_neg)
{
   ptr theta_fixed{new double{0.0}};
   ptr dthetaj{new double{0.0}};

   col_vector seq_thetaj = linspace<columnVector>(200, -10, 10);
   ptr dtheta{new double{seq_thetaj[1]-seq_thetaj[0]}};
   seq_thetaj = theta_star[index] + sqrt(COV(index,index))*seq_thetaj;

   col_vector theta(theta_star.size(),0.0);
   col_vector ldens_thetaj(seq_thetaj.size(),0.0);
   for (size_t i = 0; i < seq_thetaj.size(); i++) 
   {
      *theta_fixed =  seq_thetaj[i];
      for (size_t j = 0; j < theta_star.size(); j++) 
      {
         if (j != index) theta[j] = theta_star[j] + (COV(index, j) / COV(index, index)) * ((*theta_fixed) - theta_star[index]);
         else theta[j] = *theta_fixed;
      }
      ldens_thetaj[i] = p_thetaj(theta,theta_star,eigen_values,eigen_vectors,stdev_corr_pos,stdev_corr_neg);
   }

   *dthetaj = (*dtheta)*(sqrt(COV(index,index)));
   ldens_thetaj = blaze::softmax(ldens_thetaj);
   ldens_thetaj = ldens_thetaj/(*dthetaj);

   size_t ind = index;
   export_density(seq_thetaj,ldens_thetaj,ind,"HP");
}

void get_CDD_Design(size_t &theta_size, row_matrix &CCD_design_mat, col_vector &log_weights, ptr &fvalue)
{
   boost::scoped_ptr<string> name{new string{""}}; 
   if(theta_size==2) {CCD_design_mat.resize(9,2); *name = "./GLP/CCDdesign/dim2.txt";}
   else if(theta_size==3) {CCD_design_mat.resize(15,3); *name = "./GLP/CCDdesign/dim3.txt";}
   else if(theta_size==4) {CCD_design_mat.resize(25,4); *name = "./GLP/CCDdesign/dim4.txt";}
   else if(theta_size==5) {CCD_design_mat.resize(27,5); *name = "./GLP/CCDdesign/dim5.txt";}
   else if(theta_size==6) {CCD_design_mat.resize(45,6); *name = "./GLP/CCDdesign/dim6.txt";}

   ifstream myfile (*name);
   if (myfile.is_open())
   {
      for(size_t i=0; i <CCD_design_mat.rows() ;i++)
         for(size_t j=0; j <CCD_design_mat.columns() ;j++)
            myfile >> CCD_design_mat(i,j);
      myfile.close();
   }
   else cout << "Unable to open the CCD design table"; 

   log_weights.resize(CCD_design_mat.rows(),CCD_design_mat.columns());
   ptr f0{new double{1.1}}, central_weight{new double{0.0}},weight{new double{0.0}},nexperiments{new double{0.0}};
   *fvalue = std::max((*f0),1.0)* sqrt((double)theta_size);
   *nexperiments = (double)CCD_design_mat.rows();
   *weight = 1/((*nexperiments - 1)*(1 + std::exp(-0.5 * (pow(*fvalue,2)))*((pow(*fvalue,2))/ ((double)CCD_design_mat.columns()) - 1.0)));
   *central_weight = 1.0 - (*nexperiments - 1.0) * (*weight);
   log_weights = std::log(*weight);
   log_weights[0] = std::log(*central_weight);
   //print(trans(log_weights));
}

void for_cooking_GA(col_vector &theta, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors)
{

   correction(recipe,theta,param);
   //print((*param).correction->invQ);
   col_vector ga_xi_sd((*param).x_size,0.0);
   for(size_t iter=0;iter<(*param).x_size;iter++)  ga_xi_sd[iter] = sqrt((*param).correction->invQ(iter,iter));  
   save_vectors->means.push_back(new col_vector{(*param).correction->x_star});
   save_vectors->sds.push_back(new col_vector{ga_xi_sd});

   save_vectors->means_eta.push_back(new col_vector{(*param).update->update_eta_star});

   sym_matrix AA = (*param).A*(*param).correction->invQ*trans((*param).A);

   col_vector testss = band(AA,0);
   for(size_t iter=0;iter<testss.size();iter++) testss[iter] = sqrt(testss[iter]);

   //print(trans(testss));

   save_vectors->sds_eta.push_back(new col_vector{testss});

/*
   if(true)
   {
      col_vector ga_xi_sd((*param).x_size,0.0);
      for(size_t iter=0;iter<(*param).x_size;iter++)  ga_xi_sd[iter] = sqrt((*param).set->Qstar(iter,iter));  
      save_vectors->means.push_back(new col_vector{(*param).set->update_x_star});
      save_vectors->sds.push_back(new col_vector{ga_xi_sd});
   } else { //fix
      //the difference between the two appraches is if we are getting Qstar or Qinv
      //Qstar = inv(declsym(Qstar));
      col_vector ga_xi_sd((*param).x_size,0.0);
      //for(size_t iter=0;iter<xstar.size();iter++)  ga_xi_sd[iter] = sqrt(Qstar(iter,iter));  
      save_vectors->means.push_back(new col_vector{(*param).set->update_x_star});
      save_vectors->sds.push_back(new col_vector{ga_xi_sd});
   }
*/

}

void cooking_GA(int &task, double &value, col_vector &theta, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors){

   if((*param).id_bowl==0){

      correction(recipe,theta,param);
      col_vector ga_xi_sd((*param).x_size,0.0);
      for(size_t iter=0;iter<(*param).x_size;iter++)  ga_xi_sd[iter] = sqrt((*param).correction->invQ(iter,iter));  
      
      save_vectors->means.push_back(new col_vector{(*param).correction->x_star});
      save_vectors->sds.push_back(new col_vector{ga_xi_sd});
      save_vectors->means_eta.push_back(new col_vector{(*param).update->update_eta_star});
      
      sym_matrix AA = (*param).A*(*param).correction->invQ*trans((*param).A);
      col_vector prec_vec = band(AA,0);
      for(size_t iter=0;iter<prec_vec.size();iter++) prec_vec[iter] = sqrt(prec_vec[iter]);

      save_vectors->sds_eta.push_back(new col_vector{prec_vec});

   }else{

      correction(recipe,theta,param);
      col_vector ga_xi_sd((*param).x_size,0.0);
      for(size_t iter=0;iter<(*param).x_size;iter++)  ga_xi_sd[iter] = sqrt((*param).correction->invQ(iter,iter));  
      
      if(multipleSends){
         int tag = task + 1000;
         sym_matrix AA = (*param).A*(*param).correction->invQ*trans((*param).A);
         col_vector prec_vec = band(AA,0);
         for(size_t iter=0;iter<prec_vec.size();iter++) prec_vec[iter] = sqrt(prec_vec[iter]);
         isend_4vector(0, (*param).correction->x_star, ga_xi_sd, (*param).update->update_eta_star, prec_vec, value, tag);

      }else{
         int tag = task + 1000;
         isend_vector(0,(*param).correction->x_star, tag);
         tag = task + 1001;
         isend_vector(0,ga_xi_sd, tag);
         tag = task + 1002;
         isend_vector(0,(*param).update->update_eta_star, tag);
         
         sym_matrix AA = (*param).A*(*param).correction->invQ*trans((*param).A);
         col_vector prec_vec = band(AA,0);
         for(size_t iter=0;iter<prec_vec.size();iter++) prec_vec[iter] = sqrt(prec_vec[iter]);

         tag = task + 1003;
         isend_vector(0,prec_vec, tag);
         tag = task + 1004;
         isend_double(0,value,tag);
      }
      
   }

}

void GA_is_cooked(int id_bowl, int &task, col_vector &weights, ptr_bowl &param, boost::scoped_ptr<storage_type1> &save_vectors){

   col_vector ga_xi_sd((*param).x_size,0.0), prec_vec((*param).y_size,0.0); double value = 0.0;
   
   if(multipleSends){

      int tag = task + 1000;

      receive_4vector(id_bowl,(*param).correction->x_star,ga_xi_sd,(*param).update->update_eta_star,prec_vec,value,tag);

      save_vectors->means.push_back(new col_vector{(*param).correction->x_star});
      save_vectors->sds.push_back(new col_vector{ga_xi_sd});
      save_vectors->means_eta.push_back(new col_vector{(*param).update->update_eta_star});
      save_vectors->sds_eta.push_back(new col_vector{prec_vec});
      weights[task] = value;


   }else{
      int tag = task + 1000;
      receive_vector(id_bowl, (*param).correction->x_star, tag);
      save_vectors->means.push_back(new col_vector{(*param).correction->x_star});

      tag = task + 1001;
      receive_vector(id_bowl, ga_xi_sd, tag);
      save_vectors->sds.push_back(new col_vector{ga_xi_sd});

      tag = task + 1002;
      receive_vector(id_bowl, (*param).update->update_eta_star, tag);
      save_vectors->means_eta.push_back(new col_vector{(*param).update->update_eta_star});

      tag = task + 1003;
      receive_vector(id_bowl, prec_vec, tag);
      save_vectors->sds_eta.push_back(new col_vector{prec_vec});

      tag = task + 1004;
      receive_double(id_bowl,value,tag);
      weights[task] = value;
   }
   
}

void GA_xi_marginal_s(ptr_bowl &param, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage){


   //auto start = std::chrono::system_clock::now();

   auto xstar = GA_storage->means.begin();
   size_t xsize = (*xstar).size();

   ptr sum_w{new double{0.0}};
   col_vector weighted_mean(xsize,0.0),weighted_stdev(xsize,0.0);
   col_vector ga_xi_sd(xsize,0.0),w_sd(xsize,0.0);
   col_matrix a(xsize,weights.size()),b(xsize,weights.size()),c(xsize,weights.size());

   size_t k = 0;
   for (auto ga_xi_sd = GA_storage->sds.begin(); xstar != GA_storage->means.end(); xstar++,ga_xi_sd++)
   {
      blaze::column(a, k) = 1.0/(*ga_xi_sd);
      blaze::column(b, k) = - (*xstar)/(*ga_xi_sd);
      blaze::column(c, k) = weights[k]/(*ga_xi_sd);

      weighted_mean += weights[k]*(*xstar);
      w_sd += weights[k] * (blaze::pow((*ga_xi_sd),2) + blaze::pow((*xstar),2));
      (*sum_w) += weights[k];
      k++;
   }

   weighted_mean.scale(1.0/(*sum_w));
   w_sd.scale(1.0/(*sum_w));
   weighted_stdev = blaze::sqrt(blaze::max(0.0, w_sd - blaze::pow(weighted_mean,2)));


   //auto end = std::chrono::system_clock::now();
   //std::chrono::duration<double> elapsed_seconds = end-start;
   //std::cout << "elapsed time: " << elapsed_seconds.count() << std::endl;

   //start = std::chrono::system_clock::now();
   static const double log_norm_const_gaussian = -0.918938533204672741780329736407;	// log(1.0/sqrt(2.0*M_PI)) 
   col_vector GQ_points = linspace<columnVector>(200, -10, 10);

   if(!((*param).RTheta))
   {
      GQ_points.resize(9);
      GQ_points[0] = -4.5127458634; GQ_points[1] = -3.20542900286; GQ_points[2] = -2.07684797868; GQ_points[3] = -1.02325566379;
      GQ_points[4] = 0.0; GQ_points[5] = 1.02325566379; GQ_points[6] = 2.07684797868;GQ_points[7] = 3.20542900286 ;GQ_points[8] = 4.5127458634;
   }

   col_vector x_points(GQ_points.size(),0.0),std_x_points(GQ_points.size(),0.0);
   col_vector logdens(GQ_points.size(),0.0),dens_x_points(GQ_points.size(),0.0);
   
   for(size_t ind = 0; ind <xsize ;ind++)
   {
      x_points = weighted_stdev[ind]* GQ_points + weighted_mean[ind];
      for(size_t k = 0; k <weights.size() ;k++)
      {
         std_x_points = a(ind,k) * x_points + b(ind,k);
         logdens = log_norm_const_gaussian - 0.5 * (std_x_points * std_x_points);
         dens_x_points = dens_x_points + c(ind,k) * blaze::exp(logdens);
      }
      logdens = blaze::log(dens_x_points);
      logdens -= blaze::max(logdens);

      boost::scoped_ptr<tk::polynomial> sp_fun{new tk::polynomial{}};
      sp_fun.get()->set_points(GQ_points,logdens); 

      x_points.resize(321);dens_x_points.resize(321);
      x_points = linspace<columnVector>(321, -9.7062359974, 9.7062359974 );

      (*sp_fun)(x_points,dens_x_points);
      normalize_simpson_321_p(dens_x_points);

      x_points.scale(weighted_stdev[ind]);
      x_points += (weighted_mean[ind]);
      dens_x_points.scale(1.0/weighted_stdev[ind]);
      export_density(x_points,dens_x_points,ind,"GA");

      x_points.resize(GQ_points.size() );dens_x_points.resize(GQ_points.size());
      x_points.reset(); std_x_points.reset(); logdens.reset(); dens_x_points.reset();
      
   }      

   //end = std::chrono::system_clock::now();
   //elapsed_seconds = end-start;
   //std::cout << "elapsed time11111: " << elapsed_seconds.count() << std::endl;
}

void GA_xi_marginal_pot1(col_matrix &a, col_matrix &b, col_matrix &c, size_t &xsize, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage,col_vector &weighted_mean, col_vector &weighted_stdev){
   auto xstar = GA_storage->means.begin();

   ptr sum_w{new double{0.0}};
   col_vector ga_xi_sd(xsize,0.0),w_sd(xsize,0.0);

   size_t k = 0;
   for (auto ga_xi_sd = GA_storage->sds.begin(); xstar != GA_storage->means.end(); xstar++,ga_xi_sd++)
   {
      blaze::column(a, k) = 1.0/(*ga_xi_sd);
      blaze::column(b, k) = - (*xstar)/(*ga_xi_sd);
      blaze::column(c, k) = weights[k]/(*ga_xi_sd);

      weighted_mean += weights[k]*(*xstar);
      w_sd += weights[k] * (blaze::pow((*ga_xi_sd),2) + blaze::pow((*xstar),2));
      (*sum_w) += weights[k];
      k++;
   }

   weighted_mean.scale(1.0/(*sum_w));
   w_sd.scale(1.0/(*sum_w));
   weighted_stdev = blaze::sqrt(blaze::max(0.0, w_sd - blaze::pow(weighted_mean,2)));
}

void GA_xi_marginal_pot2(bool &RTheta, int &ind, col_vector &a, col_vector &b, col_vector &c, col_vector &weighted_mean, col_vector &weighted_stdev, col_vector &weights){
   static const double log_norm_const_gaussian = -0.918938533204672741780329736407;	// log(1.0/sqrt(2.0*M_PI)) 
   static col_vector GQ_points = linspace<columnVector>(200, -10, 10);

   //if(!((*param).RTheta))
   if(!RTheta)
   {
      GQ_points.resize(9);
      GQ_points[0] = -4.5127458634; GQ_points[1] = -3.20542900286; GQ_points[2] = -2.07684797868; GQ_points[3] = -1.02325566379;
      GQ_points[4] = 0.0; GQ_points[5] = 1.02325566379; GQ_points[6] = 2.07684797868;GQ_points[7] = 3.20542900286 ;GQ_points[8] = 4.5127458634;
   }

   col_vector x_points(GQ_points.size(),0.0),std_x_points(GQ_points.size(),0.0);
   col_vector logdens(GQ_points.size(),0.0),dens_x_points(GQ_points.size(),0.0);
   col_vector xx_points(321), ddens_x_points(321);

   x_points = weighted_stdev[ind]* GQ_points + weighted_mean[ind];
   for(size_t k = 0; k <weights.size() ;k++)
   {
      std_x_points = a[k] * x_points + b[k];
      logdens = log_norm_const_gaussian - 0.5 * (std_x_points * std_x_points);
      dens_x_points = dens_x_points + c[k] * blaze::exp(logdens);
   }
   logdens = blaze::log(dens_x_points);
   logdens -= blaze::max(logdens);

   boost::scoped_ptr<tk::polynomial> sp_fun{new tk::polynomial{}};
   sp_fun.get()->set_points(GQ_points,logdens); 

  // x_points.resize(321);dens_x_points.resize(321);

   xx_points = linspace<columnVector>(321, -9.7062359974, 9.7062359974 );

   (*sp_fun)(xx_points,ddens_x_points);
   normalize_simpson_321_p(ddens_x_points);

   xx_points.scale(weighted_stdev[ind]);
   xx_points += (weighted_mean[ind]);
   ddens_x_points.scale(1.0/weighted_stdev[ind]);
   size_t index = ind;
   export_margx_density(xx_points,ddens_x_points,index,"GA"); //esma

   //x_points.resize(GQ_points.size() );dens_x_points.resize(GQ_points.size());
   //x_points.reset(); std_x_points.reset(); logdens.reset(); dens_x_points.reset(); 

}

void CCD_Stratetgy_original(col_vector &etastar, ptr_bowl &param, row_matrix &thetas,col_vector &weights, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3){ 
   if((*param).correction->vb) (*param).correction->part_b_vec.resize((*param).y_size); //VARBAYES
   size_t x_size = etastar.size(),theta_size = theta_star.size();
  
   row_matrix all_CCD_points;
   ptr fvalue{new double{0.0}};
   col_vector log_weights;
   get_CDD_Design((*param).theta_size,all_CCD_points,log_weights,fvalue);

   for(size_t j=0; j< all_CCD_points.columns();j++)
      for(size_t i=1; i< all_CCD_points.rows();i++)
      {
         if(all_CCD_points(i,j)>0)
            all_CCD_points(i,j) = stdev_corr_pos[j]*all_CCD_points(i,j);
         else 
            all_CCD_points(i,j) = stdev_corr_neg[j]*all_CCD_points(i,j);
      }

   all_CCD_points = (*fvalue)*all_CCD_points;

   thetas.resize(all_CCD_points.rows(),all_CCD_points.columns());
   weights.resize(all_CCD_points.rows());
   col_vector give_z(theta_size,0.0),get_theta(theta_size,0.0);
   ptr get_y{new double{0.0}};
   std::cout << std::endl;
   for(size_t i=0; i< all_CCD_points.rows();i++)
   {
      for(size_t j=0; j<theta_size;j++) give_z[j] = all_CCD_points(i,j);
      get_theta = 0.0;
      z2theta(give_z,get_theta,theta_star,eigenvalues,eigenvectors);
      *get_y = 0.0; 
      opt_fun_p_theta_given_y(get_y,get_theta,etastar,param); 
      
      if((*param).GApp) {print("cooking GA..."); for_cooking_GA(get_theta, param,storage1);} 
      //else if((*param).SLApp) {print("cooking SLA..."); for_cooking_SLA(param,storage1);}
      //else if(LApp){ for_cooking_LA(x_star_k,Qx_star_k,Qx,theta_star,storage2,storage3,param,mu,D); }
      
      for(size_t j=0; j<theta_size;j++) thetas(i,j) = get_theta[j];
      weights[i] = *get_y;
   }
   ptr max_val{new double{0.0}};
   //print(trans(weights));
   //print(trans(log_weights));

   weights = weights + log_weights;
   if(max(weights)>10) weights = -weights;

   //print(trans(weights));
   *max_val = max(weights); //i added this recently
   weights = weights - *max_val; //i added this recently 
   weights = blaze::softmax(weights); // because this softmax function is not good enough


   *max_val = max(weights);
   weights = weights/(*max_val);
   //print(weights);

   if((*param).GApp) GA_xi_marginal_s(param,weights,storage1);
   //else if((*param).SLApp) SLA_xi_marginal((*param).RTheta,weights,storage1);
   // else if(LApp) LA_xi_marginal(weights,storage2,storage3,x_size,theta_size);
   //print(trans(weights));
}

void CCD_Stratetgy(col_vector &etastar, ptr_bowl &param, row_matrix &thetas,col_vector &weights, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3){  
   size_t x_size = etastar.size(),theta_size = theta_star.size();
  
   row_matrix all_CCD_points;
   ptr fvalue{new double{0.0}};
   col_vector log_weights;
   get_CDD_Design((*param).theta_size,all_CCD_points,log_weights,fvalue);


   for(size_t j=0; j< all_CCD_points.columns();j++)
      for(size_t i=1; i< all_CCD_points.rows();i++)
      {
         if(all_CCD_points(i,j)>0)
            all_CCD_points(i,j) = stdev_corr_pos[j]*all_CCD_points(i,j);
         else 
            all_CCD_points(i,j) = stdev_corr_neg[j]*all_CCD_points(i,j);
      }

   if((*param).id_bowl==0) print(stdev_corr_pos);
   if((*param).id_bowl==0) print(stdev_corr_neg);

   all_CCD_points = (*fvalue)*all_CCD_points;

   thetas.resize(all_CCD_points.rows(),all_CCD_points.columns());
   weights.resize(all_CCD_points.rows());

   /* //delete
   thetas.resize(1,(*param).theta_size);
   weights.resize(1,1);
   all_CCD_points.resize(1,(*param).theta_size);

   weights[0] = 1;
   for(size_t j=0; j<theta_size;j++) all_CCD_points(0,j) =0.0;

   print(weights);
   print(all_CCD_points);
   //delete */

   col_vector give_z(theta_size,0.0),get_theta(theta_size,0.0);
   ptr get_y{new double{0.0}};

   std::cout << std::endl;
   for(size_t i=0; i< all_CCD_points.rows();i++)
   {
      for(size_t j=0; j<theta_size;j++) give_z[j] = all_CCD_points(i,j);
      get_theta = 0.0;
      z2theta(give_z,get_theta,theta_star,eigenvalues,eigenvectors);
      *get_y = 0.0; 

      opt_fun_p_theta_given_y(get_y,get_theta,etastar,param); 

      if((*param).GApp) {print("cooking GA..."); for_cooking_GA(get_theta, param,storage1);} 
      //else if((*param).SLApp) {print("cooking SLA..."); for_cooking_SLA(param,storage1);}
      //else if(LApp){ for_cooking_LA(x_star_k,Qx_star_k,Qx,theta_star,storage2,storage3,param,mu,D); }
      
      for(size_t j=0; j<theta_size;j++) thetas(i,j) = get_theta[j];
      weights[i] = *get_y;
   }
   ptr max_val{new double{0.0}};
   //print(trans(log_weights));
   //std::cout << "weights: " << sum(weights) << std::endl;
   //print(weights); 


   weights = -weights + log_weights;
   weights = weights - weights[0];
   
   weights = blaze::exp(weights);
   //print(trans(weights));

   weights = weights/sum(weights);
   //print(trans(weights));

   


   //delete
   //weights[0] = 1;
   //delete

   if((*param).GApp) GA_xi_marginal_s(param,weights,storage1);
   //else if((*param).SLApp) SLA_xi_marginal((*param).RTheta,weights,storage1);
   // else if(LApp) LA_xi_marginal(weights,storage2,storage3,x_size,theta_size);
   //print(trans(weights));
}

void CCD_Stratetgy_shelf1(col_vector &log_weights, row_matrix &all_CCD_points, size_t &theta_size, ptr_bowl &param, row_matrix &thetas, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3){  
   
   ptr fvalue{new double{0.0}};
   get_CDD_Design((*param).theta_size,all_CCD_points,log_weights,fvalue);

   for(size_t j=0; j< all_CCD_points.columns();j++)
      for(size_t i=1; i< all_CCD_points.rows();i++)
      {
         if(all_CCD_points(i,j)>0)
            all_CCD_points(i,j) = stdev_corr_pos[j]*all_CCD_points(i,j);
         else 
            all_CCD_points(i,j) = stdev_corr_neg[j]*all_CCD_points(i,j);
      }

   all_CCD_points = (*fvalue)*all_CCD_points;
   thetas.resize(all_CCD_points.rows(),all_CCD_points.columns());
}

void CCD_Stratetgy_shelf2(int &i, double &value, row_matrix &all_CCD_points, ptr_bowl &param, row_matrix &thetas,col_vector &weights, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &eigenvalues, row_matrix  &eigenvectors,boost::scoped_ptr<storage_type1> &storage1, boost::scoped_array<storage_type2> &storage2, boost::scoped_ptr<storage_type3> &storage3){

   size_t theta_size = (*param).theta_size;
   col_vector give_z(theta_size,0.0),get_theta(theta_size,0.0);
   ptr get_y{new double{0.0}};

   for(size_t j=0; j<theta_size;j++) give_z[j] = all_CCD_points(i,j);
   get_theta = 0.0;
   z2theta(give_z,get_theta,(*param).theta_star,eigenvalues,eigenvectors);
   *get_y = 0.0; 
   //(*param).update->eta_moon = (*param).eta_star;

   opt_fun_p_theta_given_y(get_y,get_theta,(*param).update->eta_moon,param); 

   value =  *get_y; //weights[i] = *get_y;

   if((*param).GApp) {print("cooking GA..."); cooking_GA(i, value, get_theta, param,storage1);} 
   //else if((*param).SLApp) {print("cooking SLA..."); for_cooking_SLA(param,storage1);}
   //else if(LApp){ for_cooking_LA(x_star_k,Qx_star_k,Qx,theta_star,storage2,storage3,param,mu,D); } //x_size = etastar.size(),
   
   for(size_t j=0; j<theta_size;j++) thetas(i,j) = get_theta[j];
   if((*param).id_bowl==0) weights[i] = *get_y;
   
}

void CCD_Stratetgy_shelf3(col_vector &log_weights, col_vector &weights){

   weights = -weights + log_weights;
   weights = weights - weights[0];
   weights = blaze::exp(weights);
   weights = weights/sum(weights);
}

void cooking_table(col_vector &theta_star, col_vector &eta_star, ptr_bowl &param)
{
   size_t theta_size = (*param).theta_size;
   row_matrix hessian(theta_size,theta_size);
   (*param).optim->hess_tick = true;

   bool OK = true;
   col_vector eigenvalues((*param).theta_size,0.0); row_matrix eigenvectors(theta_size, theta_size,0.0); 
   if((*param).optim->smartGrad && theta_size > 1) smart_hessian_theta_given_y(opt_fun_p_theta_given_y,topt_fun_p_theta_given_y,theta_star,hessian, param, eta_star,false);
   else hessian_theta_given_y(opt_fun_p_theta_given_y,theta_star,hessian, param, eta_star,false);      
   hessian = - hessian; myeign(hessian,eigenvalues,eigenvectors); //eigenvectors = - eigenvectors;
   for(size_t p=0;p<eigenvalues.size();p++) if(eigenvalues[p]<0) OK =false;

   if(!OK){
      if((*param).optim->smartGrad && theta_size > 1) smart_hessian_theta_given_y(opt_fun_p_theta_given_y,topt_fun_p_theta_given_y,theta_star,hessian, param, eta_star,true);
      else hessian_theta_given_y(opt_fun_p_theta_given_y,theta_star,hessian, param, eta_star,true);       
      hessian = - hessian; myeign(hessian,eigenvalues,eigenvectors);  //eigenvectors = - eigenvectors;
   } OK =true;
   
   for(size_t p=0;p<eigenvalues.size();p++) if(eigenvalues[p]<0) OK =false;
   if(!OK) print("Hessian is not well estimated!");


   std::cout <<  std::endl;
   std::cout << "11. " << "Hessian: " << std::endl;
   std::cout << hessian << std::endl;

   std::cout << "12. " << "Eigenvalues: " << std::endl;
   std::cout << eigenvalues << std::endl;
   std::cout << "13. " << "Eigenvectors: " << std::endl;
   std::cout << eigenvectors << std::endl;
   ptr ldens_theta_star{new double{0.0}};
   (*param).optim->get_ldens_theta_star_equal(*ldens_theta_star);   
   //(*param).optim->hess_tick = false;

   col_vector stdev_corr_pos((*param).theta_size,0.0),stdev_corr_neg((*param).theta_size,0.0);
   row_matrix Sigma = inv(hessian);
   stdev_corr(opt_fun_p_theta_given_y,Sigma,param,stdev_corr_pos,stdev_corr_neg,theta_star,ldens_theta_star,eigenvalues,eigenvectors,eta_star);

   std::cout << "corrections: " << std::endl;
   for(size_t i =0; i< theta_star.size();i++)
      std::cout <<"negative: " << stdev_corr_neg[i] << ", positive: " << stdev_corr_pos[i] << std::endl;

   p_thetaj_given_y(theta_star,Sigma,eigenvalues, eigenvectors,stdev_corr_pos,stdev_corr_neg);

   (*param).set_utensils_x();
   //Search Strategies

   if((*param).Model=="Gaussian") {(*param).correction->Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
   }else if((*param).Model=="Poisson") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);
   }else if((*param).Model=="Binomial") {(*param).correction->Non_Gaussian_case((*param).A, (*param).x_size ,(*param).invQx_theta);}

   //print(trans(eta_star));
   boost::scoped_ptr<storage_type1> storage1{new storage_type1{(*param).x_size,(*param).theta_size}};
   boost::scoped_array<storage_type2> storage2{new storage_type2[(*param).x_size]}; 
   boost::scoped_ptr<storage_type3> storage3{new storage_type3{(*param).x_size,(*param).theta_size}};

   col_vector weights; row_matrix thetas;  
   if((*param).theta_size==1) {}//{Grid_Stratetgy_1D(param,theta_star,x_star,ldens_theta_star,eigenvalues,eigenvectors,weights,thetas);}//Grid_Stratetgy_1D(theta_star,eigenvalues,eigenvectors,weights,thetas,x_size,Qx,Qstar,xstar,mu,grad_x_like,b_vec,x_initial,D,y);
   else if((*param).theta_size==2) {} //Grid_Stratetgy_2D(theta_star,eigenvalues,eigenvectors,weights,thetas,x_size,Qx,Qstar,xstar,mu,grad_x_like,b_vec,x_initial,D,y);
   else if((*param).theta_size>=3) CCD_Stratetgy(eta_star,param,thetas,weights,stdev_corr_pos,stdev_corr_neg,theta_star,eigenvalues,eigenvectors,storage1,storage2,storage3);


   compute_marginal_likelihood(param, eigenvalues, weights,stdev_corr_pos,stdev_corr_neg);
  compute_DIC(param,weights,storage1,thetas,eta_star,theta_star);


   //print(thetas);
}

void compute_marginal_likelihood(ptr_bowl &param, col_vector &eigen_values, col_vector &weights, col_vector & stdev_corr_pos, col_vector & stdev_corr_neg){

   double log_dens_mode = 0.0; (*param).optim->get_ldens_theta_star_equal(log_dens_mode);
   double mlga = 0.0, mli = 0.0; //marginal_likelihood_gaussian_approx

   //we need to add
  

   //print(log_dens_mode);


	if ((*param).theta_size > 0) {
			mlga = 0.5 * (*param).theta_size * log(2.0 * M_PI) + log_dens_mode;
			for (size_t i = 0; i < (*param).theta_size; i++) mlga -= 0.5 * log(eigen_values[i]);
			

			if ((*param).theta_size>=3) {
				mli = 0.5 * (*param).theta_size * log(2.0 * M_PI) + log_dens_mode;
				for (size_t i = 0; i < (*param).theta_size; i++) {
					mli -= 0.5 * (log(eigen_values[i]) + 0.5 * (log(SQR(stdev_corr_pos[i])) + log(SQR(stdev_corr_neg[i]))));
				}
			} else { //Grid Strategy  
				/*double integral = 0.0, log_jacobian = 0.0;

				for (j = 0; j < weights.size(); j++) {
					integral += weights[j];
				}

				integral *= ai_par->dz;
				for (i = 0; i < (*param).theta_size; i++) {
					log_jacobian -= 0.5 * log(eigen_values[i]);
				}
				mli = log(integral) + log_jacobian + log_dens_mode;*/
			}
			
		} else {
			mlga = log_dens_mode;
			mli = log_dens_mode;
		}
	
   print("");
   std::cout << "Marginal Likelihood:" << std::endl;
   std::cout << "  -  Non - GA: " << mlga << std::endl;
   std::cout << "  -  GA      : " << mli << std::endl;
}

void compute_DIC(ptr_bowl &param, col_vector &weights,boost::scoped_ptr<storage_type1> &GA_storage, row_matrix &thetas,col_vector &eta_star,col_vector &theta_star)
{
   if(false){ //simpsons-rule
   auto xstar = GA_storage->means_eta.begin();
   size_t etasize = (*xstar).size();

   col_matrix mu(etasize,weights.size()),sd(etasize,weights.size());

   col_vector adj_weights = weights/sum(weights);
   size_t k = 0;
   for (auto ga_xi_sd = GA_storage->sds_eta.begin(); xstar != GA_storage->means_eta.end(); xstar++,ga_xi_sd++)
   {

      blaze::column(mu, k) = (*xstar);
      blaze::column(sd, k) = (*ga_xi_sd);
      k++;
   }
   
   int np = 80;
   col_vector std_x_points = linspace<columnVector>(np, -10, 10), loglike(np,0.0), DIC_eta(etasize,0.0), x_points(np,0.0);
   static const double log_norm_const_gaussian = -0.918938533204672741780329736407;	// log(1.0/sqrt(2.0*M_PI)) 
   static const double norm_const_gaussian = -0.918938533204672741780329736407;	// log(1.0/sqrt(2.0*M_PI)) 

   //print(std_x_points);
   double integral_one = 0.0, w[2] = { 4.0, 2.0 }; col_vector theta((*param).theta_size,0.0);
   col_vector mean_deviance(etasize,0.0), devaiance_mean(etasize,0.0);

   col_vector eta_star(etasize,0.0);
   //#pragma omp parallel for shared(mu,sd,log_norm_const_gaussian,std_x_points, etasize)
   for(size_t ind = 0; ind <etasize ;ind++)
   {
      col_vector weighted_DIC(weights.size(),0.0);
      for(size_t k = 0; k <weights.size() ;k++)
      {
         //print(mu(ind,k));
         //print(sd(ind,k));
         //print((*param).y_response[ind]);
         x_points = mu(ind,k) + sd(ind,k)*std_x_points;
         col_vector dens = log_norm_const_gaussian - 0.5 * (std_x_points * std_x_points);
         dens = (1.0/sd(ind,k)) * blaze::exp(dens);

         //print(dens);

         for(size_t j=0; j<(*param).theta_size;j++) theta[j] = thetas(k,j);
         loglikelihood_fun(param,ind,x_points,loglike, theta);

         if(k==0){
            
            eta_star[ind] = mu(ind,k);
            col_vector v1(1,mu(ind,k));
            col_vector v2(1,0.0);
            loglikelihood_fun(param,ind,v1,v2, theta);
            devaiance_mean[ind] = -2*v2[0];//*pi_eta;
         }

         //print(trans(loglike));

         col_vector int_fun(loglike.size(),0.0);
         int_fun[0] = loglike[0] * dens[0];
         int_fun[np-1] = loglike[np - 1] * dens[np - 1];
         integral_one = dens[0] + dens[np - 1];
         for (size_t i = 1, k = 0; i < np - 1; i++, k = (k + 1) % 2) {
            //print(k);
            int_fun[i] = w[k] * loglike[i] * dens[i];
            integral_one += w[k] * dens[i];
         }

         int_fun.scale(1.0/integral_one);
         weighted_DIC[k] = -2*blaze::sum(int_fun)*weights[k];
      }

      mean_deviance[ind] = blaze::sum(weighted_DIC);

   }

   DIC_eta = 2*mean_deviance - devaiance_mean;

   print("");
   std::cout << "Deviance:" << std::endl;
   std::cout << "  -  Mean of Deviance              : " << blaze::sum(mean_deviance) << std::endl;
   std::cout << "  -  Deviance of Mean              : " << blaze::sum(devaiance_mean) << std::endl;
   std::cout << "  -  Effective Number of Paramaters: " << blaze::sum(mean_deviance - devaiance_mean) << std::endl;
   std::cout << "  -  DIC: " << sum(DIC_eta) << std::endl;             



   }else{ //Gauss–Hermite quadrature



   ///=============================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


   auto xstar = GA_storage->means_eta.begin();
   size_t etasize = (*xstar).size();

   col_matrix mu(etasize,weights.size()),sd(etasize,weights.size());
   col_vector adj_weights = weights/sum(weights);

   size_t k = 0;
   for (auto ga_xi_sd = GA_storage->sds_eta.begin(); xstar != GA_storage->means_eta.end(); xstar++,ga_xi_sd++)
   {
      blaze::column(mu, k) = (*xstar);
      blaze::column(sd, k) = (*ga_xi_sd);
      k++;
   }

   //col_vector wp{0.000000001522476, 0.000001059115548, 0.000100004441232, 0.002778068842913, 0.030780033872546, 0.158488915795936, 0.412028687498898, 0.564100308726418, 
   //0.412028687498898, 0.158488915795935, 0.030780033872546, 0.002778068842913, 0.000100004441232, 0.000001059115548, 0.000000001522476};
   col_vector xp{-4.4999907073093909914974, -3.6699503734044540692594, -2.9671669279056027690444, -2.3257324861738579713233, -1.7199925751864888479048, -1.1361155852109203756584, -0.5650695832555758801874, -0.0000000000000003841438,
   0.5650695832555755471205,  1.1361155852109199315692,  1.7199925751864892919940,  2.3257324861738579713233, 2.9671669279056027690444,  3.6699503734044527369917,  4.4999907073093909914974};
   //wp = (1/sqrt(M_PI))*wp;

   col_vector wp{8.5896510040145377955472953327006533e-10,
               5.9754195995507364236424023215543677e-07,
               5.6421464051608148746342480395199459e-05,
               0.0015673575035500826157369713698130909,
               0.017365774492137560358617776046230574,
               0.089417795399844540726874697611492593,
               0.23246229360973186262029344106849749,
               0.31825951825951853679796954565972555,
               0.23246229360973186262029344106849749,
               0.089417795399843971737574577218765626,
               0.017365774492137560358617776046230574,
               0.0015673575035500826157369713698130909,
               5.6421464051608148746342480395199459e-05,
               5.9754195995507364236424023215543677e-07,
               8.5896510040145377955472953327006533e-10};

   double value = 0, eval_pt = 0.0;
   int np = 15;
   col_vector mean_deviance(etasize,0.0), devaiance_mean(etasize,0.0), theta((*param).theta_size,0.0), DIC_eta(etasize,0.0);

   col_vector eta_star(etasize,0.0);
   #pragma omp parallel for shared(mean_deviance,devaiance_mean) //I used serial inside loglikelihood_fun
   for(size_t ind = 0; ind <etasize ;ind++)
   {
 
      col_vector loglike(np,0.0), eval_pts(np,0.0);
      col_vector weighted_DIC(weights.size(),0.0);
      size_t k = 0;
      for(k = 0; k <weights.size() ;k++)
      {
         eval_pts = blaze::serial(sqrt(2)*xp*sd(ind,k) + mu(ind,k));
         loglikelihood_fun(param,ind,eval_pts,loglike, theta);
         weighted_DIC[k] = blaze::serial(sum(-2*wp*loglike)*weights[k]);

         if(k==0){
            eta_star[ind] = mu(ind,k);
            col_vector v1(1,mu(ind,k));
            col_vector v2(1,0.0);
            loglikelihood_fun(param,ind,v1,v2, theta);
            devaiance_mean[ind] = -2*v2[0];//*pi_eta;
         } 
      }

      //std::cout << ind << " ----- " << omp_get_thread_num() << std::endl;
      mean_deviance[ind] = blaze::sum(weighted_DIC);
   }


   DIC_eta = 2*mean_deviance - devaiance_mean;

   //export_vector(eta_star,"eta_star.txt");

   print("");
   std::cout << "Deviance:" << std::endl;
   std::cout << "  -  Mean of Deviance              : " << blaze::sum(mean_deviance) << std::endl;
   std::cout << "  -  Deviance of Mean              : " << blaze::sum(devaiance_mean) << std::endl;
   std::cout << "  -  Effective Number of Paramaters: " << blaze::sum(mean_deviance - devaiance_mean) << std::endl;
   std::cout << "  -  DIC                           : " << sum(DIC_eta) << std::endl;     
   print("");
  
   }
}

