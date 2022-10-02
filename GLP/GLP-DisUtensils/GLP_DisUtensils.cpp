
#ifndef _cGLPDisUtensils_
#define _cGLPDisUtensils_

#include "GLP_DisUtensils.h"


#define myfun_i_c_o1s3(fun,param,x,y,h,i,xstar)\
{\
  ptr yi0{new double{0.0}},yi1{new double{0.0}};\
  x[i] += (h);\
  fun(yi0,x,xstar,param);\
  x[i] -= 2*(h);\
  fun(yi1,x,xstar,param);\
  x[i] += (h);\
  *y = ((*yi0) - (*yi1))/(2*(h));\
}
  
#define myfun_ij_c_o1s3(fun,param,x,y,h,i,j,xstar)\
{\
  ptr yj0{new double{0.0}},yj1{new double{0.0}};\
  x[j] += (h);\
  myfun_i_c_o1s3(fun,param,x,yj0,h,i,xstar);\
  x[j] -= 2*(h);\
  myfun_i_c_o1s3(fun,param,x,yj1,h,i,xstar);\
  *y = ((*yj0) - (*yj1))/(2*(h));\
}

/*
#define trans_myfun_i_c_o1s3(fun1,fun2,param,x,y,h,i,xstar)\
{\
  ptr yi0{new double{0.0}},yi1{new double{0.0}};\
  x[i] += (h);\
  fun2(fun1,yi0,x,xstar,param);\
  x[i] -= 2*(h);\
  fun2(fun1,yi1,x,xstar,param);\
  x[i] += (h);\
  *y = ((*yi0) - (*yi1))/(2*(h));\
}*/

#define trans_myfun_i_f_o1s3(fun1,fun2,param,x,y,h,i,xstar)\
{\
  ptr yi0{new double{0.0}},yi1{new double{0.0}};\
  x[i] += (h);\
  fun2(fun1,yi0,x,xstar,param);\
  x[i] -= (h);\
  fun2(fun1,yi1,x,xstar,param);\
  x[i] -= (h);\
  *y = ((*yi0) - (*yi1))/(h);\
}

#define trans_myfun_ij_f_o1s3(fun1,fun2,param,x,y,h,i,j,xstar)\
{\
  ptr yj0{new double{0.0}},yj1{new double{0.0}};\
  x[j] += (h);\
  trans_myfun_i_f_o1s3(fun1,fun2,param,x,yj0,h,i,xstar);\
  x[j] -= (h);\
  trans_myfun_i_f_o1s3(fun1,fun2,param,x,yj1,h,i,xstar);\
  *y = ((*yj0) - (*yj1))/(h);\
}

#define trans_myfun_i_c_o1s3(fun1,fun2,param,x,y,h,i,xstar)\
{\
  ptr yi0{new double{0.0}},yi1{new double{0.0}};\
  x[i] += (h);\
  fun2(fun1,yi0,x,xstar,param);\
  x[i] -= 2*(h);\
  fun2(fun1,yi1,x,xstar,param);\
  x[i] += (h);\
  *y = ((*yi0) - (*yi1))/(2*(h));\
}

#define trans_myfun_ij_c_o1s3(fun1,fun2,param,x,y,h,i,j,xstar)\
{\
  ptr yj0{new double{0.0}},yj1{new double{0.0}};\
  x[j] += (h);\
  trans_myfun_i_c_o1s3(fun1,fun2,param,x,yj0,h,i,xstar);\
  x[j] -= 2*(h);\
  trans_myfun_i_c_o1s3(fun1,fun2,param,x,yj1,h,i,xstar);\
  x[j] += (h);\
  *y = ((*yj0) - (*yj1))/(2*(h));\
}
//(ptr &fx,col_vector &theta, col_vector &x, ptr_bowl &param);

void ldmvnorm(ptr &fx, col_vector &x,col_vector &mu, sym_matrix &Q)
{
  col_vector xmu = x - mu;
  *fx = -(static_cast<double>(xmu.size())/2.0)*std::log(2*M_PI) + 0.5*log_det(Q) - 0.5*(trans(xmu) * (trans(Q) * xmu));
}

void loglikelihood_fun(ptr_bowl &param, size_t &ind, col_vector &eta,col_vector &loglike, col_vector &theta) 
{


   if((*param).Model=="Gaussian"){

      size_t indexOftheta; boost::scoped_ptr<double> sd{new double{0.0}};
      auto get_it = (*param).get_from_Bowl("C_Gaussian_Noise");

      if(!isnan(get_it)) *sd = std::exp(-0.5*get_it); 
      else 
      {
         indexOftheta = (size_t)((*param).get_from_Bowl("R_Gaussian_Noise"));
         *sd = std::exp(-0.5*theta[indexOftheta]); 
      }
      loglike = blaze::serial( -(0.5)*((*param).A.rows())*std::log(2*M_PI) - 0.5*((*param).A.rows())*std::log((*sd)*(*sd)) - (0.5/((*sd)*(*sd)))*(blaze::pow((*param).y_response[ind] - eta,2)) );


   }else if((*param).Model=="Poisson") {

      loglike = blaze::serial( - (*param).Ntrials[ind]*blaze::exp(eta) + (*param).y_response[ind] * (eta) - logfactorial((*param).y_response[ind]) + (*param).y_response[ind]*blaze::log((*param).Ntrials[ind]));


   }else if((*param).Model=="Binomial") {

      col_vector eAx = blaze::serial(blaze::exp(eta));
      col_vector p = blaze::serial(eAx/(1+eAx));

      if(!(*param).y_response[ind]==0)  loglike = blaze::serial((*param).y_response[ind]*blaze::log(p) + ((*param).Ntrials[ind] - (*param).y_response[ind])*blaze::log(1-p));
      else loglike = 0.0;
   }
   

}

void like_Poisson_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param) 
{  
   *fx = - blaze::sum((*param).Ntrials*blaze::exp(eta)); //offset
   *fx += blaze::sum((*param).y_response * (log((*param).Ntrials))) + blaze::sum((*param).y_response * (eta)) + (*param).addnormcnst;
}

void like_Gaussian_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param) 
{
   size_t indexOftheta; boost::scoped_ptr<double> sd{new double{0.0}};
   auto get_it = (*param).get_from_Bowl("C_Gaussian_Noise");
   //print(get_it);

   if(!isnan(get_it)) *sd = std::exp(-0.5*get_it); 
   else 
   {
      indexOftheta = (size_t)((*param).get_from_Bowl("R_Gaussian_Noise"));
      *sd = std::exp(-0.5*theta[indexOftheta]); 
   }
   *fx = -(0.5)*((*param).A.rows())*std::log(2*M_PI) - 0.5*((*param).A.rows())*std::log((*sd)*(*sd)) - (0.5/((*sd)*(*sd)))*sum(blaze::pow((*param).y_response - eta,2)) ;
}

void like_Binomial_eta(ptr &fx, col_vector &eta,col_vector &theta, ptr_bowl &param) 
{
   double bin_size = (*param).get_from_Bowl("C_Binomial_Size");
   col_vector eAx = blaze::exp(eta);
   col_vector p = eAx/(1+eAx);

   for(size_t i=0; i<p.size();i++){
      if(!(*param).y_response[i]==0)  *fx += (*param).y_response[i]*blaze::log(p[i]) + ((*param).Ntrials[i] - (*param).y_response[i])*blaze::log(1-p[i]);// + fac;
   }
   
   //*fx = blaze::sum((*param).y_response*blaze::log(p) + ((*param).Ntrials - (*param).y_response)*blaze::log(1-p));// + fac;
}

bool check_theta(col_vector &theta){

   for(size_t i=0;i<theta.size();i++){
      if(theta[i]>10) return true;
   }

   return false;

}


void hessian_theta_given_y(fun_type7 fun, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central)
{
   (*param).optim->smartGrad = false;
   boost::scoped_ptr<double> h{new double{sqrt((*param).optim->grad_stepsize)}};
   boost::scoped_ptr<int> num_threads{new int{}};   
   //*num_threads = (( omp_get_num_procs() > omp_get_max_threads()) ? omp_get_max_threads() : omp_get_num_procs());

   //if(blaze::getNumThreads()>(*param).theta_size) *num_threads = (*param).theta_size;
   //else *num_threads = blaze::getNumThreads();
   
   size_t i,j;
   //#pragma omp parallel for private(i) num_threads(*num_threads) schedule(dynamic,1)
   for(i=0; i<x.size(); i++)
   {
      size_t j;
      for(j=i+1; j<x.size(); j++)
      {
         col_vector xcopy1(x);   
         ptr y{new double{}};   

         if(central){
            myfun_ij_c_o1s3(fun,param,xcopy1,y,*h,i,j,xstar);
         }else{
            col_vector val(4,0);
            fun(y,xcopy1,xstar,param);
            val[0] = *y; //i,j 
            //std::cout << std::setprecision(6) << "i " << i << " and j:" << j << "(i,j) " << *y << std::endl;
            xcopy1[i] += *h;
            fun(y,xcopy1,xstar,param);
            //std::cout << std::setprecision(6) << "i " << i << " and j:" << j << "(i+h,j)" << *y << std::endl;
            val[1] = *y; //i+h,j
            xcopy1[j] += *h;
            fun(y,xcopy1,xstar,param);
            //std::cout << std::setprecision(6) << "i " << i << " and j:" << j << "(i+h,j+h)" << *y << std::endl;
            val[2] = *y; //i+h,j+h
            xcopy1[i] -= *h; 
            fun(y,xcopy1,xstar,param);
            //std::cout << std::setprecision(6) << "i " << i << " and j:" << j << "(i,j+h)" << *y << std::endl;
            val[3] = *y; //i,j+h
            *y = (val[2]-val[3]-val[1]+val[0])/((*h)*(*h));
            //std::cout << "h: " << ((*h)*(*h)) << std::endl;

         }

         m(i,j) = m(j,i) =  -(*y);
      }
      col_vector xcopy2(x);   
      boost::scoped_ptr<double> y{new double{0.0}};   
      myfun_ij_c_o1s3(fun,param,xcopy2,y,*h,i,i,xstar);
      m(i,i) = -(*y);
   }  
}

void trans_hessian_theta_given_y(fun_type7 fun1,fun_type8 fun2, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central)
{
   //(*param).optim->smartGrad = false;
   boost::scoped_ptr<double> h{new double{sqrt((*param).optim->grad_stepsize)}};
   //boost::scoped_ptr<int> num_threads{new int{}};   
   //*num_threads = (( omp_get_num_procs() > omp_get_max_threads()) ? omp_get_max_threads() : omp_get_num_procs());

   size_t i,j;
   //#pragma omp parallel for private(i) num_threads(1)//*num_threads) schedule(dynamic,1)
   for(i=0; i<x.size(); i++)
   {
      size_t j;
      for(j=i+1; j<x.size(); j++)
      {
         //print("sasassaas");
         col_vector xcopy1(x);   
         ptr y{new double{}};  

         if(central){
            trans_myfun_ij_c_o1s3(fun1,fun2,param,xcopy1,y,*h,i,j,xstar);
         }else{
            trans_myfun_ij_f_o1s3(fun1,fun2,param,xcopy1,y,*h,i,j,xstar);
         }

         m(i,j) = m(j,i) =  -(*y);
      }
      col_vector xcopy2(x);   
      boost::scoped_ptr<double> y{new double{0.0}};
      if(central){trans_myfun_ij_c_o1s3(fun1,fun2,param,xcopy2,y,*h,i,i,xstar);
      }else{trans_myfun_ij_f_o1s3(fun1,fun2,param,xcopy2,y,*h,i,i,xstar);}
      m(i,i) = -(*y);

      //print(m);
   }  
}

void smart_hessian_theta_given_y(fun_type7 fun1,fun_type8 fun2, col_vector &x, row_matrix &m, ptr_bowl &param, col_vector &xstar, bool central)
{
   (*param).optim->curr_x = x;
   col_vector zero_theta(x.size(),0);
   
   trans_hessian_theta_given_y(fun1,fun2,zero_theta,m,param,xstar,central);
   m = solve(trans((*param).optim->G),m)*trans((*param).optim->G);
}

void optimize_p_theta_given_y(opt_fun2 fun, col_vector &thetastar,col_vector &etastar, ptr_bowl &param)
{  
   
   bool switch_optimizer = false;
   
   //#constrained
   

   const int n = thetastar.size();
   col_vector initialvalue(n);
   get_txt_column("././Data/initial.txt",initialvalue);
   bool opt_satisfied;
   get_txt_bool("././Data/safemode.txt",opt_satisfied);
   (*param).optim->opt_satisfied = !opt_satisfied;

   //#unconstrained
   //#constrained

   

   //other options
   //LBFGSSolver<double> solver(param_opt);
   //LBFGSSolver<double, LineSearchBracketing> solver(param_opt);

   //VectorXd x = VectorXd::Zero(n);
   std::cout << "-> Theta Initial Value: " << trans(initialvalue) << std::endl;
   VectorXd th = VectorXd::Ones(n);
   for(size_t i = 0; i < n; i++)
      th[i] = initialvalue[i];   

   double fx=0.0; int niter, all_iter =0;
   bool tryagain = true, increasemaxlinesearch = false;
   while(tryagain)
   {
      try{
         try{ 
            if((*param).optim->peppers==0) std::cout << "Optimization Starts" << std::endl;
            else std::cout << "\nRestart Optimization" << std::endl;
            
            if(switch_optimizer){
               
               LBFGSParam<double> param_opt;

               if((*param).optim->peppers>0) param_opt.max_linesearch = 5;
               if(increasemaxlinesearch) param_opt.max_linesearch = 20; 
               param_opt.epsilon = 0.005;
               param_opt.max_iterations = 100;
               LBFGSSolver<double,LineSearchBacktracking> solver(param_opt);

               niter = solver.minimize(fun, th, fx, param, etastar);
               all_iter += niter;
               tryagain = false;

            }else{

               LBFGSBParam<double> param_opt;  // New parameter class

               if((*param).optim->peppers>0) param_opt.max_linesearch = 5;
               if(increasemaxlinesearch) param_opt.max_linesearch = 20; 

               param_opt.epsilon = 0.005;
               param_opt.max_iterations = 100;
               LBFGSBSolver<double> solver(param_opt);  // New solver class

               VectorXd lb = VectorXd::Constant(n, -14);
               VectorXd ub = VectorXd::Constant(n, 14);
               niter = solver.minimize(fun, th, fx, lb, ub, param, etastar);
               all_iter += niter;
               tryagain = false;
               if(niter==100)
               {
                  for(size_t i = 0; i < n; i++) th[i] = th[i] = -4 + rand() % 4; 
                  print("No convergence using the current initial theta");
                  print("Initial value is reset! - v0");
                  tryagain = true;
               }
            }

      
         } 
         
         catch (int myNum) {
            
            if(myNum==0){
               
               fx = (*param).optim->f_old;
               for(size_t i = 0; i < n; i++) th[i] = (*param).optim->theta_old[i];
               (*param).optim->grad_stepsize = 0.001;
               (*param).optim->central = true;
               tryagain = true;
               (*param).optim->peppers += 1;

            }else if(myNum==1){
               
               fx = (*param).optim->f_old;
               for(size_t i = 0; i < n; i++) th[i] = (*param).optim->theta_old[i];
               (*param).optim->central = true;
               (*param).optim->grad_stepsize = 0.001;
               tryagain = true;
               (*param).optim->peppers += 1;
               
            }else if(myNum==2){
               
               fx = (*param).optim->f_old;
               for(size_t i = 0; i < n; i++) th[i] = (*param).optim->theta_old[i];
               (*param).optim->central = false;
               (*param).optim->grad_stepsize = 0.01;
               tryagain = true;
               (*param).optim->peppers += 1;

            }else if(myNum==3){

               bool switch_optimizer = true;
               tryagain = true;

            }else if(myNum==4){

               fx = (*param).optim->f_old;
               for(size_t i = 0; i < n; i++) th[i] = (*param).optim->theta_old[i];
               print("Cannot add more peppers");
               tryagain = true;
            }
            
         }}

         catch(const char *error) //catch (const std::exception &ex)
         {
            print(error);
            print("Initial value is reset!");
            increasemaxlinesearch = true;
            for(size_t i = 0; i < n; i++)
               th[i] = th[i] = -10 + rand() % 15;
         }
         catch (const std::exception &ex) 
         {
            print("Best optimum reached!");
            tryagain = false;
            //std::cout << ex.what() << std::endl; //maximum line search is reached
         }
      
      
   }  

   //fx += (*param).max_subtract;
   //(*param).optim->n_iter = all_iter;  //better to do it internally 

   (*param).optim->f_old = fx;
   for(size_t i = 0; i < n; i++) (*param).optim->theta_old[i] = th[i];

   std::cout << "\nOptimization: " << std::endl;
   std::cout << "8. " << (*param).optim->n_iter << " iterations" << std::endl;
   std::cout << "9. " << "theta star is " << th.transpose() << std::endl;
   std::cout << "10. " << "f(x) = " << - fx << std::endl;

   fx = - fx;
   (*param).optim->set_ldens_theta_star_equal(fx);

   for(size_t i=0;i <thetastar.size();i++)
      thetastar[i] = th[i];

   (*param).correction->eta_star = (*param).update->update_eta_star;

}


void update_invQx_by_theta(col_vector &theta, ptr_bowl &param)
  {  
    size_t th = 0;
    for (size_t i = 0; i < (*param).theta_size_Qx; i++)
      if ((*param).prior_blocks[i]==1)
      {
        if((*param).effi[i]=="bym2"){
          if((*param).xpts[0+i]!=-1) {
          blaze::IdentityMatrix<double, blaze::rowMajor> I((*param).xpts[0 + i]);
          double tau = exp(theta[th]);
          double phi = exp(theta[th+1])/ (1.0 + exp(theta[th+1]));
          if((*param).xpts[0+i]!=-1) blaze::submatrix((*param).invQx_theta,(*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]) = (1/tau)*( (1-phi)*I + (phi)*blaze::submatrix((*param).invQx_fixed, (*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i])); 
          th += 2;
          }
        } else if((*param).effi[i]=="iid"){
          if((*param).xpts[0+i]!=-1) {blaze::band(blaze::submatrix((*param).invQx_theta,(*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]),0) = exp(-theta[th])*blaze::band(blaze::submatrix((*param).invQx_fixed, (*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]),0);th++;}
        } else {
          if((*param).xpts[0+i]!=-1) {blaze::submatrix((*param).invQx_theta,(*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]) = exp(-theta[th])*blaze::submatrix((*param).invQx_fixed, (*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]);th++;}
        }
      } 

  }

  void update_Qx_by_theta(col_vector &theta, ptr_bowl &param)
  {  
    size_t th = 0;
    for (size_t i = 0; i < (*param).theta_size_Qx; i++)
      if ((*param).prior_blocks[i]==1)
      {
        if((*param).effi[i]=="bym2"){
          if((*param).xpts[0+i]!=-1) {
          blaze::IdentityMatrix<double, blaze::rowMajor> I((*param).xpts[0 + i]);
          double tau = exp(theta[th]);
          double phi = exp(theta[th+1])/ (1.0 + exp(theta[th+1]));
          //update later
          //if(xpts[0+i]!=-1) blaze::submatrix(invQx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]) = (1/tau)*( (1-phi)*I + (phi)*blaze::submatrix(invQx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i])); 
          th += 2;
          }
        } else if((*param).effi[i]=="iid"){
          if((*param).xpts[0+i]!=-1) {blaze::band(blaze::submatrix((*param).Qx_theta,(*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]),0) = exp(theta[th])*blaze::band(blaze::submatrix((*param).Qx_fixed, (*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]),0);th++;}
        } else {
          if((*param).xpts[0+i]!=-1) {blaze::submatrix((*param).Qx_theta,(*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]) = exp(theta[th])*blaze::submatrix((*param).Qx_fixed, (*param).xpts[2*(*param).theta_size_Qx+i],(*param).xpts[2*(*param).theta_size_Qx+i], (*param).xpts[i], (*param).xpts[i]);th++;}
        }
      } 

  }



#endif