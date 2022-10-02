
#ifndef _GLPlibraries_
#define _GLPlibraries_

//std library
#include <math.h>
#include <limits>
#include <cmath>
#include <random>
//needed for files
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include "GLP_splines.h"

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>


#include <blaze/Math.h>

using col_vector = blaze::DynamicVector<double>;

#include <random>

//Blaze:
#include <blaze/Blaze.h>
#include <blaze/Math.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/SymmetricMatrix.h>
#include <blaze/math/Rows.h>
#include <blaze/math/Columns.h>
#include <blaze/math/IdentityMatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/Row.h>
#include <blaze/math/Column.h>
#include <blaze/config/Vectorization.h>
#include <blaze/config/BLAS.h>
#include <blaze/config/Config.h>
#include <blaze/config/Thresholds.h>
#include <blaze/config/SMP.h>
#include <blaze/math/LowerMatrix.h>
#include <blaze/math/Submatrix.h>

//shortcuts for blaze
using col_vector = blaze::DynamicVector<double>;
using col_vector_D = blaze::CompressedVector<double, blaze::columnVector>;
using col_matrix = blaze::DynamicMatrix<double, blaze::columnMajor>;
using row_vector = blaze::DynamicVector<double, blaze::rowVector>;
using row_matrix = blaze::DynamicMatrix<double, blaze::rowMajor>;
using int_row_matrix = blaze::DynamicMatrix<int, blaze::rowMajor>;
using sym_matrix = blaze::SymmetricMatrix<blaze::DynamicMatrix<double, blaze::rowMajor>>;
using sym_matrix2 = blaze::SymmetricMatrix<blaze::CompressedMatrix<double, blaze::columnMajor>>;
using low_matrix = blaze::LowerMatrix<blaze::DynamicMatrix<double, blaze::rowMajor>>;

using blaze::columnVector;
using blaze::DiagonalMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using blaze::generate;
using blaze::linspace;
using blaze::rowMajor;
using blaze::SymmetricMatrix;

//Boost
#include <boost/container/vector.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_set.hpp>
#include <boost/ptr_container/indirect_fun.hpp>
#include <set>
#include <memory>
#include <functional>
#include <boost/foreach.hpp>
#include <array>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/special_functions/binomial.hpp>

//shortcuts for boost
using ptr = boost::scoped_ptr<double>;
using array_ptr = boost::scoped_array<double>;

#define SQR(x) ((x)*(x))

//SMARTGRAD

typedef double (*prior_fun)(double &theta, double p1, double p2);

struct storage_type1
{
   boost::ptr_vector<col_vector> means;
   boost::ptr_vector<col_vector> sds;

   boost::ptr_vector<col_vector> means_eta;
   boost::ptr_vector<col_vector> sds_eta;

   boost::ptr_vector<col_vector> para1;
   boost::ptr_vector<col_vector> para2;
   boost::ptr_vector<col_vector> para3;
   boost::ptr_vector<double> weights;

   double sum_weights = 0.0;
   size_t x_size, theta_size;

   storage_type1(size_t p1,size_t p2) : x_size(p1), theta_size(p2) {};

};

struct storage_type3
{
   boost::ptr_vector<col_vector> means;
   boost::ptr_vector<col_vector> sds;
   size_t x_size, theta_size;

   storage_type3(size_t p1,size_t p2) : x_size(p1), theta_size(p2) {};

};

struct node
{
  string type;
  double value;
  node *next;
};

struct det_blocks
{
  DynamicVector<double,columnVector> evalues_eff;
  row_matrix V;
};

struct priors
{
  string type;
  double p1, p2;
  prior_fun f;
  boost::scoped_ptr<splines> Sp{new splines{}};

};

class optim_Bowl
{
  private:
    int n, count;

  public:
    optim_Bowl() {}
    col_vector theta_old,best_theta; double f_old, f_diff=999;
    bool smartGrad, central, hess_tick;
    int optfunctioncall = 0, num_threads = 1;
    double grad_stepsize = 0.005, eps = std::numeric_limits<double>::epsilon(), y =0;
    col_matrix G, diff_x;
    col_vector prev_x, curr_x;
    size_t n_iter = 0;
    boost::scoped_ptr<double> ldens_theta_star{new double{0.0}};
    int peppers = 0; bool opt_satisfied = true;
    col_vector save_grad;

    void set_ldens_theta_star_equal(double &val) {*ldens_theta_star = val;}
    void get_ldens_theta_star_equal(double &vval) {vval = *ldens_theta_star;}

    void set_G(size_t theta_size)
    {
      n = theta_size;  
      diff_x.resize(n,n); 
      diff_x = 0.0;
      band(diff_x,0) =1.0;

      G.resize(n,n); 
      G = diff_x;
      
      prev_x.resize(n); prev_x = 0;
      curr_x.resize(n); curr_x = 0;
      count = 0;
    }

    void construct(bool &ssmartGrad, bool &ccentral, double &ggrad_stepsize, int &nnum_threads, size_t theta_size)
    {
      smartGrad = ssmartGrad; if(smartGrad) set_G(theta_size);

      central = ccentral;
      grad_stepsize = ggrad_stepsize;
      num_threads = nnum_threads;
      *ldens_theta_star = 0.0;
      hess_tick = false;

      theta_old.resize(theta_size); best_theta.resize(theta_size);
      save_grad.resize(theta_size);
      theta_old = -10;
      f_old = 0.0;
    }

    void set_prev_x(col_vector &x)  {prev_x = x; count++;}
    void set_curr_x(col_vector &x)  {curr_x = x;}
    void scale(col_vector &x){  
        const double m = blaze::mean(x);
        const double sd = blaze::stddev(x); //0.0;
        x = (x - m)/sd;

        //for(size_t i=0;i<x.size();i++) sd += (x[i] - m)*(x[i] - m);
        //sd = sqrt(sd/(x.size()-1.0));
        //for(size_t i=0;i<x.size();i++)  x[i] = (x[i] - m)/sd;
    }
    void MGS_orthogonalization()
    {
      size_t i, j, k, m = G.columns();
      double r;

      col_vector q(m, 0.0);

      for (i = 0; i < m; i++)
      {

        for (r = 0.0, j = 0; j < m; j++)
          r += (G(j, i) * G(j, i));
        r = sqrt(r);
        for (j = 0; j < m; j++)
        {
          q[j] = G(j, i) / r;
          G(j, i) = q[j];
        }

        for (j = i + 1; j < m; j++)
        {
          for (r = 0, k = 0; k < m; k++)
            r += q[k] * G(k, j);
          for (k = 0; k < m; k++)
            G(k, j) = G(k, j) - r * q[k];
        }
      }

      //for (j = 0; j < m; j++)  G(j,0) = G(j,0) + eps;
      //std::cout << G << std::endl;
    }
    void update_G(col_vector &current_theta)  
    {
        //std::cout << "test here: " << std::endl;
        curr_x = current_theta;
        if(count>0)
        {
          size_t m = diff_x.columns();
          //std::cout <<  diff_x << std::endl;
          blaze::submatrix(diff_x, 0, 1, m, (m - 1)) = blaze::submatrix(diff_x, 0, 0, m, (m - 1));
          //std::cout << diff_x  << std::endl;
          col_vector xdiff = curr_x - prev_x;
          for(size_t ii=0; ii<m; ii++) xdiff[ii] = xdiff[ii] + 7e-8;

          //std::cout << curr_x << std::endl;
          //std::cout << prev_x << std::endl; 
          for(size_t i=0;i<m;i++) diff_x(i,0) = 0.0 ;

          //scale(xdiff);
          //          std::cout << diff_x  << std::endl;

          for(size_t i=0;i<m;i++) diff_x(i,0) = xdiff[i] ;
          //          std::cout << diff_x  << std::endl;

          G = diff_x;
        }
        
        //std::cout<< current_x <<std::endl;
        //std::cout<< prev_x <<std::endl;
    }

    void smart_update(col_vector &current_theta)  
    { 
      update_G(current_theta);
      MGS_orthogonalization();
      set_prev_x(current_theta);
    }
};

class Utensils_eta
{
  public:
    sym_matrix cov_eta,invQ_eta;
    col_vector b_eta, update_eta_star,eta_moon, exp_eta_star;
    blaze::DiagonalMatrix<blaze::DynamicMatrix<double>> D_Qlike_eta;
    DynamicVector<double,columnVector> w;
    DynamicMatrix<double,rowMajor> V;

    double y =0;
    Utensils_eta() {}
    void construct(size_t &y_size)
    {
      cov_eta.resize(y_size); invQ_eta.resize(y_size);  
      b_eta.resize(y_size); update_eta_star.resize(y_size); eta_moon.resize(y_size); exp_eta_star.resize(y_size);
      D_Qlike_eta.resize(y_size,y_size);

      w.resize(y_size);       // The vector for the real eigenvalues
      V.resize(y_size,y_size);
    }
    void set_eta_Qlike(double theta_value)
    {
      if (theta_value == 0)
        blaze::diagonal(D_Qlike_eta) = 1;
      else
        blaze::diagonal(D_Qlike_eta) = std::exp(theta_value) * 1;
    }
};

class Utensils
{
  
  public:
    col_vector x_star, b_vec, part_b_vec, eta_star, sd_eta_star;
    sym_matrix invQ, Qlike;
    row_matrix S1, S2;
    blaze::IdentityMatrix<double> I;
    blaze::DiagonalMatrix<blaze::DynamicMatrix<double>> D_Qlike_x;
    bool vb = false;

    Utensils() {}
    void construct(size_t &x_size,size_t &y_size)
    {
      x_star.resize(x_size); b_vec.resize(x_size); 
      invQ.resize(x_size); eta_star.resize(y_size);
      sd_eta_star.resize(y_size);
      /*Qx.resize(x_size);  
      grad_x_like.resize(x_size); b_vec.resize(x_size);  
      grad_x_like = 0.0; b_vec = 0.0;
      D.resize(y_size,y_size);
      y = 0.0; *ldens_theta_star = 0.0;
       update_x_star = 0.1;
      Ax.resize(D.rows());
      if(RD) invQx.resize(x_size);*/
    }

    void Gaussian_case(blaze::CompressedMatrix<double> &A,size_t &x_size, sym_matrix &invQx_theta){ 
      S1 = invQx_theta*trans(A)*A; I.resize(x_size);
      S2.resize(x_size,x_size);
      }

    void Non_Gaussian_case(blaze::CompressedMatrix<double> &A,size_t &x_size, sym_matrix &invQx_theta){ 
      S1 = invQx_theta*trans(A)*A; I.resize(x_size); 
      S2.resize(x_size,x_size);
      D_Qlike_x.resize(A.rows(),A.rows());
      }
};


class Bowl
{
  private:
    node *head, *tail;
    col_vector Qpts;
    size_t t_rankdef;
      
  //added
  public:
    
    //mpi
    int id_bowl,size_bowl, id_boss;
    int grad_workers, hess_workers, std_workers, margtheta_workers, ccd_workers, margx_workers, linesearch_workers;
    int num_tasks_grad, num_tasks_hess, num_tasks_std, num_tasks_margtheta, num_tasks_ccd, num_tasks_margx, num_tasks_linesearch;
    col_vector theta_star; row_matrix save_hihj; //, eta_star
    bool go_to_oven = false; size_t num_Con = 0;
    bool internalverbose = true, pcjoint = false;


    //
    col_vector logfac; double addnormcnst = 0.0;
    blaze::CompressedMatrix<double> A;//,AQ,QA;
    sym_matrix invQx_fixed, invQx_theta;
    sym_matrix Qx_fixed, Qx_theta;

    bool use_double_optim = false; 
    blaze::DynamicVector<string> effi, priors_types; //make it private?
    blaze::DynamicVector<int> rankdef; 
    blaze::DynamicVector<double> xpts;
    col_vector y_response, prior_blocks, Ntrials; //col_vector mu;
    size_t x_size, y_size, theta_size, theta_size_Qx,zcov = 0,x_mu=0, num_blocks;
    string Model, Qx_type, strategy;
    
    bool RD_system;
    double max_subtract;

    
    boost::scoped_ptr<optim_Bowl> optim{new optim_Bowl{}}; //optim
    boost::scoped_array<priors> pr{new priors[0]}; //priors
    boost::scoped_array<det_blocks> evalues_effs{new det_blocks[0]};

    //strategies
    bool GApp = true,LApp = false, SLApp = false, RTheta = true;
    //utensils:
    boost::scoped_ptr<Utensils> correction{new Utensils{}}; //x_param
    boost::scoped_ptr<Utensils_eta> update{new Utensils_eta{}}; //eta_param

    Bowl()
    {
      head = NULL;
      tail = NULL;
    }

    //from here
    void construct(string &MModel, string &QQx_type, string &sstrategy, blaze::CompressedMatrix<double> &AA, col_vector &yy_response, size_t &ttheta_size)
    {
      x_size = AA.columns();
      y_size = AA.rows();
      strategy = sstrategy;
      {
        if(strategy=="GA") {GApp = true; LApp = false; SLApp = false; if(id_bowl==0) std::cout << "1. Stratetgy is Gaussian" << std::endl;}
        else if(strategy=="LA") {GApp = false; LApp = true; SLApp = false; if(id_bowl==0) std::cout << "1. Stratetgy is Laplace" << std::endl;}
        else if(strategy=="SLA"){GApp = false; LApp = false; SLApp = true; if(id_bowl==0) std::cout << "1. Stratetgy is Simplified Laplace" << std::endl;}
      }
      Model = MModel;
      Qx_type = QQx_type;
      A = AA;
      //AQ.resize(y_size,x_size);
      //QA.resize(x_size,y_size);

      y_response = yy_response;
      theta_size = ttheta_size;
      pr.reset(new priors[theta_size]);
      //mu.resize(x_size); mu = 0;
    } 

    void set_utensils_x() {(*correction).construct(x_size,y_size);}
    void set_utensils_eta() {(*update).construct(y_size);}
    void set_Ntrials(col_vector &bNtrials) {Ntrials.resize(bNtrials.size()); Ntrials = bNtrials;}
    void set_xpts(blaze::DynamicVector<double> &xxpts) {xpts = xxpts;}
    void set_rankdef(blaze::DynamicVector<int> &rrankdef) {rankdef = rrankdef;}
    void resize_Qx_and_invQx() {invQx_fixed.resize(x_size); invQx_theta.resize(x_size); Qx_fixed.resize(x_size); Qx_theta.resize(x_size);}
    void add_invQx_fixed_to_Bowl(sym_matrix &getinvQxfixed) {invQx_fixed = getinvQxfixed; invQx_theta = getinvQxfixed;}
    void add_Qx_fixed_to_Bowl(sym_matrix &getQxfixed) {Qx_fixed = getQxfixed; Qx_theta = getQxfixed;}

    void add_eff_to_Bowl(blaze::DynamicVector<string> &eeffi) {effi.resize(eeffi.size()); effi = eeffi;}
    double get_eff_from_Bowl(string s){ for(size_t i =0; i< effi.size(); i++) if(s==effi[i]) return (double)i; return std::numeric_limits<double>::quiet_NaN();}

  /*void update_invQx_by_theta(col_vector &theta)
  {  
    size_t th = 0;
    for (size_t i = 0; i < theta_size_Qx; i++)
      if (prior_blocks[i]==1)
      {
        if(effi[i]=="bym2"){
          if(xpts[0+i]!=-1) {
          blaze::IdentityMatrix<double, blaze::rowMajor> I(xpts[0 + i]);
          double tau = exp(theta[th]);
          double phi = exp(theta[th+1])/ (1.0 + exp(theta[th+1]));
          if(xpts[0+i]!=-1) blaze::submatrix(invQx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]) = (1/tau)*( (1-phi)*I + (phi)*blaze::submatrix(invQx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i])); 
          th += 2;
          }
        } else if(effi[i]=="iid"){
          if(xpts[0+i]!=-1) {blaze::band(blaze::submatrix(invQx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]),0) = exp(-theta[th])*blaze::band(blaze::submatrix(invQx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]),0);th++;}
        } else {
          if(xpts[0+i]!=-1) {blaze::submatrix(invQx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]) = exp(-theta[th])*blaze::submatrix(invQx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]);th++;}
        }
      } 
  }

  void update_Qx_by_theta(col_vector &theta)
  {  
    size_t th = 0;
    for (size_t i = 0; i < theta_size_Qx; i++)
      if (prior_blocks[i]==1)
      {
        if(effi[i]=="bym2"){
          if(xpts[0+i]!=-1) {
          blaze::IdentityMatrix<double, blaze::rowMajor> I(xpts[0 + i]);
          double tau = exp(theta[th]);
          double phi = exp(theta[th+1])/ (1.0 + exp(theta[th+1]));
          //update later
          //if(xpts[0+i]!=-1) blaze::submatrix(invQx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]) = (1/tau)*( (1-phi)*I + (phi)*blaze::submatrix(invQx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i])); 
          th += 2;
          }
        } else if(effi[i]=="iid"){
          if(xpts[0+i]!=-1) {blaze::band(blaze::submatrix(Qx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]),0) = exp(theta[th])*blaze::band(blaze::submatrix(Qx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]),0);th++;}
        } else {
          if(xpts[0+i]!=-1) {blaze::submatrix(Qx_theta,xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]) = exp(theta[th])*blaze::submatrix(Qx_fixed, xpts[2*theta_size_Qx+i],xpts[2*theta_size_Qx+i], xpts[i], xpts[i]);th++;}
        }
      } 

  }*/


/*
  // to exploit structure of AQA
  void get_AQ()
  {  
    size_t NB = num_blocks - x_mu - zcov;
    if(x_mu==1) blaze::submatrix(AQ,0,0, y_size, 1) = blaze::submatrix(A,0,0, y_size, 1)*blaze::submatrix(invQx_theta,0,0, 1, 1);
    if(zcov>0) {size_t ind_Q  = x_mu + zcov - 1; blaze::submatrix(AQ,0,ind_Q, y_size, zcov) = blaze::submatrix(A,0,ind_Q, y_size, zcov)*blaze::submatrix(invQx_theta,ind_Q,ind_Q, zcov, zcov);}
    for (size_t i = 0; i < NB; i++){
      size_t ind_Q = xpts[2*theta_size_Qx+i];
      size_t block_s = xpts[i];
      blaze::submatrix(AQ,0, ind_Q, y_size, block_s) = blaze::submatrix(A,0,ind_Q, y_size, block_s)*blaze::submatrix(invQx_theta,ind_Q,ind_Q, block_s, block_s);
    }
    //std::cout << A*invQx_theta - AQ << std::endl;

  }

 void get_QA()
  {  
    size_t NB = num_blocks - x_mu - zcov;
    for (size_t i = 0; i < NB; i++){
      size_t ind_Q = xpts[2*theta_size_Qx+i];
      size_t block_s = xpts[i];
      blaze::submatrix(QA,ind_Q,0, block_s,y_size) = blaze::submatrix(invQx_theta,ind_Q,ind_Q, block_s, block_s)*blaze::submatrix(trans(A),ind_Q,0, block_s, y_size);
    }

  }
*/

  void add_eigenvalues_for_bym2(size_t &num_sects) {evalues_effs.reset(new det_blocks[num_sects]);}
  //till here 
  void add_Qpts_to_Bowl(col_vector &Q_pts){ Qpts = Q_pts;}
  col_vector get_Qpts_from_Bowl(){ return Qpts;}

  void add_to_Bowl(string s, double v)
  {
    node *tmp = new node;
    tmp->type = s;
    tmp->value = v;
    tmp->next = NULL;

    if (head == NULL)
    {
      head = tmp;
      tail = tmp;
    }
    else
    {
      tail->next = tmp;
      tail = tail->next;
    }
  }

  void pour_the_Bowl()
  {
    node *tmp;
    tmp = head;
    while (tmp != NULL)
    {
      if(id_bowl==0) std::cout << " " << tmp->type << " is " << tmp->value << std::endl;
      tmp = tmp->next;
    }
  }

  double get_from_Bowl(string s)
  {
    node *tmp;
    tmp = head;
    while (tmp != NULL)
    {
      if (tmp->type == s)
        return tmp->value;
      tmp = tmp->next;
    }

    return std::numeric_limits<double>::quiet_NaN();
  }
};

using ptr_bowl = boost::scoped_ptr<Bowl>;
using ptr_optimBowl = boost::scoped_ptr<optim_Bowl>;

//optimization
#include "Eigen/Core"
#include "Eigen/Eigenvalues" // header file
#include "LBFGS.h"
#include "LBFGSB.h"

#include "Eigen/QR"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace LBFGSpp;

class GLP_libraries
{
};


#endif

