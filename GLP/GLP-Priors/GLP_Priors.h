
#ifndef _GLPPriors_
#define _GLPPriors_

#include "../GLP-Libraries/GLP_libraries.h"

double priors_precision_loggamma(double &theta, double p1, double p2)
{   
  double cnst = p1*std::log(p2)-std::log(tgamma(p1));
  return (theta*p1)-p2*std::exp(theta) + cnst;
}

double priors_phi_gaussian(double &theta, double p1, double p2)
{ 
  static const double log_norm_const_gaussian = -0.918938533204672741780329736407;	// log(1.0/sqrt(2.0*M_PI)) 

  return 0.5*log(p2) + log_norm_const_gaussian -0.5*p2*pow((theta - p1),2);
}

double priors_pc_precision(double &theta, double p1, double p2)
{   
  //a is p1 and b is p2 //no constant need to be added
  double lam = - std::log(p2)/p1; //p2 is between 0 and 1  
  return std::log(0.5*(lam)) - (lam)*exp(-0.5*theta) -0.5*theta; 
}

void set_priors(ptr_bowl &param)
{
  size_t theta_size = (*param).theta_size, size_priors_types;
  get_txt_size_t("././Data/size_priors_types.txt",size_priors_types);

  blaze::DynamicVector<string> priors_types(size_priors_types,"");
  col_vector priors_p1(theta_size,-1), priors_p2(theta_size,-1);

  get_txt_string_column("././Data/priors_types.txt",priors_types);
  (*param).priors_types = priors_types;

  string getprior_type; 
  get_txt_string("././Data/prior.txt",getprior_type);
  if(getprior_type=="pc.joint") (*param).pcjoint = true;

  get_txt_column("././Data/priors_p1.txt",priors_p1);
  get_txt_column("././Data/priors_p2.txt",priors_p2);
  size_t th = 0;
  for(size_t i = 0; i < priors_types.size(); i++)
  {
    if(priors_types[i]=="NoPrior" || priors_types[i]=="pc.joint"){ //do nothing!
    } else{
       if(priors_types[i]=="loggama.prec"){
        (*param).pr[th].p1 = priors_p1[th];
        (*param).pr[th].p2 = priors_p2[th];
        (*param).pr[th].type = priors_types[i];
        (*param).pr[th].f = &priors_precision_loggamma; th++;
       }else if(priors_types[i]=="gaussian"){
        (*param).pr[th].p1 = priors_p1[th];
        (*param).pr[th].p2 = priors_p2[th];
        (*param).pr[th].type = priors_types[i];
        (*param).pr[th].f = &priors_phi_gaussian; th++;
       } else if(priors_types[i]=="pc.prec"){ //i fixed this!
        (*param).pr[th].p2 = priors_p2[th];
        (*param).pr[th].p1 = priors_p1[th];
        (*param).pr[th].type = priors_types[i];
        (*param).pr[th].f = &priors_pc_precision; th++;

       } else if(priors_types[i]=="pc") {
          
          std::vector<double> sp_x, sp_y;
          std::string line; double val;
          string s = "././Data/pc-priors/prior" +std::to_string(th)+ "/spx.txt";
          //print(s);
          std::ifstream myFile1(s);
          while(getline(myFile1, line))
          {
            std::istringstream lineStream(line);
            
            lineStream >> val;
            sp_x.push_back(val);
          }
          s = "././Data/pc-priors/prior" +std::to_string(th)+ "/spy.txt";
          //print(s);

          std::ifstream myFile2(s);
          while(getline(myFile2, line))
          {
            std::istringstream lineStream(line);
            lineStream >> val;
            sp_y.push_back(val);
          }

          (*param).pr[th].Sp->set_points(sp_x,sp_y); 
          (*param).pr[th].type = priors_types[i];
          th++;
       } 
    }
  }
  
  






}

double log_Jacobian_pc_joint_types1234(col_vector &theta){

  /*
  ( a  -a   0   0   0 )
  ( 0   0   b  -b   0 )
  ( c   d   e   f   k )
  ( x   y   z   w   n )
  ( t   u   j   m   0 )
  */

  // det = ab*(k*part2 -n*part1); part1 = -mc - jc + ft + fu + et + eu -dm -dj; part2 = -mx - jx + zt + zu + wt + wu - jy - my;//
  
  double sumall = std::exp(-theta[0]) +  std::exp(- theta[1]) +  std::exp(- theta[2]) +  std::exp(- theta[3]) + std::exp(- theta[4]);
  double sumall2 = sumall*sumall;
  double logab = -theta[0] - theta[1] - theta[2] - theta[3] - 2*std::log((std::exp(-theta[0] - theta[2]) + std::exp(-theta[0] - theta[3]) + std::exp(-theta[1] - theta[2]) + std::exp(-theta[1] - theta[3]))); 
  double minus_n = (std::exp(- theta[4]-theta[0]) +  std::exp(- theta[4]- theta[1]) +  std::exp(- theta[4]- theta[2]) +  std::exp(- theta[4]- theta[3]))/sumall2;
  double plus_k = std::exp(- theta[4])/sumall2;

  double cc_dd = (std::exp(-theta[0]) + std::exp(-theta[1]))/sumall2,
  ff_ee = (std::exp(-theta[3]) + std::exp(-theta[2]))/sumall2,
  kk = std::exp(-theta[4])/sumall2,
  xx = std::exp(-theta[4]-theta[0])/sumall2,
  yy = std::exp(-theta[4]-theta[1])/sumall2,
  zz = std::exp(-theta[4]-theta[2])/sumall2,
  ww = std::exp(-theta[4]-theta[3])/sumall2,
  nn = -plus_k,
  sum1234_2 = (std::exp(-theta[0]) +  std::exp(- theta[1]) +  std::exp(- theta[2]) +  std::exp(- theta[3]))*(std::exp(-theta[0]) +  std::exp(- theta[1]) +  std::exp(- theta[2]) +  std::exp(- theta[3])),
  tt = (std::exp(-theta[0] - theta[2]) + std::exp(-theta[0] - theta[3]))/sum1234_2,
  uu = (std::exp(-theta[1] - theta[2]) + std::exp(-theta[1] - theta[3]))/sum1234_2,
  jj = (-std::exp(-theta[2] - theta[0]) - std::exp(-theta[2] - theta[1]))/sum1234_2,
  mm = (-std::exp(-theta[3] - theta[0]) - std::exp(-theta[3] - theta[1]))/sum1234_2;

  double part1 = - (cc_dd)*(mm + jj) + (ff_ee)*(tt + uu); 
  double part2 = - (jj+mm)*(xx + uu) + (zz+ww)*(tt + uu) ; 
  
  double logdet_1 = logab + std::log((plus_k*part2 + minus_n*part1));

/*
  blaze::DynamicMatrix<double,blaze::rowMajor> A(5,5);
  double s1 = std::exp(-theta[0]) +  std::exp(- theta[1]) +  std::exp(- theta[2]) +  std::exp(- theta[3]) + std::exp(- theta[4]);
  double s2 = std::exp(-theta[0]) +  std::exp(- theta[1]) +  std::exp(- theta[2]) +  std::exp(- theta[3]);
  
  A(0,0) = exp(-theta[0] - theta[1])/((exp(-theta[0])+exp(-theta[1]))*(exp(-theta[0])+exp(-theta[1])));
  A(0,1) = - exp(-theta[0] - theta[1])/((exp(-theta[0])+exp(-theta[1]))*(exp(-theta[0])+exp(-theta[1])));
  A(0,2) = 0;
  A(0,3) = 0;
  A(0,4) = 0;

  A(1,0) = 0;
  A(1,1) = 0;
  A(1,2) = exp(-theta[0] - theta[1])/((exp(-theta[0])+exp(-theta[1]))*(exp(-theta[0])+exp(-theta[1])));
  A(1,3) = exp(-theta[0] - theta[1])/((exp(-theta[0])+exp(-theta[1]))*(exp(-theta[0])+exp(-theta[1])));
  A(1,4) = 0;

  A(2,0) = exp(-theta[0])/(s1*s1);
  A(2,1) = exp(-theta[1])/(s1*s1);
  A(2,2) = exp(-theta[2])/(s1*s1);
  A(2,3) = exp(-theta[3])/(s1*s1);
  A(2,4) = exp(-theta[4])/(s1*s1);

  A(3,0) = exp(-theta[4]-theta[0])/(s1*s1);
  A(3,1) = exp(-theta[4] -theta[1])/(s1*s1);
  A(3,2) = exp(-theta[4] -theta[2])/(s1*s1);
  A(3,3) = exp(-theta[4] -theta[3])/(s1*s1);
  A(3,4) = - A(3,0) - A(3,1) - A(3,2) - A(3,3);                 

  A(4,0) = (exp(-theta[0] - theta[2]) + exp(-theta[0] - theta[3]))  /(s2*s2);
  A(4,1) = (exp(-theta[1] - theta[2]) + exp(-theta[1] - theta[3]))  /(s2*s2);
  A(4,2) = (- exp(-theta[2] - theta[0]) - exp(-theta[2] - theta[1]))  /(s2*s2);
  A(4,3) = (- exp(-theta[3] - theta[0]) - exp(-theta[3] - theta[1]))  /(s2*s2);
  A(4,4) = 0;


  double logdet_2 = std::log(det( A ));  
  
  std::cout << "logdet_1: " << logdet_1 << " logdet_2: " << logdet_2 << std::endl;

  */
  return logdet_1;

}

double pc_joint(col_vector &theta){

  //b = 1.67263 ---  U = 0.95, a = 0.99
  //b = 6.26982 ---  U = 0.5, a = 0.99
  //b = 20.59495 --- U = 0.05, a = 0.99

  double sum12345 = std::exp(-theta[0]) + std::exp(-theta[1]) + std::exp(-theta[2]) + std::exp(-theta[3]) + std::exp(-theta[4]);
  double gamma = std::exp(-theta[4])/sum12345, b = 6.26982; 
  double value_gamma = std::log(b) - b*std::sqrt(gamma) - std::log(2*std::sqrt(gamma)) - std::log(2) - std::log(1 - std::exp(-b));

  double invtau = sum12345, a = 0.01, U = 1.0/0.31, lambda = -std::log(a)/U;
  double value_tau = std::log(lambda/2.0)  + 1.5*std::log(invtau) - (lambda*std::sqrt(invtau));

  return value_gamma + value_tau + log_Jacobian_pc_joint_types1234(theta);
  
} 
#endif


