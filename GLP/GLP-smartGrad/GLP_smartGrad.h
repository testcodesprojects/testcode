
#ifndef smartGrad_H
#define smartGrad_H

//library
#include <iostream>
#include <chrono>
#include <random>

//FILES
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include "../GLP-Libraries/GLP_libraries.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace Eigen;
double mach_eps = std::numeric_limits<double>::epsilon();
typedef double ( fun1)(const VectorXd& x);

class smartGrad
{
    private:
        Eigen::MatrixXd G;
        Eigen::VectorXd prev_x, curr_x;
        int n, count;

    public:
        smartGrad(int x_size) 
        {
            n = x_size;  
            G.resize(n,n);
            for(size_t i=0;i<n;i++) G(i,i) = 1.0;//1.0;
            prev_x = VectorXd::Zero(n);
            curr_x = VectorXd::Zero(n);
            //G = 0.0;
            //band(G,0) =1.0;
            //prev_x.resize(n);
            //prev_x = 0.0; 
            count = 0;

        }

        void set_prev_x(VectorXd x)  {prev_x = x; count++;}
        void set_curr_x(VectorXd x)  {curr_x = x;}
        void get_curr_x(VectorXd x)  {x = curr_x;}


        void update_G(VectorXd &current_x)  
        {
            curr_x = current_x;
            for(size_t i=(n-1);i>=1;i--) for(size_t j=0;j<n;j++) G(j,i) = G(j,i-1);
            VectorXd xdiff = current_x - prev_x + distribution(generator);
            scale(xdiff);
            for(size_t i=0;i<n;i++) G(i,0) = xdiff[i] ;
            //std::cout<< current_x <<std::endl;
            //std::cout<< prev_x <<std::endl;

        }

        void MGS_orthogonalization()
        {   
            size_t i, j, k;
            double r;
            VectorXd q = VectorXd::Zero(n);

            for (i = 0; i < n; i++) 
            {

                for (r = 0.0, j = 0; j < n; j++) r += (G(j,i)*G(j,i));
                r = std::sqrt(r);
                for (j = 0; j < n; j++) {q[j] = G(j,i)/r; G(j,i) = q[j];}

                for (j = i + 1; j < n; j++) 
                {
                    for (r = 0, k = 0; k < n; k++) r += q[k] * G(k,j); 
                    for (k = 0; k < n; k++) G(k,j) = G(k,j) - r*q[k]; 
                }
            }

            for (j = 0; j < n; j++)  G(j,0) = G(j,0) + mach_eps;
        }

        void get_Gx(VectorXd &x) {x = G*x;}
        void get_G(Eigen::MatrixXd &GG) {GG = G;}

        void get_invGx(VectorXd &x) {x = G.colPivHouseholderQr().solve(x);}

        void transform_grad(VectorXd &x)
        {

           // std::cout<< x.transpose() <<std::endl;
           // std::cout<< prev_x.transpose() <<std::endl;
           // std::cout<< count <<std::endl;

           //             print_G();

            if(count>0) update_G(x);
            MGS_orthogonalization();
            //            print_G();
             set_prev_x(x);

            //                         std::cout<< "till here! " <<std::endl;

        }

        void get_grad(VectorXd &grad) {grad = (G.transpose()).colPivHouseholderQr().solve(grad);}

        void print_G() {std::cout << G << std::endl;   }
        void print_prev_x() { std::cout << prev_x << std::endl; }
        void scale(VectorXd &x)
        {  
            double m = x.mean() ;
            double sd = 0.0;
            for(size_t i=0;i<x.size();i++) sd += (x[i] - m)*(x[i] - m);
            sd = sqrt(sd/(x.size()-1.0));
            for(size_t i=0;i<x.size();i++)  x[i] = (x[i] - m)/sd;
        }

        double mytfun(fun1 &f,const VectorXd& phi)
        {
            VectorXd xx = curr_x + G*phi;
            double y = f(xx);
            return y;
        }

};

#endif // smartGrad_H






