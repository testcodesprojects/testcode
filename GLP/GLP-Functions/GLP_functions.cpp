
#ifndef _cGLPfunctions_
#define _cGLPfunctions_

#include "GLP_functions.h"

void log_fac(col_vector &logfac)
{
   size_t n = logfac.size();
   if(n==1) logfac[0] = 0;
   if(n==2) {logfac[0] = 0; logfac[1] = 0;
   }else{
      logfac[0] = 0; logfac[1] = 0;
      for(size_t i = 2; i < n; i++)
         logfac[i] = std::log(i) + logfac[i-1];
   }
   
}

double log_det(sym_matrix &Q)
{
   low_matrix L;
   blaze::llh(Q,L);

   double value = 0.0;
   for(size_t i=0; i<L.rows();i++)
      value += log(L(i,i));

   value *= 2;
   return value;
}

void get_txt_size_t(string s,size_t &info)
{
   ifstream input(s);
   if (input.is_open())
      {input >> info;  input.close(); input.clear();}
   else cout << "Unable to open this file: " + s; 
}

void get_txt_bool(string s,bool &tinfo)
{
   size_t info;
   ifstream input(s);
   if (input.is_open())
      {input >> info;  input.close(); input.clear();}
   else cout << "Unable to open this file: " + s; 
   if(info==0)  tinfo =  false; else  tinfo =  true;
}

void get_txt_column(string s,col_vector &info)
{
   string line; size_t iter = 0;
   ifstream input(s, std::ifstream::in);
   if (input.is_open())
   {
      while ( getline (input,line) )
         {info[iter] = std::stod(line);iter++;}
      input.close();
   }
   else cout << "Unable to open this file: " + s; 
}

void get_txt_string_column(string s,blaze::DynamicVector<string> &info)
{
   string line; size_t iter = 0;
   ifstream input(s, std::ifstream::in);
   if (input.is_open())
   {
      while ( getline (input,line) )
         {
            //if(line=="NoPrior"){} else{info[iter] = line;}
            info[iter] = line;
            iter++;
         }
      input.close();
   }
   else cout << "Unable to open this file: " + s; 
}

void get_txt_string(string s,string &info)
{
   ifstream input(s);
   if (input.is_open())
      {input >> info;  input.close(); input.clear();}
   else cout << "Unable to open this file: " + s; 
}

void get_txt_string_p(string s,string &info)
{
   ifstream input(s);
   if (input.is_open())
      {input >> info;  input.close(); input.clear();}
   else info = "";
}

void get_txt_double(string s,double &info)
{
   ifstream input(s);
   if (input.is_open())
      {input >> info;  input.close(); input.clear();}
   else cout << "Unable to open this file: " + s; 
}

void get_txt_int(string s,int &info)
{
   double tinfo;
   ifstream input(s);
   if (input.is_open())
      {input >> tinfo;  input.close(); input.clear();}
   else cout << "Unable to open this file: " + s;
   info = (int)tinfo;
}

void ginv_sym(sym_matrix &Q, size_t &rankdef, sym_matrix &invQ)
{
   DynamicVector<double,columnVector> w(Q.rows());       // The vector for the real eigenvalues
   DynamicMatrix<double,rowMajor>     V(Q.rows(),Q.rows());  // The matrix for the left eigenvectors
   eigen( Q, w, V );

   //std::cout << trans(w) << std::endl;
   //std::cout << rankdef << std::endl;

   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D(Q.rows()-rankdef);
   blaze::diagonal(D) = 1.0/subvector(w,rankdef,Q.rows()-rankdef);
   invQ = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());
   //blaze::diagonal(D) = subvector(w,rankdef,Q.rows()-rankdef);
  // Q = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());

   //print(invQ);



}

void ginv_sym_with_Qx(sym_matrix &Q, size_t &rankdef, sym_matrix &invQ)
{
   DynamicVector<double,columnVector> w(Q.rows());       // The vector for the real eigenvalues
   DynamicMatrix<double,rowMajor>     V(Q.rows(),Q.rows());  // The matrix for the left eigenvectors
   eigen( Q, w, V );

   //std::cout << trans(w) << std::endl;
   //std::cout << rankdef << std::endl;

   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D(Q.rows()-rankdef);
   blaze::diagonal(D) = 1.0/subvector(w,rankdef,Q.rows()-rankdef);
   invQ = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());
   blaze::diagonal(D) = subvector(w,rankdef,Q.rows()-rankdef);
   Q = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());

   //print(invQ);
}

void ginv_sym_with_eigenvals_bym2(sym_matrix &Q, size_t &rankdef,ptr_bowl &param, size_t r, sym_matrix &invQ)
{
   size_t n = Q.rows();
   DynamicVector<double,columnVector> w(n);       // The vector for the real eigenvalues
   DynamicMatrix<double,rowMajor>     V(n,n);  // The matrix for the left eigenvectors
   eigen( Q, w, V );

   for(size_t i = 0; i < rankdef; i++) w[i] = 0.0;
   //blaze::submatrix(V, 0,0, 1, n) = 0.0;
   (*param).evalues_effs[r].evalues_eff.resize(n); 
   (*param).evalues_effs[r].evalues_eff = w;
   (*param).evalues_effs[r].V.resize(n,n); 
   (*param).evalues_effs[r].V = V;
   //std::cout << V << std::endl;
   //std::cout << w << std::endl;

   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D(Q.rows()-rankdef);
   blaze::diagonal(D) = 1.0/subvector(w,rankdef,Q.rows()-rankdef);
   invQ = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());
   //blaze::diagonal(D) = subvector(w,rankdef,Q.rows()-rankdef);
   //Q = submatrix(trans(V),0,rankdef,V.rows(),V.columns()-rankdef)*D*submatrix(V,rankdef,0,V.rows()-rankdef,V.columns());

}

void MGS_orthonormalization(col_matrix &G)
{   
   size_t i, j, k, n = G.columns();
	double r;

   col_vector q(n,0.0);

	for (i = 0; i < n; i++) {

		for (r = 0.0, j = 0; j < n; j++) r += (G(j,i)*G(j,i));
		r = sqrt(r);
		for (j = 0; j < n; j++) {q[j] = G(j,i)/r; G(j,i) = q[j];}

		for (j = i + 1; j < n; j++) {
			for (r = 0, k = 0; k < n; k++) r +=q[k] * G(k,j); 
			for (k = 0; k < n; k++) G(k,j) = G(k,j) - r*q[k]; 
				
		}
	}

}

void myeign(row_matrix &hessian_theta_star, col_vector &eigenvalues,row_matrix  &eigenvectors)
{
   if(hessian_theta_star.rows()==1)
   {
      DynamicMatrix<double,rowMajor> A(hessian_theta_star);  
      DynamicVector<complex<double>,columnVector> w( hessian_theta_star.rows() );  
      DynamicMatrix<complex<double>,rowMajor> V( hessian_theta_star.rows(), hessian_theta_star.rows() );

      eigen( A, w, V ); 
      eigenvalues = real(w);
      eigenvectors = real(eigenvectors);
      for(size_t i=0;i<hessian_theta_star.rows();i++)
         for(size_t j=0;j<hessian_theta_star.rows();j++)
            eigenvectors(i,j) = real(V(j,i));
   }
   else
   {
      //this is good!
      SymmetricMatrix< blaze::DynamicMatrix<double,blaze::rowMajor>> A(hessian_theta_star);  // The symmetric matrix A
      eigen(A, eigenvalues, eigenvectors);  
      eigenvectors = trans(eigenvectors);
   }
}

void z2theta(col_vector &z,col_vector &theta, col_vector &theta_mode, col_vector eigen_values, row_matrix eigen_vectors)
{
	// theta = theta_mode + eigen_vectors * diag(1/sqrt_eigen_values) * z 
   size_t theta_size = theta_mode.size();
   eigen_values = invsqrt(eigen_values);
   DiagonalMatrix< DynamicMatrix<double> > D(theta_size,theta_size);
   for(size_t i=0;i<theta_size;i++) D(i,i) = eigen_values[i];
   theta = theta_mode + eigen_vectors * D * z;
}

void theta2z(col_vector &theta,col_vector &z, col_vector &theta_mode, col_vector eigen_values, row_matrix eigen_vectors)
{
	// theta = theta_mode + eigen_vectors * diag(1/sqrt_eigen_values) * z 
   size_t theta_size = theta_mode.size();
   eigen_values = sqrt(eigen_values);
   DiagonalMatrix< DynamicMatrix<double> > D(theta_size,theta_size);
   for(size_t i=0;i<theta_size;i++) D(i,i) = eigen_values[i];
   z = D*trans(eigen_vectors)*(theta - theta_mode); 
} 

void export_vector(col_vector &x, string T)
{
   string sx = "Results/" + T;

    std::ofstream myfile;
   myfile.open (sx);
   for(size_t i=0; i< x.size(); i++)
      myfile << x[i] << "\n";
   myfile.close();

}

void export_density(col_vector &x, col_vector &y, size_t &index, string T)
{
   string sx = "Results/"+ T +"/x" + std::to_string(index+1) + ".txt",sy = "Results/"+ T +"/y" + std::to_string(index+1) + ".txt";

    std::ofstream myfile;
   myfile.open (sx);
   for(size_t i=0; i< x.size(); i++)
      myfile << x[i] << "\n";
   myfile.close();

   myfile.open (sy);
   for(size_t i=0; i< y.size(); i++)
      myfile << y[i] << "\n";
   myfile.close();

}

void export_margx_density(col_vector &x, col_vector &y, size_t &index, string T)
{
   string sx = "Results/"+ T +"/x" + std::to_string(index) + ".txt",sy = "Results/"+ T +"/y" + std::to_string(index) + ".txt";

    std::ofstream myfile;
   myfile.open (sx);
   for(size_t i=0; i< x.size(); i++)
      myfile << x[i] << "\n";
   myfile.close();

   myfile.open (sy);
   for(size_t i=0; i< y.size(); i++)
      myfile << y[i] << "\n";
   myfile.close();

}

void normalize_simpson_321_p(col_vector &y){
   y = y - blaze::max(y);
   col_vector s{1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,1};

   ptr norm_cnst{new double {0.02022132499*blaze::sum(blaze::exp(y)*s)}};
   y -= log((*norm_cnst));
   y = blaze::exp(y);
}

void normalize_simpson_321_p_dic(col_vector &y, col_vector &d){

   col_vector s{1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,
   4,2,4,2,4,2,4,1};

   ptr norm_cnst{new double {0.02022132499*blaze::sum(d*s)}};
   y.scale(1/(*norm_cnst));
}

double factorial(double n) {
  if(n > 1)
    return n * factorial(n - 1);
  else
    return 1;
}

double logfactorial(double n) {
  if(n > 1)
    return log(n) + logfactorial(n - 1);
  else
    return 0;
}

void seasoning(int num_tasks, int num_workers, int worker_index, bool include_boss, boost::ptr_vector<int> &tasks){

    int c_num_tasks = num_tasks - 1; 
    size_t start=0;
    if(!include_boss) start++;
    int work;

    if(num_tasks<=num_workers){
        if(worker_index<(num_tasks+start)){
            if(worker_index==0) work = (worker_index)%num_tasks;
            else work = (worker_index-start)%num_tasks;
            //std::cout << "Index: " << worker_index << " is working on " << work << std::endl;
            tasks.push_back(new int{work});
            
        }
    }else{
        
        int work = 0; int i =0;
        while(true){

            work = (worker_index+i*num_workers - start)%num_tasks;
            if(work<((worker_index+i*num_workers - start))) break;
            //std::cout << "Index: " << worker_index << " is working on " << work << std::endl;
            i++;
            tasks.push_back(new int{work});

        }
    }
}

void seasoning_margx(int num_tasks, int num_workers, int worker_index, bool include_boss, boost::ptr_vector<int> &tasks){

    //std::cout << "------------->>>>>>> " << num_workers << std::endl;

    int c_num_tasks = num_tasks - 1; 
    size_t start=0;
    if(!include_boss) start++;
    int work;

    if(num_tasks<=num_workers){

        if(worker_index<(num_tasks+start)){
            if(worker_index==0) work = (worker_index)%num_tasks;
            else work = (worker_index-start)%num_tasks;
            //std::cout << "Index: " << worker_index << " is working on " << work << std::endl;
            tasks.push_back(new int{work});
        }

    }else{
        
        int work = 0; int i =0;
        while(true){

            work = (worker_index+i*(num_workers-start) - start)%num_tasks;
            //std::cout << "work: " << worker_index << " is working on " << work <<  "----- " << num_workers << "------" << start  << "------" << num_tasks << std::endl;
            if(work<((worker_index+i*(num_workers-start) - start))) break;
            //std::cout << "Index: " << worker_index << " is working on " << work << std::endl;
            i++;
            tasks.push_back(new int{work});

        }
    }
}

int gradient_majic_spread(bool &central, int &pos, bool &left, bool &right, size_t &n){

    right = false; left = false; 
    if(pos==0){ return -999;}
    else{
        if(central){
            if ( pos % 2 == 0)
                {
                    right = true; size_t ind =0;
                    for(size_t k=2;k<=(n*2);k=k+2){
                        if(pos==k) return ind;
                        ind++;
                    }
                }
            else
                {
                    left = true; size_t ind =0;
                    for(size_t k=1;k<(n*2);k=k+2){
                        if(pos==k) return ind;
                        ind++;
                    }
                }
        }else{ //forward
            right = true;
            return pos - 1;
        }
    }
    return 999;
}

/*
void gradient(int &worker_index, int &num_workers,int &num_tasks, bool &central, size_t &n, int *ptr_tasks, int &size_tasks){

   boost::ptr_vector<int> tasks;
   majic_spread(num_tasks, num_workers, worker_index, true,tasks);
   size_tasks = tasks.size();
   ptr_tasks = new int[size_tasks];

   size_t i=0;
   for (auto task = tasks.begin(); task != tasks.end(); task++) {ptr_tasks[i] = *task; i++;}
    
}
*/

bool hessian_majic_spread(int &task_number, int &index_i, int &step_i, int &index_j, int &step_j, size_t &n, bool safe_mode){

   if(safe_mode){

      size_t tot1 = (n*n - n)/2, tot2 = 2*tot1, tot3 = 3*tot1;
      size_t k = 0;

      if(task_number<tot1){

         for(size_t ii=0; ii< n; ii++)
         {
            size_t jj;
            for(size_t jj = ii+1; jj < n; jj++)
            {
                  if(k==task_number){
                     index_i = ii;
                     index_j = jj;
                     step_i = 1;
                     step_j = -1;
                     return true;
                  }
                  k++;
            }
         }


      }else if(task_number>=tot1 && task_number<tot2){
         k = tot1;
         for(size_t ii=0; ii< n; ii++)
         {
            size_t jj;
            for(size_t jj = ii+1; jj < n; jj++)
            {
                  if(k==task_number){
                     index_i = ii;
                     index_j = jj;
                     step_i = -1;
                     step_j = 1;
                     return true;
                  }
                  k++;
            }
         }

      }else if(task_number>=tot2 && task_number<tot3){
         k = tot2;
         for(size_t ii=0; ii< n; ii++)
         {
            size_t jj;
            for(size_t jj = ii+1; jj < n; jj++)
            {
                  if(k==task_number){
                     index_i = ii;
                     index_j = jj;
                     step_i = -1;
                     step_j = -1;
                     return true;
                  }
                  k++;
            }
         }
      }

   }else{
      size_t k = 0;
      for(size_t i =0; i<3*n; i++){

         if(i<n){
               //std::cout << "Task #: " << k << " is working on " << i%n << " + h" << std::endl;

               if(k==task_number){
                  index_i = i%n;
                  step_i = 1;
                  return true;
               }

         } else if(i>=n && i<2*n){
               //std::cout << "Task #: " << k << " is working on " << i%n << " + 2h" << std::endl;

               if(k==task_number){
                  index_i = i%n;
                  step_i = 2;
                  return true;
               }

         } else if(i>=2*n && i <3*n){
               //std::cout << "Task #: " << k << " is working on " << i%n << " - 2h" << std::endl;

               if(k==task_number){
                  index_i = i%n;
                  step_i = -2;
                  return true;
               }

         }

         k++;
      }


      for(size_t ii=0; ii< n; ii++)
      {
         size_t jj;
         for(size_t jj = ii+1; jj < n; jj++)
         {
               //std::cout << "Task #: " << k << " is working on " << ii << " + h" <<  " | " << jj << " + h" << std::endl;

               if(k==task_number){
                  index_i = ii;
                  index_j = jj;
                  step_i = 1;
                  step_j = 1;
                  return true;
               }
               k++;
         }
      }

      }
      
      return false;
   }


   void task_position_connection(int task_number, row_matrix &hessian, double *value, size_t &n, bool safe_mode,row_matrix &savehihj){

      //std::cout <<  "task_number: " << task_number << std::endl;

      if(safe_mode){

         size_t tot1 = (n*n - n)/2, tot2 = 2*tot1, tot3 = 3*tot1;
         size_t k = 0;

         if(task_number<tot1){

            for(size_t ii=0; ii< n; ii++)
            {
               size_t jj;
               for(size_t jj = ii+1; jj < n; jj++)
               {
                     if(k==task_number){
                        hessian(ii,jj) -= *value; 
                     }
                     k++;
               }
            }


         }else if(task_number>=tot1 && task_number<tot2){
            k = tot1;
            for(size_t ii=0; ii< n; ii++)
            {
               size_t jj;
               for(size_t jj = ii+1; jj < n; jj++)
               {
                     if(k==task_number){
                        hessian(ii,jj) -= (*value);
                     }
                     k++;
               }
            }

         }else if(task_number>=tot2 && task_number<tot3){
            k = tot2;
            for(size_t ii=0; ii< n; ii++)
            {
               size_t jj;
               for(size_t jj = ii+1; jj < n; jj++)
               {
                     if(k==task_number){
                        hessian(ii,jj) += *value;
                     }
                     k++;
               }
            }
         }

      }else{
         if(task_number<n){

         for(size_t ii=0; ii< n; ii++)
         {
               size_t jj;
               for(size_t jj = ii+1; jj < n; jj++)
               {
                  
                  if(ii==task_number)  { hessian(ii,jj) -= *value;}
                  else if(jj==task_number) { hessian(ii,jj) -= *value;}
         
               }
         }

      }else if(task_number>=n && task_number<2*n){

         for(size_t ii=0; ii< n; ii++) if(ii==(task_number-n)) { hessian(ii,ii) += (*value);}

      }else if(task_number>=2*n && task_number<3*n){

         for(size_t ii=0; ii< n; ii++) if(ii==(task_number-2*n)) { hessian(ii,ii) += (*value);}

      }else if(task_number>=3*n){
         //size_t total = (n*n - n)/2;
         size_t task = 3*n;
         for(size_t ii=0; ii< n; ii++)
         {
               size_t jj;
               for(size_t jj = ii+1; jj < n; jj++)
               {
                  if(task==task_number) {  hessian(ii,jj) += *value; savehihj(ii,jj) = *value; }
                  task++;
               }
         }
      }
   }

}

int oven_majic_temperature(int &task_number, int &pos, size_t &n){

   int k = 0;
   for(size_t i =0; i<n; i++){

      if(task_number==k){
         pos = -1;
         return i;
      }
      k++;
      if(task_number==k){
         pos = 1;
         return i;
      }
      k++;
   }
   return 9999;
}

int check_oven(int &task_number, size_t &n){

   int k = 0;
   for(size_t i =0; i<n; i++){

      if(task_number==k){
         return i;
      }
      k++;
   }
   return 9999;
}

void oven_gloves(int &task_number, double &value, size_t &n, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg){

   int k = 0;
   for(size_t i =0; i<n; i++){

      if(task_number==k){
         stdev_corr_neg[i] = value;
      }
      k++;
      if(task_number==k){
         stdev_corr_pos[i] = value;
      }
      k++;
   }
}

void cupboard(ptr_bowl & param){

   (*param).grad_workers = (*param).size_bowl;

   if((*param).optim->central){
      (*param).num_tasks_grad = (*param).theta_size*2 + 1;
   }else{
      (*param).num_tasks_grad = (*param).theta_size + 1;
   }

   if((*param).grad_workers > (*param).num_tasks_grad)  (*param).grad_workers = (*param).num_tasks_grad;

   (*param).linesearch_workers = (*param).size_bowl - (*param).grad_workers;
   (*param).num_tasks_linesearch = (*param).size_bowl - (*param).grad_workers;

}

void drawer1(ptr_bowl & param){

   (*param).hess_workers = (*param).size_bowl;
   (*param).num_tasks_hess = ((*param).theta_size*(*param).theta_size - (*param).theta_size)/2 + 3*(*param).theta_size;
   if((*param).hess_workers > (*param).num_tasks_hess)  (*param).hess_workers = (*param).num_tasks_hess;
}

void drawer2(ptr_bowl & param){

   (*param).hess_workers = (*param).size_bowl;
   size_t n = (*param).theta_size;
   (*param).num_tasks_hess = 3*(n*n - n)/2;
   if((*param).hess_workers > (*param).num_tasks_hess)  (*param).hess_workers = (*param).num_tasks_hess;
}

void increase_temp(ptr_bowl & param){
   (*param).std_workers = (*param).size_bowl;
   (*param).num_tasks_std = 2*(*param).theta_size;
   if((*param).std_workers > (*param).num_tasks_std)  (*param).std_workers = (*param).num_tasks_std;
}

void decrease_temp(ptr_bowl & param){
   (*param).margtheta_workers = (*param).size_bowl;
   (*param).num_tasks_margtheta = (*param).theta_size;
   if((*param).margtheta_workers > (*param).num_tasks_margtheta)  (*param).margtheta_workers = (*param).num_tasks_margtheta;
}

void monitor_food_oven(ptr_bowl & param, size_t n){
   (*param).ccd_workers = (*param).size_bowl;
   (*param).num_tasks_ccd = n;
   if((*param).ccd_workers > (*param).num_tasks_ccd)  (*param).ccd_workers = (*param).num_tasks_ccd;

}

void turnoff_oven(ptr_bowl & param, size_t n){
   (*param).margx_workers = (*param).size_bowl;
   (*param).num_tasks_margx = n;
   if((*param).margx_workers > (*param).num_tasks_margx)  (*param).margx_workers = (*param).num_tasks_margx;

}

void polyfit(	const std::vector<double> &t,
		const std::vector<double> &v,
		std::vector<double> &coeff,
		int order

	     )
{
	// Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
	Eigen::MatrixXd T(t.size(), order + 1);
	Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
	Eigen::VectorXd result;

	// check to make sure inputs are correct
	assert(t.size() == v.size());
	assert(t.size() >= order + 1);
	// Populate the matrix
	for(size_t i = 0 ; i < t.size(); ++i)
	{
		for(size_t j = 0; j < order + 1; ++j)
		{
			T(i, j) = pow(t.at(i), j);
		}
	}
	std::cout<<T<<std::endl;
	
	// Solve for linear least square fit
	result  = T.householderQr().solve(V);
	coeff.resize(order+1);
	for (int k = 0; k < order+1; k++)
	{
		coeff[k] = result[k];
	}

}

#endif

/*
11. Hessian: 
(       1.6873   -0.0490519  -0.00016245 -0.000190869     0.017991 )
(   -0.0490519      2.27002    0.0066519   0.00766599     0.046311 )
(  -0.00016245    0.0066519      2.26422    -0.248736    0.0189386 )
( -0.000190869   0.00766599    -0.248736      2.16579    0.0213376 )
(     0.017991     0.046311    0.0189386    0.0213376      4.03821 )
*/