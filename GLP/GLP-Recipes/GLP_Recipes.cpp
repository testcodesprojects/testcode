
#ifndef _cGLPRecipes_
#define _cGLPRecipes_

#include "GLP_Recipes.h"
#define xx_size 46//3D


void poisson_recipe(ptr_bowl &param, col_vector &pts, int comp_case){
   /*
      if(vb){
         (*param).update->exp_eta_star = blaze::exp(pts);
         (*param).correction->part_b_vec = (*param).y_response - (*param).update->exp_eta_star;
         diagonal((*param).correction->D_Qlike_x) = -(*param).update->exp_eta_star;
      }else{
         diagonal((*param).correction->D_Qlike_x) = (*param).update->exp_eta_star;
         (*param).correction->b_vec = trans((*param).A) * blaze::eval( - (*param).update->exp_eta_star + (*param).y_response);
      }
   */

   switch (comp_case) {
   case 1: //true
      (*param).update->exp_eta_star = (*param).Ntrials*blaze::exp(pts);
      (*param).correction->part_b_vec = (*param).y_response - (*param).update->exp_eta_star;
      diagonal((*param).correction->D_Qlike_x) = -(*param).update->exp_eta_star;
      break;
   case 2: //false
      diagonal((*param).correction->D_Qlike_x) = (*param).update->exp_eta_star;
      (*param).correction->b_vec = trans((*param).A) * blaze::eval( - (*param).update->exp_eta_star + (*param).y_response);
      break;
   case 3:
      (*param).update->exp_eta_star = (*param).Ntrials*blaze::exp(pts);
      diagonal((*param).correction->D_Qlike_x) = (*param).update->exp_eta_star;
      break;
   case 4:
      (*param).update->exp_eta_star = (*param).Ntrials*exp(pts);
      blaze::diagonal((*param).update->D_Qlike_eta) = (*param).update->exp_eta_star;
      (*param).update->b_eta = (*param).y_response - (*param).update->exp_eta_star;
      (*param).update->b_eta += (*param).update->D_Qlike_eta*pts;
      break;
   }

}

void binomail_recipe(ptr_bowl &param, col_vector &pts, int comp_case){
   switch (comp_case) {
   case 1: //true
      (*param).update->exp_eta_star = blaze::exp(pts);
      (*param).correction->part_b_vec = (*param).y_response - ((*param).Ntrials)*((*param).update->exp_eta_star/(1 + (*param).update->exp_eta_star));
      diagonal((*param).correction->D_Qlike_x) = -((*param).Ntrials)*((*param).update->exp_eta_star/pow((1+(*param).update->exp_eta_star),2));
      break;
   case 2: //false
      diagonal((*param).correction->D_Qlike_x) =  ((*param).Ntrials)*((*param).update->exp_eta_star/pow((1+(*param).update->exp_eta_star),2));
      (*param).correction->b_vec = trans((*param).A) *((*param).y_response - ((*param).Ntrials)*((*param).update->exp_eta_star/(1 + (*param).update->exp_eta_star)));
      break;
   case 3:
      (*param).update->exp_eta_star = blaze::exp(pts);
      diagonal((*param).correction->D_Qlike_x) = ((*param).Ntrials)*((*param).update->exp_eta_star/pow((1+(*param).update->exp_eta_star),2));
      break;
   }
}

void approx_NoNGaussian_RD_eta_Poisson(fun_type9 recipe,ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param)
{
   double value = 0.0;
   low_matrix L;

   eta_star = (*param).update->eta_moon;
   double f = 1.0; size_t count = 1; 
   update_invQx_by_theta(theta,param); 
  
   size_t eta_size = (*param).y_size;
   row_matrix AAA = (*param).invQx_theta*trans((*param).A);
   (*param).update->cov_eta = (*param).A*(*param).invQx_theta*trans((*param).A);
  
   band((*param).update->cov_eta,0) = band((*param).update->cov_eta,0)  + 1e-10;
   eigen((*param).update->cov_eta, (*param).update->w, (*param).update->V );
   size_t rankdef = 0; size_t i=0;
   while(true){ if((*param).update->w[i]>1e-8) break; else rankdef++; i++;}
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D((*param).y_size-rankdef);
   blaze::diagonal(D) = 1.0/subvector((*param).update->w,rankdef,(*param).y_size-rankdef);
   (*param).update->cov_eta = submatrix(trans((*param).update->V),0,rankdef,(*param).update->V.rows(),(*param).update->V.columns()-rankdef)*D*submatrix((*param).update->V,rankdef,0,(*param).update->V.rows()-rankdef,(*param).update->V.columns());
   
   double tol_x = 1e-5; int loops = 0;
   while(true){

      recipe(param, eta_star,4);

      blaze::diagonal((*param).update->D_Qlike_eta) = (*param).update->exp_eta_star;
      (*param).update->b_eta = (*param).y_response - (*param).update->exp_eta_star;
      (*param).update->b_eta += (*param).update->D_Qlike_eta*eta_star;

      blaze::llh((*param).update->D_Qlike_eta +  (*param).update->cov_eta,L);
      col_vector y;  // The solution vector
      solve( decllow( L ), y, (*param).update->b_eta);     // Solving the LSE with a lower system matrix
      solve( declupp( ctrans( L ) ), (*param).update->update_eta_star, y );     // Solving the LSE with an upper system matrix
     
      if(blaze::norm(eta_star - (*param).update->update_eta_star)<tol_x) {eta_star = (*param).update->update_eta_star; break;}
      if((*param).optim->optfunctioncall<1){
         f = min(1.0,(count+1.0)*0.4);
         eta_star = f*(*param).update->update_eta_star;
      } else eta_star = (*param).update->update_eta_star;

      count++;
   }

   eta_star = (*param).update->update_eta_star;
   *y = -0.5*trans(eta_star)*(*param).update->cov_eta*eta_star; //print("1"); print(-0.5*trans(eta_star)*(*param).update->cov_eta*eta_star);
   *y -= 0.5*sum(log(subvector((*param).update->w,rankdef,(*param).y_size-rankdef))); //print("2"); print(-0.5*sum(log(subvector((*param).update->w,rankdef,(*param).y_size-rankdef))));

   value = 0.0;
   for(size_t i=0; i<L.rows();i++) value += log(L(i,i));
   value *= 2;
   *y -= 0.5*value;

   if(!(*param).optim->hess_tick) (*param).update->eta_moon = eta_star;

}

void approx_NoNGaussian_RD_eta_Binomial(fun_type9 recipe,ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param){

   //print((*param).update->invQ_eta.rows());
   //print((*param).update->invQ_eta.columns());

   eta_star = (*param).update->eta_moon;
   double f = 1.0; size_t count = 1; 
   update_invQx_by_theta(theta,param); //update: invQx_theta

   size_t eta_size = (*param).y_size;
   band((*param).update->cov_eta,0) = band((*param).update->cov_eta,0)  + 1e-10;
   (*param).update->cov_eta = (*param).A*(*param).invQx_theta*trans((*param).A);
   eigen((*param).update->cov_eta, (*param).update->w, (*param).update->V );
   size_t rankdef = 0; size_t i=0;
   while(true){ if((*param).update->w[i]>1e-8) break; else rankdef++; i++;}
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D((*param).y_size-rankdef);
   blaze::diagonal(D) = 1.0/subvector((*param).update->w,rankdef,(*param).y_size-rankdef);
   (*param).update->cov_eta = submatrix(trans((*param).update->V),0,rankdef,(*param).update->V.rows(),(*param).update->V.columns()-rankdef)*D*submatrix((*param).update->V,rankdef,0,(*param).update->V.rows()-rankdef,(*param).update->V.columns());
   
   double tol_x = 1e-5; int loops = 0; col_vector p(eta_size);

   while(true){
      (*param).update->exp_eta_star = exp(eta_star);
      p = (*param).update->exp_eta_star/(1 + (*param).update->exp_eta_star);
      ////////----> The difference between models is here
      blaze::diagonal((*param).update->D_Qlike_eta) = ((*param).Ntrials)*(p/(1+(*param).update->exp_eta_star));
      (*param).update->b_eta = ((*param).y_response - ((*param).Ntrials)*p);

      (*param).update->b_eta += (*param).update->D_Qlike_eta*eta_star;
      (*param).update->invQ_eta = inv((*param).update->D_Qlike_eta +  (*param).update->cov_eta);
      (*param).update->update_eta_star = (*param).update->invQ_eta*(*param).update->b_eta;

     // print((*param).update->invQ_eta.rows());

      print(blaze::norm(eta_star - (*param).update->update_eta_star));
      if(blaze::norm(eta_star - (*param).update->update_eta_star)<tol_x) {eta_star = (*param).update->update_eta_star; break;}
      if(blaze::norm(eta_star - (*param).update->update_eta_star)<1e-3 && count >5 && (*param).optim->optfunctioncall>1) {eta_star = (*param).update->update_eta_star; break;}
      if((*param).optim->optfunctioncall<2){
         f = min(1.0,(count+1.0)*0.4);
         eta_star = f*(*param).update->update_eta_star;
      } else eta_star = (*param).update->update_eta_star;

      count++;
   }
   eta_star = (*param).update->update_eta_star;

   *y = -0.5*trans(eta_star)*(*param).update->cov_eta*eta_star; //print("1"); print(-0.5*trans(eta_star)*(*param).update->cov_eta*eta_star);
   *y -= 0.5*sum(log(subvector((*param).update->w,rankdef,(*param).y_size-rankdef))); //print("2"); print(-0.5*sum(log(subvector((*param).update->w,rankdef,(*param).y_size-rankdef))));

   band((*param).update->invQ_eta,0) = band((*param).update->invQ_eta,0) + 1e-10;
   *y += 0.5*log_det((*param).update->invQ_eta); //print("3"); print(0.5*log_det((*param).update->invQ_eta));
   if(!(*param).optim->hess_tick) (*param).update->eta_moon = eta_star;
}

void approx_Gaussian_RD_eta(fun_type9 recipe,ptr &y,col_vector &eta_star,col_vector &theta, ptr_bowl &param){

   //std::cout << std::setprecision(10);
   //print(trans(theta));
   //get "good" initial value and update covariance of eta 
   eta_star = (*param).update->eta_moon;
   double f = 1.0; size_t count = 1; 
   update_invQx_by_theta(theta,param); //update: invQx_theta

   size_t eta_size = (*param).y_size;
   (*param).update->cov_eta = (*param).A*(*param).invQx_theta*trans((*param).A);

   eigen((*param).update->cov_eta, (*param).update->w, (*param).update->V );

   size_t indexOftheta; double gauss_theta=0.0; auto get_it = (*param).get_from_Bowl("C_Gaussian_Noise");
   if(!isnan(get_it)) gauss_theta = get_it; 
   else 
   {
      indexOftheta = (size_t)((*param).get_from_Bowl("R_Gaussian_Noise"));
      gauss_theta = theta[indexOftheta]; 
   }
   
   col_vector invw = 1.0/(*param).update->w;
   blaze::diagonal((*param).update->D_Qlike_eta) = (*param).update->w/(std::exp(gauss_theta)*(*param).update->w+1);
   (*param).update->invQ_eta = trans((*param).update->V)*(*param).update->D_Qlike_eta*(*param).update->V;
   blaze::diagonal((*param).update->D_Qlike_eta) = std::exp(gauss_theta);
   (*param).update->update_eta_star = (*param).update->invQ_eta*(*param).update->D_Qlike_eta*(*param).y_response;
   eta_star = (*param).update->update_eta_star;

/*
   size_t rankdef = 0; size_t i=0;
   while(true){ if((*param).update->w[i]>1e-8) break; else rankdef++; i++;}
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D((*param).y_size-rankdef);
   blaze::diagonal(D) = 1.0/subvector((*param).update->w,rankdef,(*param).y_size-rankdef);
   (*param).update->cov_eta = submatrix(trans((*param).update->V),0,rankdef,(*param).update->V.rows(),(*param).update->V.columns()-rankdef)*D*submatrix((*param).update->V,rankdef,0,(*param).update->V.rows()-rankdef,(*param).update->V.columns());
   *y = -0.5*trans(eta_star)*(*param).update->cov_eta*eta_star;
*/


   size_t rankdef = 0; size_t i=0;
   while(true){ if((*param).update->w[i]>1e-8) break; else rankdef++; i++;}
   blaze::DiagonalMatrix< blaze::DynamicMatrix<double>> D((*param).y_size-rankdef);
   blaze::diagonal(D) = 1.0/subvector((*param).update->w,rankdef,(*param).y_size-rankdef);
   *y = -0.5*trans(eta_star)*submatrix(trans((*param).update->V),0,rankdef,(*param).update->V.rows(),(*param).update->V.columns()-rankdef)*D*submatrix((*param).update->V,rankdef,0,(*param).update->V.rows()-rankdef,(*param).update->V.columns())*eta_star;

   //blaze::diagonal((*param).update->D_Qlike_eta) = 1.0/(*param).update->w;
   //*y = -0.5*trans(eta_star)*trans((*param).update->V)*(*param).update->D_Qlike_eta*(*param).update->V*eta_star;

   //sym_matrix TTT(inv((*param).update->cov_eta));
   //print(-0.5*trans(eta_star)*TTT*eta_star-*y);
   //std::cout << "p(x): " << *y << std::endl;

   *y -= 0.5*sum(log(std::exp(gauss_theta)*(*param).update->w + 1));
   //std::cout << "logdet: " << -0.5*sum(log(std::exp(gauss_theta)*(*param).update->w + 1)) << std::endl;

   if(!(*param).optim->hess_tick) (*param).update->eta_moon = eta_star;

}

void correction_Gaussian(fun_type9 recipe, col_vector &theta, ptr_bowl &param){

   //print("-0--");
   size_t s = (*param).x_size;
   size_t indexOftheta; double gauss_theta=0.0; auto get_it = (*param).get_from_Bowl("C_Gaussian_Noise");
   if(!isnan(get_it)){
      gauss_theta = get_it; 
      (*param).correction->S2 = (*param).correction->S1;
   } else 
   {
      //print("-01--");
      indexOftheta = (size_t)((*param).get_from_Bowl("R_Gaussian_Noise"));
      //print(theta);

      gauss_theta = theta[indexOftheta]; 

      (*param).correction->S2 = exp(gauss_theta)*(*param).correction->S1;
   }
   //print("-1--");
   //print(trans(eta_star));

   (*param).correction->S2 = (*param).correction->S2*solve((*param).correction->I + (*param).correction->S2,(*param).invQx_theta);

   //print((*param).correction->S2);

   /*if((*param).x_mu>0)
   {
      blaze::submatrix((*param).correction->S2,0, 0,1, s) = 0.5*(trans(blaze::submatrix((*param).correction->S2,0, 0,s,1)) + blaze::submatrix((*param).correction->S2,0, 0,1, s));
      blaze::submatrix((*param).correction->S2,0, 0,s,1) = trans(blaze::submatrix((*param).correction->S2,0, 0,1, s)) ;
   }

   for(size_t b = 0; b < (*param).zcov; b++)
   {
      blaze::submatrix((*param).correction->S2,(b+1), 0,1, s) = 0.5*(trans(blaze::submatrix((*param).correction->S2,0, (b+1),s,1)) + blaze::submatrix((*param).correction->S2,(b+1), 0,1, s));
      blaze::submatrix((*param).correction->S2,0, (b+1),s,1) = trans(blaze::submatrix((*param).correction->S2,(b+1), 0,1, s)) ;
   }*/
   (*param).correction->S2 = 0.5*((*param).correction->S2 + trans((*param).correction->S2));
   (*param).correction->invQ = (*param).invQx_theta - (*param).correction->S2;
   (*param).correction->b_vec = exp(gauss_theta)*eval(trans((*param).A)*(*param).y_response);
   (*param).correction->x_star = (*param).correction->invQ*(*param).correction->b_vec;

   //print(trans((*param).correction->x_star));
   //print((*param).correction->b_vec);
}

void correction_Non_Gaussian(fun_type9 recipe, col_vector &theta, ptr_bowl &param){

   //print("-0--");
   size_t s = (*param).x_size;

   recipe(param,(*param).correction->part_b_vec, 2);
   //Poisson //inside recipe
   //diagonal((*param).correction->D_Qlike_x) = (*param).update->exp_eta_star;
   //Binomial
   //diagonal((*param).correction->D_Qlike_x) =  ((*param).Ntrials)*((*param).update->exp_eta_star/pow((1+(*param).update->exp_eta_star),2));

   (*param).correction->S1 = trans((*param).A)*(*param).correction->D_Qlike_x*(*param).A;
   (*param).correction->S2 = (*param).invQx_theta*(*param).correction->S1;

   (*param).correction->S2 = (*param).correction->S2*solve((*param).correction->I + (*param).correction->S2,(*param).invQx_theta);
   (*param).correction->S2 = 0.5*((*param).correction->S2 + trans((*param).correction->S2));

   (*param).correction->invQ = (*param).invQx_theta - (*param).correction->S2;

   //Poisson //inside recipe
   //(*param).correction->b_vec = (trans((*param).A) * blaze::eval( - (*param).update->exp_eta_star + (*param).y_response));
   //Binomial
   //(*param).correction->b_vec = trans((*param).A) *((*param).y_response - ((*param).Ntrials)*((*param).update->exp_eta_star/(1 + (*param).update->exp_eta_star)));

   (*param).correction->b_vec += trans((*param).A)*blaze::eval((*param).correction->D_Qlike_x * (*param).update->update_eta_star);
   (*param).correction->x_star = (*param).correction->invQ*(*param).correction->b_vec;

   //print(trans((*param).correction->x_star));
   //print((*param).correction->b_vec);

   //if(false){
   if((*param).correction->vb){

      update_Qx_by_theta(theta,param);
      (*param).update->update_eta_star = (*param).A * (*param).correction->x_star;
      recipe(param,(*param).update->update_eta_star, 3);

      (*param).update->invQ_eta = (*param).correction->D_Qlike_x + (*param).update->cov_eta;
      blaze::invert<blaze::byLLH>((*param).update->invQ_eta);

      //(*param).update->invQ_eta = blaze::inv((*param).update->invQ_eta);      //15 points from library(fastGHQuad) in R
      /*col_vector wp = {
      0.000000001522475804253521533004, 0.000001059115547711060988591452, 0.000100004441232499715242094951, 0.002778068842912759413288359411,
      0.030780033872546106593626191739, 0.158488915795935797481419626820, 0.412028687498898260610502575219, 0.564100308726418031568528022035,
      0.412028687498898260610502575219, 0.158488915795935492170087854902, 0.030780033872546040674134104620, 0.002778068842912776760523119179,
      0.000100004441232499891424947980, 0.000001059115547711068188371504, 0.000000001522475804253518224282};
      col_vector xp = {
      -4.4999907073093909914973664854187518358, -3.6699503734044540692593727726489305496, -2.9671669279056027690444352629128843546,
      -2.3257324861738579713232866197358816862, -1.7199925751864888479047976943547837436, -1.1361155852109203756583610811503604054,
      -0.5650695832555758801873935226467438042, -0.0000000000000003841438361818757626743,  0.5650695832555755471204861350997816771,
      1.1361155852109199315691512310877442360, 1.7199925751864892919940075444173999131,  2.3257324861738579713232866197358816862,
      2.9671669279056027690444352629128843546,  3.6699503734044527369917432224610820413,  4.4999907073093909914973664854187518358};*/

      static col_vector wp = { //(1/sqrt(M_PI))*wp[i]
      0.00000000085896498996332963347633, 0.00000059754195979205728122570342, 0.00005642146405189008196877017975, 0.00156735750354994687362497618466,
      0.01736577449213761933921595925767, 0.08941779539984442970457223509584, 0.23246229360973202915374713484198, 0.31825951825951853679796954565973,
      0.23246229360973202915374713484198, 0.08941779539984426317111854132236, 0.01736577449213758117529948776792, 0.00156735750354995663144452855420,
      0.00005642146405189018361272385027, 0.00000059754195979206130463220288, 0.00000000085896498996332777231995};

      static col_vector xp = { //sqrt(2)*xp[i]
       -6.36394788882983775124557723756880, -5.19009359130478387100993131753057, -4.19620771126901548342402747948654,
      -3.28908242439876685736521721992176, -2.43243682700975805133225549070630, -1.60671006902872948174376688257325,
      -0.79912906832454821959288437938085, -0.00000000000000054326142303043719, 0.79912906832454777550367452931823,
      1.60671006902872881560995210747933, 2.43243682700975849542146534076892, 3.28908242439876685736521721992176, 
      4.19620771126901548342402747948654, 5.19009359130478209465309191728011, 6.36394788882983775124557723756880}; 
      
      col_vector S = blaze::sqrt(blaze::diagonal((*param).update->invQ_eta));
      col_vector b((*param).y_size), C((*param).y_size), eval_pts((*param).y_size); double tmp = 0.0;
      col_vector x_star_vb((*param).x_size), del((*param).x_size);
      sym_matrix Qvb((*param).x_size);

      int count = 0; 
      while(true){
         b = C = 0.0;
         for(size_t i =0; i <15; i++){
            eval_pts = xp[i]*S + (*param).update->update_eta_star;
            tmp = wp[i];

            recipe(param,eval_pts, 1);
            //b -= tmp*( (*param).y_response - exp(eval_pts));// (*param).correction->part_b_vec; // ---> first derivative of the likelihood
            //C -= tmp*(- exp(eval_pts));    // diagonal((*param).correction->D_Qlike_x);  //  ---> second derivative of the likelihood
            b -= tmp*(*param).correction->part_b_vec; // ---> first derivative of the likelihood
            C -= tmp*diagonal((*param).correction->D_Qlike_x);  //  ---> second derivative of the likelihood
         }

         diagonal((*param).correction->D_Qlike_x) = C; 
         //sym_matrix Qvb =  trans((*param).A)*(*param).correction->D_Qlike_x*(*param).A + (*param).Qx_theta;
         col_vector bvb = - trans((*param).A)*b - (*param).Qx_theta*(*param).correction->x_star;

         (*param).correction->S1 = trans((*param).A)*(*param).correction->D_Qlike_x*(*param).A;
         (*param).correction->S2 = (*param).invQx_theta*(*param).correction->S1;

         (*param).correction->S2 = (*param).correction->S2*solve((*param).correction->I + (*param).correction->S2,(*param).invQx_theta);
         (*param).correction->S2 = 0.5*((*param).correction->S2 + trans((*param).correction->S2));

         Qvb = (*param).invQx_theta - (*param).correction->S2;
         del = Qvb*bvb;
         
         x_star_vb = (*param).correction->x_star + del;

         if(blaze::norm(x_star_vb - (*param).correction->x_star)<1e-3){
            (*param).correction->x_star = x_star_vb;
            (*param).update->update_eta_star = (*param).A * (*param).correction->x_star;
            break;
         }
         count++;

         (*param).correction->x_star = x_star_vb;
         (*param).update->update_eta_star = (*param).A * (*param).correction->x_star;
         if(count==6) break; 
      }
   }
}




void stdev_corr(fun_type7 fun, row_matrix &COV, ptr_bowl &param, col_vector &stdev_corr_pos,col_vector &stdev_corr_neg, col_vector &theta_star, ptr &ldense_theta_star,col_vector &eigenvalues, row_matrix  &eigenvectors, col_vector &eta_star_k)
{
   size_t theta_size = theta_star.size();
   col_vector give_z(theta_size,0.0);
   col_vector get_theta(theta_size,0.0);
   ptr ftheta{new double{0.0}};
   ptr diff{new double{0.0}};
   give_z =0.0;
   for(size_t k=0;k<theta_size;k++)
   {
      *ftheta=0;
      give_z[k] = 2.0;
      //std::cout << "testtt" << std::endl;
      get_theta = 0.0;
      z2theta(give_z,get_theta,theta_star,eigenvalues,eigenvectors);
      *ftheta = 0.0; 
      /*{  
         for(size_t j=0;j<theta_size;j++)
         if (j != k) get_theta[j] = theta_star[j] + (COV(k, j) / COV(k, k)) * (get_theta[k] - theta_star[k]);
      }*/
      fun(ftheta,get_theta,eta_star_k,param);
      //std::cout << *ftheta << std::endl;
      //std::cout << *ldense_theta_star << std::endl;

      *diff = *ftheta + *ldense_theta_star;
      stdev_corr_pos[k] = ((*diff) > 0.0) ? sqrt(2.0 / (*diff)) : 1.0;

      give_z[k] = -2.0;
      get_theta = 0.0;
      z2theta(give_z,get_theta,theta_star,eigenvalues,eigenvectors);
      
      *ftheta = 0.0; 
      /*{  
         for(size_t j=0;j<theta_size;j++)
         if (j != k) get_theta[j] = theta_star[j] + (COV(k, j) / COV(k, k)) * (get_theta[k] - theta_star[k]);
      }*/
      fun(ftheta,get_theta,eta_star_k,param); 
      //std::cout << get_theta - theta_star << std::endl;

      *diff = *ftheta + *ldense_theta_star;
      stdev_corr_neg[k] = ((*diff) > 0.0) ? sqrt(2.0 / (*diff)) : 1.0;

      give_z.reset();
   }
}

void stdev_corr_position(fun_type7 fun, ptr_bowl &param, col_vector &theta_star, ptr &ldense_theta_star,col_vector &eigenvalues, row_matrix  &eigenvectors, col_vector &eta_star_k, int &index, int &position, double &value)
{
   size_t theta_size = theta_star.size();
   col_vector give_z(theta_size,0.0), get_theta(theta_size,0.0);
   ptr ftheta{new double{0.0}}, diff{new double{0.0}};
   give_z =0.0;

   *ftheta=0;
   give_z[index] = 2.0*position;
   get_theta = 0.0;
   z2theta(give_z,get_theta,theta_star,eigenvalues,eigenvectors);
   *ftheta = 0.0; 
   fun(ftheta,get_theta,eta_star_k,param);
   *diff = *ftheta + *ldense_theta_star;
   value = ((*diff) > 0.0) ? sqrt(2.0 / (*diff)) : 1.0;

}

#endif



