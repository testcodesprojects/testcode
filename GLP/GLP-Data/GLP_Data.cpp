
#ifndef _cGLPData_
#define _cGLPData_

#include "GLP_Data.h"


void set_generic_Qx(ptr_bowl &param,size_t &num_sects, size_t &iter)
{
    string envi = "";
    get_txt_string_p("././Data/envi.txt",envi);
    (*param).add_eigenvalues_for_bym2(num_sects);
    blaze::DynamicVector<string> effi(num_sects),RD_effi(num_sects);
    col_vector RD(num_sects), sizes(num_sects),prior_blocks(num_sects);blaze::DynamicVector<int> rankdef(num_sects);
    double prec_mu = 0.0;
    get_txt_string_column("././Data/effi.txt",effi);
    get_txt_string_column("././Data/RD_effi.txt",RD_effi);
    get_txt_column("././Data/RD.txt",RD);
    get_txt_column("././Data/sizes.txt",sizes);
    get_txt_column("././Data/prior_blocks.txt",prior_blocks);
    get_txt_double("././Data/prec_mu.txt",prec_mu);

    //string type="";
    //get_txt_string("././Data/type.txt",type);

    //std::cout << "Type is " << type << std::endl;


    (*param).prior_blocks = prior_blocks;
    
    //std::cout << "prior_blocks " << prior_blocks << std::endl;
    //std::cout << "prior_blocks " << (*param).prior_blocks << std::endl;

    blaze::DynamicVector<double> xpts(num_sects*3,-1); xpts[num_sects*3] = 0;
    size_t theta_size_Qx = 0;
    sym_matrix invQxfixed(param->x_size);
    sym_matrix Qxfixed;
    if((*param).correction->vb) Qxfixed.resize(param->x_size);

    if((*param).x_mu==1) {
        invQxfixed(0,0) = 1/prec_mu; 
        if((*param).correction->vb) Qxfixed(0,0) = prec_mu;}  
    if((*param).zcov>0)
    {
        sym_matrix prec_beta((*param).zcov), invprec_beta((*param).zcov);

        size_t index_start = (*param).x_mu;
        size_t index_end = (*param).x_mu + (*param).zcov - 1;

        ifstream f("././Data/prec_beta.txt");
        double info = 0;
        for (size_t i = 0; i < (*param).zcov; i++) for (size_t j = 0; j <= i; j++) {f >> info; prec_beta(i,j) = info;}

        //std::cout << prec_beta << std::endl;
        size_t cov_RD = 0;
        //
        if((*param).correction->vb){
            ginv_sym_with_Qx(prec_beta,cov_RD,invprec_beta);
            submatrix(Qxfixed,index_start,index_start,index_end,index_end) = prec_beta;
        }else{ginv_sym(prec_beta,cov_RD,invprec_beta);}

        submatrix(invQxfixed,index_start,index_start,index_end,index_end) = invprec_beta;         
    }  //print(num_sects);//print("position1"); 
    size_t fix_iter = 0;
    for(size_t r=0; r<num_sects; r++)
    {
            size_t n = sizes[r],Rdef = RD[r];
            sym_matrix subinvQx,subQx(n);
            xpts[0+theta_size_Qx] = n;
            xpts[num_sects+theta_size_Qx] = Rdef;
            xpts[2*num_sects+theta_size_Qx] = iter;
            (*param).add_to_Bowl(effi[r],(double)theta_size_Qx); 
            theta_size_Qx++;
            rankdef[theta_size_Qx-1] = n - Rdef; 

            if(RD_effi[r]!= "type1" || RD_effi[r]!= "type2" || RD_effi[r]!= "type3" || RD_effi[r]!= "type4" || RD_effi[r]!= "interaction1" || RD_effi[r]!= "interaction2" || RD_effi[r]!= "interaction3" || RD_effi[r]!= "interaction4")
            {   
                if(envi == "usingPython"){
                    double info = 0;
                    if(RD_effi[r]== "iid_time" || RD_effi[r]== "iid_space") {
                        for (size_t i = 0; i < n; i++) subQx(i,i) = 1.0;
                    }else{
                        string s = "././Data/Qx" +effi[r]+ ".txt";
                        ifstream f(s);
                        for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) {f >> info; subQx(i,j) = info;}
                    }

                    //std::cout << subQx << std::endl;
                }else{
                    string s = "././Data/Qx" +effi[r]+ ".txt";
                    ifstream f(s);
                    double info = 0;
                    for (size_t i = 0; i < n; i++) for (size_t j = 0; j <= i; j++) {f >> info; subQx(i,j) = info;}
                }
                
                //--------------> recently added
                band(subQx,0) = band(subQx,0) + 1e-6;
                //print(subQx); print("position2");
                fix_iter = iter;
                //std::cout << r << std::endl;
                if(RD_effi[r]=="bym2"){
                
                //submatrix(Qxfixed,iter,iter,n,n) = subQx; //fixthis         
                ginv_sym_with_eigenvals_bym2(subQx,Rdef,param,r,subinvQx);
                submatrix(invQxfixed,iter,iter,n,n) = subinvQx;
                iter = iter + n;
                } else{
                    if((*param).correction->vb){
                        ginv_sym_with_Qx(subQx,Rdef,subinvQx);
                        submatrix(Qxfixed,iter,iter,n,n) = subQx;
                    }else{
                        ginv_sym(subQx,Rdef,subinvQx);
                    }
                    submatrix(invQxfixed,iter,iter,n,n) = subinvQx;
                    iter = iter + n;
                }        
            }
            
        
 
    }


    size_t index_TSE, size_TSE, index_SSE, size_SSE;

    for(size_t r=0; r<num_sects; r++)
    {
        if(RD_effi[r]== "RW1" || RD_effi[r]== "RW2" || RD_effi[r]== "generic_time"){
            index_TSE = xpts[2*num_sects+r];
            size_TSE = xpts[r]; 
        }

        if(RD_effi[r]== "besag" || RD_effi[r]== "ICAR" || RD_effi[r]== "generic_space"){
            index_SSE = xpts[2*num_sects+r];  
            size_SSE = xpts[r];
        }

    }

    /*
    std::cout << index_TSE << std::endl;
    std::cout << size_TSE << std::endl;
    std::cout << index_SSE << std::endl;
    std::cout << size_SSE << std::endl;

    std::cout << submatrix(invQxfixed,index_TSE,index_TSE,size_TSE,size_TSE) << std::endl;
    std::cout << submatrix(invQxfixed,index_SSE,index_SSE,size_SSE,size_SSE) << std::endl;
    */

    for(size_t r=0; r<num_sects; r++)
    {

        if(RD_effi[r]== "type1" || RD_effi[r]== "type2" || RD_effi[r]== "type3" || RD_effi[r]== "type4" || RD_effi[r]== "interaction1" || RD_effi[r]== "interaction2" || RD_effi[r]== "interaction3" || RD_effi[r]== "interaction4")
            {

               

                size_t n = sizes[r];
                //std::cout << xpts << std::endl;
                iter = fix_iter;

                if(RD_effi[r]== "type1"){

                    blaze::IdentityMatrix<double> I_time(size_TSE);
                    blaze::IdentityMatrix<double> I_space(size_SSE);
                    if((*param).correction->vb){
                        submatrix(Qxfixed,iter,iter,n,n) = blaze::kron(I_time,I_space);
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(I_time,I_space);
                    }else{
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(I_time,I_space);
                    }

                }else if(RD_effi[r]== "type2"){

                    blaze::IdentityMatrix<double> I_time(size_TSE);
                    if((*param).correction->vb){
                        submatrix(Qxfixed,iter,iter,n,n) = blaze::kron(I_time,submatrix(Qxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(I_time,submatrix(invQxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                    }else{
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(I_time,submatrix(invQxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                    }

                }else if(RD_effi[r]== "type3"){

                    blaze::IdentityMatrix<double> I_space(size_SSE);
                    if((*param).correction->vb){
                        submatrix(Qxfixed,iter,iter,n,n) = blaze::kron(submatrix(Qxfixed,index_TSE,index_TSE,size_TSE,size_TSE),I_space);
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(submatrix(invQxfixed,index_TSE,index_TSE,size_TSE,size_TSE),I_space);
                    }else{
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(submatrix(invQxfixed,index_TSE,index_TSE,size_TSE,size_TSE),I_space);
                    }

                }else if(RD_effi[r]== "type4"){

                    if((*param).correction->vb){
                        submatrix(Qxfixed,iter,iter,n,n) = blaze::kron(submatrix(Qxfixed,index_TSE,index_TSE,size_TSE,size_TSE),submatrix(Qxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(submatrix(invQxfixed,index_TSE,index_TSE,size_TSE,size_TSE),submatrix(invQxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                    }else{
                        submatrix(invQxfixed,iter,iter,n,n) = blaze::kron(submatrix(invQxfixed,index_TSE,index_TSE,size_TSE,size_TSE),submatrix(invQxfixed,index_SSE,index_SSE,size_SSE,size_SSE));
                    }
                }

                if((*param).correction->vb) band(submatrix(Qxfixed,iter,iter,n,n),0) = band(submatrix(invQxfixed,iter,iter,n,n),0) + 1e-6;
                band(submatrix(invQxfixed,iter,iter,n,n),0) = band(submatrix(invQxfixed,iter,iter,n,n),0) + 1e-6;
            }

    }
    (*param).set_xpts(xpts);
    (*param).set_rankdef(rankdef);
    (*param).resize_Qx_and_invQx();
    (*param).add_invQx_fixed_to_Bowl(invQxfixed);
    if((*param).correction->vb) (*param).add_Qx_fixed_to_Bowl(Qxfixed);
    (*param).add_eff_to_Bowl(RD_effi); 
    if(sum(RD)==0) (*param).RD_system = false; else (*param).RD_system = true;

    //(*param).effi = effi;
    //print("inside data");
    //print(trans(xpts));
    //print(trans(rankdef));
    //if((*param).id_bowl==0) std::cout << "0. Number of Constraints: " << sum(RD) << std::endl;
    (*param).num_Con = sum(RD);
    //print(Qxfixed);
    //print(invQxfixed);
    

    //print(trans(RD_effi));
    //print(trans(effi));
    //print(trans(xpts));

    /*

    (*param).add_eigenvalues_for_bym2(num_sects);
    blaze::DynamicVector<string> effi(num_sects),RD_effi(num_sects);
    col_vector RD(num_sects), sizes(num_sects),prior_blocks(num_sects);blaze::DynamicVector<int> rankdef(num_sects);
    double prec_mu = 0.0;
    get_txt_string_column("././Data/effi.txt",effi);
    get_txt_string_column("././Data/RD_effi.txt",RD_effi);
    get_txt_column("././Data/RD.txt",RD);
    get_txt_column("././Data/sizes.txt",sizes);
    get_txt_column("././Data/prior_blocks.txt",prior_blocks);
    get_txt_double("././Data/prec_mu.txt",prec_mu);
    
    //prec_mu = 1;
    (*param).prior_blocks = prior_blocks;
    
    //std::cout << "prior_blocks " << prior_blocks << std::endl;
    //std::cout << "prior_blocks " << (*param).prior_blocks << std::endl;

    blaze::DynamicVector<double> xpts(num_sects*3,-1); xpts[num_sects*3] = 0;
    size_t theta_size_Qx = 0;
    sym_matrix invQxfixed(param->x_size);
    sym_matrix Qxfixed;
    if((*param).correction->vb) Qxfixed.resize(param->x_size);

    if((*param).x_mu==1) {
        invQxfixed(0,0) = 1/prec_mu; 
        if((*param).correction->vb) Qxfixed(0,0) = prec_mu;}  
    if((*param).zcov>0)
    {
        sym_matrix prec_beta((*param).zcov), invprec_beta((*param).zcov);

        size_t index_start = (*param).x_mu;
        size_t index_end = (*param).x_mu + (*param).zcov - 1;

        ifstream f("././Data/prec_beta.txt");
        double info = 0;
        for (size_t i = 0; i < (*param).zcov; i++) for (size_t j = 0; j <= i; j++) {f >> info; prec_beta(i,j) = info;}

        //std::cout << prec_beta << std::endl;
        size_t cov_RD = 0;
        //
        if((*param).correction->vb){
            ginv_sym_with_Qx(prec_beta,cov_RD,invprec_beta);
            submatrix(Qxfixed,index_start,index_start,index_end,index_end) = prec_beta;
        }else{ginv_sym(prec_beta,cov_RD,invprec_beta);}

        submatrix(invQxfixed,index_start,index_start,index_end,index_end) = invprec_beta;         
    }  //print(num_sects);//print("position1"); 
    for(size_t r=0; r<num_sects; r++)
    {
        size_t n = sizes[r],Rdef = RD[r];
        sym_matrix subinvQx,subQx(n);
        xpts[0+theta_size_Qx] = n;
        xpts[num_sects+theta_size_Qx] = Rdef;
        xpts[2*num_sects+theta_size_Qx] = iter;
        (*param).add_to_Bowl(effi[r],(double)theta_size_Qx); 
        theta_size_Qx++;
        rankdef[theta_size_Qx-1] = n - Rdef; 

        string s = "././Data/Qx" +effi[r]+ ".txt";
        ifstream f(s);
        double info = 0;
        for (size_t i = 0; i < n; i++) for (size_t j = 0; j <= i; j++) {f >> info; subQx(i,j) = info;}
        //--------------> recently added
        band(subQx,0) = band(subQx,0) + 1e-6;
        //print(subQx); print("position2");
        size_t fix_iter = iter;
        //std::cout << r << std::endl;
        if(RD_effi[r]=="bym2"){
            
            //submatrix(Qxfixed,iter,iter,n,n) = subQx; //fixthis         
            ginv_sym_with_eigenvals_bym2(subQx,Rdef,param,r,subinvQx);
            submatrix(invQxfixed,iter,iter,n,n) = subinvQx;
            iter = iter + n;
        } else{
            if((*param).correction->vb){
                ginv_sym_with_Qx(subQx,Rdef,subinvQx);
                submatrix(Qxfixed,iter,iter,n,n) = subQx;
            }else{ginv_sym(subQx,Rdef,subinvQx);}
            submatrix(invQxfixed,iter,iter,n,n) = subinvQx;
            iter = iter + n;
        }        
    }

    (*param).set_xpts(xpts);
    (*param).set_rankdef(rankdef);
    (*param).resize_Qx_and_invQx();
    (*param).add_invQx_fixed_to_Bowl(invQxfixed);
    if((*param).correction->vb) (*param).add_Qx_fixed_to_Bowl(Qxfixed);
    (*param).add_eff_to_Bowl(RD_effi); 
    if(sum(RD)==0) (*param).RD_system = false; else (*param).RD_system = true;


    */

    //(*param).effi = effi;
    //print("inside data");
    //print(trans(xpts));
    //print(trans(rankdef));
    //if((*param).id_bowl==0) std::cout << "0. Number of Constraints: " << sum(RD) << std::endl;
    //print(Qxfixed);
    //print(invQxfixed);
    

    //print(trans(RD_effi));
    //print(trans(effi));
    //print(trans(xpts));

}

void add_model_specifications(ptr_bowl &param)
{
    size_t theta_size = (*param).theta_size;
    if((*param).Model=="Gaussian")
    {   
        string Gaussian_Noise;
        get_txt_string("././Data/Gaussian_Noise.txt",Gaussian_Noise);
        
        if(Gaussian_Noise=="R_Gaussian_Noise") {
            (*param).add_to_Bowl("R_Gaussian_Noise",(double)(theta_size-1));
            (*param).update->set_eta_Qlike(0.0);
        } else if(!Gaussian_Noise.empty()) {
            (*param).add_to_Bowl("C_Gaussian_Noise",std::log(std::stod(Gaussian_Noise))); //sd = 0.1 
            (*param).update->set_eta_Qlike(std::log(std::stod(Gaussian_Noise)));
        }         
    } else if((*param).Model=="Binomial")
    {
        //double Binomial_Size;
        //get_txt_double("././Data/Binomial_Size.txt",Binomial_Size); 
        //(*param).add_to_Bowl("C_Binomial_Size", Binomial_Size);
        
        col_vector Ntrials; //check
        Ntrials.resize((*param).y_size);  get_txt_column("././Data/Ntrials.txt",Ntrials);
        (*param).set_Ntrials(Ntrials);
        //std::cout << "" << std::endl;
        //std::cout << Ntrials << std::endl;

    }else if((*param).Model=="Poisson"){
        col_vector Ntrials; //check
        Ntrials.resize((*param).y_size);  get_txt_column("././Data/Ntrials.txt",Ntrials);
        (*param).set_Ntrials(Ntrials);
    }

    (*param).theta_size = theta_size;

}

void getdata(ptr_bowl &param)
{

    //First we get the following: 
    /*
        1. size of x
        2. size of y
        3. intercept as boolean
        4. z covariates as boolean
        5. theta size
        6. theta size in Qx matrix
        7. number of random effects blocks in Qx
        8. Model or the family: "Gaussian for example"
        9. Qx type: Generic Mixed
        10. strategy 
        11. vb
    */

    size_t x_size,num_effs, y_size,iter=1,theta_size_Qx,theta_size = 0, zcov =0, x_mu = 0, num_blocks; //1 for mu
    string Model, Qx_type, strategy;
    blaze::CompressedMatrix<double> A; 
    col_vector id_effi,y_response;

    get_txt_size_t("././Data/x_size.txt",x_size);
    get_txt_size_t("././Data/y_size.txt",y_size);
    get_txt_string("././Data/strategy.txt",strategy);
    get_txt_string("././Data/Model.txt",Model);
    get_txt_string("././Data/Qx_type.txt",Qx_type);
    get_txt_size_t("././Data/theta_size_Qx.txt",theta_size_Qx); //id_effi.size()/y_size; 
    get_txt_size_t("././Data/theta_size.txt",theta_size);
    get_txt_size_t("././Data/num_effs.txt",num_effs);
    get_txt_size_t("././Data/Cov_Z.txt",zcov);
    get_txt_size_t("././Data/x_mu.txt",x_mu);
    get_txt_size_t("././Data/num_blocks.txt",num_blocks);

    (*param).num_blocks = num_blocks;
    (*param).zcov = zcov;
    (*param).x_mu = x_mu;
    A.resize(y_size,x_size);
    
    //std::cout << "inside getdata----------------------------" << std::endl;
    //std::cout << "zcov-------" << zcov << std::endl;
    //std::cout << "x_mu-------" << x_mu << std::endl;
    //std::cout << x_size << std::endl;
    //std::cout << y_size << std::endl;
    
    y_response.resize(y_size);  get_txt_column("././Data/y_response.txt",y_response);
    id_effi.resize(num_effs*y_size); id_effi = 0; get_txt_column("././Data/id_effi.txt",id_effi);

    //print(y_response);
    //Constructing the id of fixed effects in matrix A
    if(x_mu>0) for(size_t i=0; i<y_size;i++) A(i,0) = 1;
    if(zcov>0) 
    {
        blaze::DynamicMatrix<double> z_covariate(y_size,zcov); z_covariate = 0.0;
        //std::cout << "zcov-------" << z_covariate << std::endl;

        ifstream f("././Data/z_covariate.txt");
        double info = 0;
        for (size_t i = 0; i < y_size; i++) for (size_t j = 0; j < zcov; j++) {f >> info; z_covariate(i,j) = info;}
        //std::cout << "zcov-------" << z_covariate << std::endl;

        blaze::submatrix(A,0,x_mu,y_size,zcov) = z_covariate;
        iter = zcov + iter;
    }

    //Constructing the id of random effects in matrix A
    size_t j =0; for(size_t k = 0; k < theta_size_Qx; k++ ) for(size_t i=0; i<y_size;i++) {A(i,id_effi[j]) = 1; j++;} 
    //std::cout << "----------------------------" << std::endl;
    //std::cout << id_effi << std::endl;
    //std::cout << A << std::endl;
    //for(size_t r1=0; r1 < 15; r1++)
    //{
    //    for(size_t r2=0; r2 < 15; r2++)    
    //        std::cout << A(r1,r2) << ", ";

    //std::cout << std::endl;
    //}
        
    (*param).theta_size = theta_size;
    param->construct(Model,Qx_type,strategy,A,y_response,theta_size);
    //set bowl optimum.
    bool smartGrad, central, vb; double grad_stepsize; int num_threads;
    get_txt_bool("././Data/smartGrad.txt",smartGrad);
    get_txt_bool("././Data/central.txt",central);
    get_txt_bool("././Data/vbc.txt",vb);
    get_txt_double("././Data/grad_stepsize.txt",grad_stepsize);
    get_txt_int("././Data/num_threads.txt",num_threads);
    //if(x_size>700) {central = false;}

    if(vb==1) (*param).correction->vb = true; else (*param).correction->vb = false; 
    (*param).optim->construct(smartGrad,central,grad_stepsize,num_threads,theta_size);

    if(Qx_type=="generic_Qx") set_generic_Qx(param,theta_size_Qx,iter);
    (*param).theta_size_Qx = theta_size_Qx;
    
    (*param).set_utensils_eta();
    add_model_specifications(param);
}


#endif

/*
std::cout << effi << std::endl;
std::cout << RD_effi << std::endl; RW1, ...
std::cout << RD << std::endl;
std::cout << sizes << std::endl;

  //  *(param).construct(Model,Qx_type,strategy,A,y_response);
   // std::cout << y_response << std::endl;
   // std::cout << y_size << std::endl;
   // std::cout << strategy << std::endl;
   // std::cout << id_effi << std::endl;


   //size_t num_hyper = 0; get_txt_size_t("././Data/theta_size.txt",num_hyper); 

1. Stratetgy is Gaussian
 eff1 is 0
 eff2 is 1
 eff3 is 2
 eff4 is 3
 eff5 is 4
priors types
(    pc.joint )
(    pc.joint )
(    pc.joint )
(    pc.joint )
(    pc.joint )

effi
(         RW2 )
(    iid_time )
(       besag )
(   iid_space )
(       type4 )

rankdef
(           3 )
(           5 )
(           9 )
(          10 )
(          27 )

xpts
(           5 )
(           5 )
(          10 )
(          10 )
(          50 )
(           2 )
(           0 )
(           1 )
(           0 )
(          23 )
(           1 )
(           6 )
(          11 )
(          21 )
(          31 )

theta_size_Qx
5
theta_size
5
intercept
1
zcov
0
-> Theta Initial Value: ( 4 4 4 4 4 )

Optimization Starts
theta: ( 4 4 4 4 4 )

Optimization: 
8. 10 iterations
9. theta star is 0.197929 0.914026  1.05525  1.06602 0.216762
10. f(x) = -202.894
To get theta star: 0.0215616

Hessian: 
(      1.42618     0.191264   -0.0313173   -0.0350622    -0.155411 )
(     0.191264     0.692444   -0.0411219   -0.0411711   -0.0480738 )
(   -0.0313173   -0.0411219      1.56319     0.507221    -0.112067 )
(   -0.0350622   -0.0411711     0.507221      1.44882    -0.105891 )
(    -0.155411   -0.0480738    -0.112067    -0.105891      8.10975 )

Eigenvalues: 
(    0.644038 )
(      0.9955 )
(       1.462 )
(     2.02129 )
(     8.11755 )

Eigenvectors: 
(     0.235102   -0.0137431     0.966159    -0.102642   -0.0232329 )
(     -0.97138   0.00247468     0.229977   -0.0589737  -0.00687313 )
(   -0.0211268     0.665792    0.0927692     0.739821   -0.0182563 )
(    -0.026406    -0.746007    0.0657258      0.66194   -0.0170931 )
(  -0.00205263 -0.000899469    0.0268607    0.0220444     0.999394 )

corrections: 
negative: 1.23899, positive: 0.752114
negative: 0.909037, positive: 0.858796
negative: 1.00023, positive: 0.990838
negative: 1.1007, positive: 0.924747
negative: 1.00686, positive: 1.01342
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...
cooking GA...

Cooked Successfully

Marginal Likelihood:
  -  Non - GA: -199.666
  -  GA      : -199.521

Deviance:
  -  Mean of Deviance              : 259.866
  -  Deviance of Mean              : 222.41
  -  Effective Number of Paramaters: 37.4557
  -  DIC                           : 297.322

To get inference: 0.527399

Number of Constraints: 26
Number of threads: 1
Number of processes: 6
*/
