#include <iostream>

#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_VECTOR(Y);
    DATA_VECTOR(x);
    PARAMETER(inflate_coef)
    PARAMETER(reg_const_coef)
    PARAMETER(reg_zipf_coef)
  
    Type nll = 0;
    //std::cout << "inflate_coef: " << inflate_coef << std::endl;
    Type inflate_one = dbinom_robust(Type(1), Type(1), inflate_coef, true);
    Type inflate_zero = dbinom_robust(Type(0), Type(1), inflate_coef, true);
    //std::cout << "inflate_one: " << inflate_one << std::endl;
    //std::cout << "inflate_zero: " << inflate_zero << std::endl;
    for (size_t i = 0; i < Y.size(); i++) {
    	Type theta = reg_zipf_coef * x[i] + reg_const_coef;
        Type elem_nll;

        if (Y[i] == 1) {
            //std::cout << "y[i]==1" << std::endl;
            elem_nll = logspace_add(
                inflate_one,
                inflate_zero + dbinom_robust(Type(1), Type(1), theta, true)
            );
        } else {
            //std::cout << "y[i]==0" << std::endl;
            elem_nll = inflate_zero + dbinom_robust(Type(0), Type(1), theta, true);
        }
        //std::cout << "elem_nll: " << elem_nll << std::endl;
        nll -= elem_nll;
    }
    return nll;
}
