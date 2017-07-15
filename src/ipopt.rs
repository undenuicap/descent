extern crate libc;

use std::ptr;
use std::os::raw::c_char;
use std::ffi::CString;
use self::libc::c_void;

type Number = f64;
type Index = i32;
type Int = i32;
type Bool = i32;
type UserDataPtr = *mut c_void;

struct IpoptProblemInfo;
type IpoptProblem = *mut IpoptProblemInfo;

enum ApplicationReturnStatus {
    Solve_Succeeded = 0,
    Solved_To_Acceptable_Level = 1,
    Infeasible_Problem_Detected = 2,
    Search_Direction_Becomes_Too_Small = 3,
    Diverging_Iterates = 4,
    User_Requested_Stop = 5,
    Feasible_Point_Found = 6,

    Maximum_Iterations_Exceeded = -1,
    Restoration_Failed = -2,
    Error_In_Step_Computation = -3,
    Maximum_CpuTime_Exceeded = -4,
    Not_Enough_Degrees_Of_Freedom = -10,
    Invalid_Problem_Definition = -11,
    Invalid_Option = -12,
    Invalid_Number_Detected = -13,

    Unrecoverable_Exception = -100,
    NonIpopt_Exception_Thrown = -101,
    Insufficient_Memory = -102,
    Internal_Error = -199
}

// See IpStdCInterface.h for descriptions
type Eval_F_CB = fn(n: Index,
                    x: *const Number,
                    new_x: Bool,
                    obj_value: *mut Number,
                    user_data: UserDataPtr) -> Bool;

type Eval_Grad_F_CB = fn(n: Index,
                         x: *const Number,
                         new_x: Bool,
                         grad_f: *mut Number,
                         user_data: UserDataPtr) -> Bool;

type Eval_G_CB = fn(n: Index,
                    x: *const Number,
                    new_x: Bool,
                    m: Index,
                    g: *const Number,
                    user_data: UserDataPtr) -> Bool;

type Eval_Jac_G_CB = fn(n: Index,
                        x: *const Number,
                        new_x: Bool,
                        m: Index,
                        nele_jac: Index,
                        iRow: *mut Index,
                        jCol: *mut Index,
                        values: *mut Number,
                        user_data: UserDataPtr) -> Bool;

type Eval_H_CB = fn(n: Index,
                    x: *const Number,
                    new_x: Bool,
                    obj_factor: Number,
                    m: Index,
                    lambda: *const Number,
                    new_lambda: Bool,
                    nele_hess: Index,
                    iRow: *mut Index,
                    jCol: *mut Index,
                    values: *mut Number,
                    user_data: UserDataPtr) -> Bool;

type Intermediate_CB = fn(alg_mod: Index,
                          iter_count: Index,
                          obj_value: Number,
                          inf_pr: Number,
                          inf_du: Number,
                          mu: Number,
                          d_norm: Number,
                          regularization_size: Number,
                          alpha_du: Number,
                          alpha_pr: Number,
                          ls_trials: Number,
                          user_data: UserDataPtr) -> Bool;

#[link(name = "ipopt")]
extern {
    fn CreateIpoptProblem(
            n: Index, // number of variables
            x_L: *const Number, // variable lower bounds
            x_U: *const Number, // variable upper bounds
            m: Index, // number of constraints
            g_L: *const Number, // constraint lower bounds
            g_U: *const Number, // constraint upper bounds
            nele_jac: Index, // non-zeros in constraint jacobian
            nele_hess: Index, // non-zeros in lagrangian hessian
            index_style: Index, // indexing style, 0: C, 1 Fortran
            eval_f: Eval_F_CB,
            eval_g: Eval_G_CB,
            eval_grad_f: Eval_Grad_F_CB,
            eval_jac_g: Eval_Jac_G_CB,
            eval_h: Eval_H_CB) -> IpoptProblem;

    fn FreeIpoptProblem(ipopt_problem: IpoptProblem);

    fn AddIpoptStrOption(ipopt_problem: IpoptProblem,
                         keyword: *const c_char,
                         val: *const c_char) -> Bool;

    fn AddIpoptNumOption(ipopt_problem: IpoptProblem,
                         keyword: *const c_char,
                         val: Number) -> Bool;

    fn AddIpoptIntOption(ipopt_problem: IpoptProblem,
                         keyword: *const c_char,
                         val: Int) -> Bool;

    fn IpoptSolve(
        ipopt_problem: IpoptProblem,
        x: *const Number,
        g: *mut Number,
        obj_val: *mut Number,
        mult_g: *const Number,
        mult_x_L: *const Number,
        mult_x_U: *const Number,
        user_data: UserDataPtr) -> ApplicationReturnStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(n: Index,
         x: *const Number,
         new_x: Bool,
         obj_value: *mut Number,
         user_data: UserDataPtr) -> Bool {
        unsafe {
            *obj_value = (*x)*(*x);
        }
        1
    }

    fn f_grad(n: Index,
              x: *const Number,
              new_x: Bool,
              grad_f: *mut Number,
              user_data: UserDataPtr) -> Bool {
        unsafe {
            *grad_f = 1.0;
        }
        1
    }

    fn g(n: Index,
         x: *const Number,
         new_x: Bool,
         m: Index,
         g: *const Number,
         user_data: UserDataPtr) -> Bool {
        1
    }

    fn g_jac(n: Index,
             x: *const Number,
             new_x: Bool,
             m: Index,
             nele_jac: Index,
             iRow: *mut Index,
             jCol: *mut Index,
             values: *mut Number,
             user_data: UserDataPtr) -> Bool {
        if iRow != ptr::null_mut() && jCol != ptr::null_mut() {
            // set sparsity
        } else {
            // set values
        }
        1
    }

    fn l_hess(n: Index,
              x: *const Number,
              new_x: Bool,
              obj_factor: Number,
              m: Index,
              lambda: *const Number,
              new_lambda: Bool,
              nele_hess: Index,
              iRow: *mut Index,
              jCol: *mut Index,
              values: *mut Number,
              user_data: UserDataPtr) -> Bool {
        unsafe {
            if iRow != ptr::null_mut() && jCol != ptr::null_mut() {
                *iRow = 0;
                *jCol = 0;
                // set sparsity
            } else {
                *values = 2.0;
                // set values
            }
        }
        1
    }

    #[test]
    fn test_it() {
        let x_lb = vec![0.2];
        let x_ub = vec![1.0];
        let g_lb = vec![];
        let g_ub = vec![];
        unsafe {
            let mut prob = CreateIpoptProblem(1,
                                              x_lb.as_ptr(),
                                              x_ub.as_ptr(),
                                              0,
                                              g_lb.as_ptr(),
                                             g_ub.as_ptr(),
                                              0,
                                              1,
                                              0,
                                              f,
                                              g,
                                              f_grad,
                                              g_jac,
                                              l_hess);
            assert!(prob != ptr::null_mut());
            let mut opt = CString::new("print_level").unwrap();
            let set = AddIpoptIntOption(prob, opt.as_ptr(), 0);
            assert!(set != 0);
            let opt = CString::new("sb").unwrap();
            let opt_val = CString::new("yes").unwrap();
            AddIpoptStrOption(prob, opt.as_ptr(), opt_val.as_ptr());
            let mut x = vec![0.5];
            let mut obj_val = 0.0;
            let ret = IpoptSolve(prob,
                                 x.as_ptr(),
                                 ptr::null_mut(),
                                 &mut obj_val,
                                 ptr::null_mut(),
                                 ptr::null_mut(),
                                 ptr::null_mut(),
                                 ptr::null_mut());
            assert_eq!(x[0], 0.2);
            FreeIpoptProblem(prob);
        }
    }
}
