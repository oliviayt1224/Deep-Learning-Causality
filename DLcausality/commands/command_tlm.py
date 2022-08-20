import argparse
import sys
sys.path.append('.')
from DLcausality.functions.TE_class_functions import *
from DLcausality.functions.input_validation import *


def process_tlm():
    """Set the command line arguments."""
    parser = argparse.ArgumentParser(prog='Deep Learning Causality for Ternary Logistic Maps', description='Investigate the Granger causality using linear and non-linear measures.')
    parser.add_argument('--T', type=float, help='Specify the time length.')
    parser.add_argument('--N', type=int, help='Specify the time step.')
    parser.add_argument('--alpha', type=float, help='Specify the coefficient alpha.')
    parser.add_argument('--epsilon', type=float, help='Specify the coefficient epsilon.')
    parser.add_argument('--num_exp', type=int, help='Specify the number of experiments.')
    args = parser.parse_args()

    try:
        if args.T != None:
            T = validation_T(args.T)
        else:
            T = 1

        if args.N != None:
            N = validation_N(args.N)
        else:
            N = 700

        if args.alpha != None:
            alpha = validation_coeff(args.alpha)
        else:
            alpha = 0.4

        if args.epsilon != None:
            epsilon = validation_coeff(args.epsilon)
        else:
            epsilon = 0.9

        if args.num_exp != None:
            num_exp = validation_num_exp(args.num_exp)
        else:
            num_exp = 100

        validation_N_lag(N, 1)
        tlm = TE_tlm(0.4, 0.4, T, N, alpha, epsilon)
        tlm.data_generation()
        tlm.multiple_experiment(num_exp)
        tlm.compute_z_scores_t()
        md_TE_linear, md_TE_nonlinear = tlm.mean_of_diff_TE()
        z_mean_linear = np.mean(tlm.z_scores_linear)
        z_mean_nonlinear = np.mean(tlm.z_scores_nonlinear)

    except ValueError as error_message:
        print(error_message)

    else:
        print("The mean of the linear z-scores for ternary logistic maps after {} experiments is {:.2f}.".format(num_exp, z_mean_linear))
        print("The mean of the nonlinear z-scores for ternary logistic maps after {} experiments is {:.2f}.".format(num_exp,z_mean_nonlinear))
        print("The mean of the difference between linear TE and conditional TE is {:.2f}.".format(md_TE_linear))
        print("The mean of the difference between nonlinear TE and conditional TE is {:.2f}.".format(md_TE_nonlinear))


if __name__ == "__main__":
    process_tlm()

