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

        for i in range(num_exp):
            tlm.data_generation()
            tlm.experiment(reverse=False)
            tlm.experiment(reverse=True)

        tlm.compute_z_scores_t(reverse=False)
        tlm.compute_z_scores_t(reverse=True)

        z_mean_linear = np.mean(tlm.z_scores_linear)
        z_mean_nonlinear = np.mean(tlm.z_scores_nonlinear)
        z_mean_linear_rev = np.mean(tlm.z_scores_lin_rev)
        z_mean_nonlinear_rev = np.mean(tlm.z_scores_nonlin_rev)

        mean_z_diff_lin, mean_z_diff_nonlin = tlm.z_score_diff_TE()

    except ValueError as error_message:
        print(error_message)

    else:
        print(
            "The mean of the linear z-scores for ternary logistic maps regarding causality from X to Y after {} experiments is {:.2f}.".format(
                num_exp, z_mean_linear))
        print(
            "The mean of the nonlinear z-scores for ternary logistic maps regarding causality from X to Y after {} experiments is {:.2f}.".format(
                num_exp, z_mean_nonlinear))
        print(
            "The mean of the linear z-scores for ternary logistic maps regarding causality from Y to X after {} experiments is {:.2f}.".format(
                num_exp, z_mean_linear_rev))
        print(
            "The mean of the nonlinear z-scores for ternary logistic maps regarding causality from Y to X after {} experiments is {:.2f}.".format(
                num_exp, z_mean_nonlinear_rev))
        print("The mean of the linear z-scores for the difference between TE and conditional TE is {:.2f}.".format(
            mean_z_diff_lin))
        print("The mean of the nonlinear z-scores for the difference between TE and conditional TE is {:.2f}.".format(
            mean_z_diff_nonlin))


if __name__ == "__main__":
    process_tlm()

