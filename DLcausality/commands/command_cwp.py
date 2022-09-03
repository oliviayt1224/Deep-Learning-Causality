import argparse
import sys
sys.path.append('.')
from DLcausality.functions.TE_class_functions import *
from DLcausality.functions.input_validation import *


def process_cwp():
    """Set the command line arguments."""
    parser = argparse.ArgumentParser(prog='Deep Learning Causality for Coupled Wiener Processes', description='Investigate the Granger causality using linear and non-linear measures.')
    parser.add_argument('--T', type=float, help='Specify the time length.')
    parser.add_argument('--N', type=int, help='Specify the time step.')
    parser.add_argument('--alpha', type=float, help='Specify the coefficient alpha.')
    parser.add_argument('--lag', type=int, help='Specify the value of time-lag.')
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
            N = 300

        if args.alpha != None:
            alpha = validation_coeff(args.alpha)
        else:
            alpha = 0.5

        if args.lag != None:
            lag = validation_lag(args.lag)
        else:
            lag = 5

        if args.num_exp != None:
            num_exp = validation_num_exp(args.num_exp)
        else:
            num_exp = 100

        validation_N_lag(N, lag)
        cwp = TE_cwp(T, N, alpha, lag)

        for i in range(num_exp):
            cwp.data_generation()
            # cwp.experiment(reverse=False)
            cwp.experiment(reverse=True)

        # cwp.compute_z_scores_c(reverse=False)
        cwp.compute_z_scores_c(reverse=True)

        # z_mean_linear = np.mean(cwp.z_scores_linear)
        # z_mean_nonlinear = np.mean(cwp.z_scores_nonlinear)
        z_mean_lin_rev = np.mean(cwp.z_scores_lin_rev)
        z_mean_nonlin_rev = np.mean(cwp.z_scores_nonlin_rev)

    except ValueError as error_message:
        print(error_message)

    else:
        # print("The mean of the linear z-scores for coupled wiener processes regarding causality from X to Y after {} experiments is {:.2f}.".format(num_exp, z_mean_linear))
        # print("The mean of the nonlinear z-scores for coupled wiener processes regarding causality from X to Y after {} experiments is {:.2f}.".format(num_exp,
        #                                                                                                         z_mean_nonlinear))
        print("The mean of the linear z-scores for coupled wiener processes regarding causality from Y to X after {} experiments is {:.2f}.".format(
            num_exp, z_mean_lin_rev))
        print("The mean of the nonlinear z-scores for coupled wiener processes regarding causality from Y to X after {} experiments is {:.2f}.".format(
            num_exp,
            z_mean_nonlin_rev))


if __name__ == "__main__":
    process_cwp()

