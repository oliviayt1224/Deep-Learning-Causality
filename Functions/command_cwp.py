import argparse
import sys
sys.path.append('.')
from Functions.TE_class_functions import *
from Functions.input_validation import *


def process_cwp():
    """Set the command line arguments."""
    parser = argparse.ArgumentParser(prog='Deep Learning Causality for Coupled Wiener Processes', description='Investigate the Granger causality using linear and non-linear measures.')
    parser.add_argument('--T', nargs=1, type=str, help='Specify the time length.')
    parser.add_argument('--N', nargs=1, type=int, help='Specify the time step.')
    parser.add_argument('--alpha', nargs=1, type=str, help='Specify the coefficient alpha.')
    parser.add_argument('--lag', nargs=1, type=int, help='Specify the value of time-lag.')
    parser.add_argument('--num_exp', nargs=1, type=int, help='Specify the number of experiments.')

    args = parser.parse_args()

    try:
        if args.T != None:
            T = validation_T(args.T)
        else:
            T = 1

        if args.N != None:
            N = validation_N(args.N)
        else:
            N = 100

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
        cwp.data_generation()
        cwp.multiple_experiment(num_exp)
        cwp.compute_z_scores()
        z_mean_linear = np.mean(cwp.z_scores_linear)
        z_mean_nonlinear = np.mean(cwp.z_scores_nonlinear)

    except ValueError as error_message:
        print(error_message)

    else:
        print("The mean of the linear z-scores for coupled wiener processes after {} experiments is {:.2f} seconds.".format(num_exp, z_mean_linear))
        print("The mean of the nonlinear z-scores for coupled wiener processes after {} experiments is {:.2f} seconds.".format(num_exp,
                                                                                                                z_mean_nonlinear))

if __name__ == "__main__":
    process_cwp()

