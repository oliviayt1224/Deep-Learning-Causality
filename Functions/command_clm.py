import argparse
import sys
sys.path.append('.')
from Functions.TE_class_functions import *
from Functions.input_validation import *


def process_clm():
    """Set the command line arguments."""
    parser = argparse.ArgumentParser(prog='Deep Learning Causality for Coupled Logistic Maps', description='Investigate the Granger causality using linear and non-linear measures.')
    parser.add_argument('--T', type=float, help='Specify the time length.')
    parser.add_argument('--N', type=int, help='Specify the time step.')
    parser.add_argument('--X', type=float, help='Specify the initial value of X.')
    parser.add_argument('--Y', type=float, help='Specify the initial value of Y.')
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
            N = 100

        if args.alpha != None:
            alpha = validation_coeff(args.alpha)
        else:
            alpha = 0.5

        if args.epsilon != None:
            epsilon = validation_coeff(args.epsilon)
        else:
            epsilon = 0.9

        if args.X != None:
            X = validation_XY(args.X)
        else:
            X = 0.5

        if args.Y != None:
            Y = validation_XY(args.Y)
        else:
            Y = 0.5

        if args.num_exp != None:
            num_exp = validation_num_exp(args.num_exp)
        else:
            num_exp = 100

        validation_N_lag(N, 1)
        cwp = TE_clm(X, Y, T, N, alpha, epsilon,)
        cwp.data_generation()
        cwp.multiple_experiment(num_exp)
        cwp.compute_z_scores()
        z_mean_linear = np.mean(cwp.z_scores_linear)
        z_mean_nonlinear = np.mean(cwp.z_scores_nonlinear)

    except ValueError as error_message:
        print(error_message)

    else:
        print("The mean of the linear z-scores for coupled logistic maps after {} experiments is {:.2f}.".format(num_exp, z_mean_linear))
        print("The mean of the nonlinear z-scores for coupled logistic maps after {} experiments is {:.2f}.".format(num_exp,
                                                                                                                z_mean_nonlinear))


if __name__ == "__main__":
    process_clm()

