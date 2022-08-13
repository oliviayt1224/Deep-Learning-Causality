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

            T = validation_T()
            queue_list = queue_validation(args.queue)
            queue = list_to_queue(queue_list)

            num_tap = num_tap_validation(args.num_tap)

            if (args.walk_time != None):
                walk_time = walk_time_validation(args.walk_time)

                if (args.flow_rate != None):


                    flow_rate = flow_rate_validation(num_tap, args.flow_rate)

                    args.num_tap = len(flow_rate)
                    time = fill_water_diff_flow(queue, num_tap, walk_time, flow_rate)

                else:

                    time = fill_water_walk_time(queue, num_tap, walk_time)

            else:

                time = fill_water_same_flow(queue, num_tap)

        else:
            raise RuntimeError("Invalid use of the command. Please check the usage information below.")

    except ValueError as error_message:
        print(error_message)

    except RuntimeError as error_message:
        print(error_message)
        print("Three different modules of this command:")
        print("Module 1 (no walking time and same flow rates): ")
        print("\t You should specify the queue and the number of taps.")
        print("Module 2 (a fixed walking time and same flow rates): ")
        print("\t You should specify the queue, the number of taps and the walking time.")
        print("Module 3 (a fixed walking time and different flow rates): ")
        print("\t You should specify the queue, the number of taps, the walking time and the flow rates.")

    else:
        print("The total time for all people filling up their water bottles is {:.2f} seconds.".format(time))


if __name__ == "__main__":
    process()

