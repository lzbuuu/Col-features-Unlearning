import argparse


def args_parser():
    """
    Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-nb", "--num_batches", type=int, default=938)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1000)
    parser.add_argument("-ls", "--log_steps", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-ep", "--epochs", type=int, default=5)
    parser.add_argument("-uep", "--unlearn_epochs", type=int, default=60)
    parser.add_argument("-p", "--plot", type=bool, default=True)
    parser.add_argument("-save", "--save", type=bool, default=True)
    parser.add_argument("-sd", "--seed", type=int, default=3499)
    parser.add_argument("-dataset", "--dataset", type=str, default="Diabetes")
    parser.add_argument("-nc", "--num_clients", type=int, default=3)
    parser.add_argument("-size", "--n_total_features", type=int, default=32)
    parser.add_argument("-device", "--device", type=str, default='cuda')
    parser.add_argument("-rounds", "--rounds", type=int, default=100)
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('-retain', '--retain', type=bool, default=False)
    parser.add_argument('-stone2', '--stone2', type=float, default=0.8)
    parser.add_argument('-step_gamma', '--step_gamma', default=0.1, type=float, help='gamma for step scheduler')
    parser.add_argument('-test_size', '--test_size', default=0.2, type=float, help='test_size of Criteo')
    parser.add_argument("-n_unlearned_features", '--n_unlearned_features', default=1, type=int)
    parser.add_argument("-unlearned_id", '--unlearned_id', default=-1, type=int)
    parser.add_argument("-percent_poison", '--percent_poison', default=0.5, type=float)
    parser.add_argument("-bd", '--backdoor', default=True, type=bool)
    args = parser.parse_args()
    return args
