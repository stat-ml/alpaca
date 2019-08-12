import argparse
from model.mlp import MLP
from dataloader.rosen import RosenData


def main(config):
    X_train, y_train, X_val, y_val, _, _, X_pool, y_pool = RosenData(
        config['n_train'], config['n_val'], config['n_test'], config['n_pool'], config['n_dim']
    ).dataset(use_cache=config['use_cache'])

    model = MLP(config['n_dim'])

    model.fit(
        X_train, y_train, batch_size=64, epochs=config['epochs'],
        validation_data=[X_val, y_val])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Change experiment parameters')
    parser.add_argument(
        '--estimators', choices=['nngp', 'mcdue', 'random'], default=['nngp', 'mcdue', 'random'],
        nargs='+', help='Estimator types for the experiment')
    parser.add_argument(
        '--random-seed', type=int, default=None,
        help='Set the seed to make result reproducible')
    parser.add_argument(
        '--n-dim', type=int, default=10, help='Rosenbrock function dimentions')
    parser.add_argument(
        '--n-train', type=int, default=200, help='Initial size of train dataset')
    parser.add_argument(
        '--n-val', type=int, default=200, help='Initial size of validation dataset')
    parser.add_argument(
        '--n-test', type=int, default=200, help='Initial size of test dataset')
    parser.add_argument(
        '--n-pool', type=int, default=1000, help='Initial size of test dataset')
    parser.add_argument(
        '--epochs', type=int, default=10000, help='Initial size of test dataset')
    parser.add_argument(
        '--update-size', type=int, default=100,
        help='Amount of samples to take from pool per iteration')
    parser.add_argument(
        '--al-iterations', '-i', type=int, default=10, help='Number of learning iterations')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--no-use-cache', dest='use_cache', action='store_false',
        help='To generate new sample points for rosenbrock function')
    parser.add_argument(
        '--layers', type=int, nargs='+', default=[128, 64, 32],
        help='Size of the layers in neural net')

    return vars(parser.parse_args())


if __name__ == '__main__':
    config = parse_arguments()
    main(config)