import argparse
from model.mlp import MLP
from dataloader.rosen import RosenData


def main(config):
    rosen = RosenData(
        config['n_dim'], config['data_size'], config['data_split'],
        use_cache=config['use_cache'])
    train_set = rosen.dataset('train')
    val_set = rosen.dataset('train')
    pool_set = rosen.dataset('pool')

    model = MLP(config['layers'])
    model.fit(train_set, val_set, epochs=config['epochs'])

    print(model.evaluate(pool_set))


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
        '--data-size', type=int, default=2000, help='Size of dataset')
    parser.add_argument(
        '--data-split', type=int, default=[0.1, 0.05, 0.05, 0.8], help='Size of dataset')
    parser.add_argument(
        '--epochs', '-e', type=int, default=10000, help='Initial size of test dataset')
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
        '--layers', type=int, nargs='+', default=[10, 128, 64, 32, 1],
        help='Size of the layers in neural net')

    return vars(parser.parse_args())


if __name__ == '__main__':
    config = parse_arguments()
    main(config)