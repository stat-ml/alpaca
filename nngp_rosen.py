import argparse

import torch

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.mcdue import MCDUE


def main(config):
    rosen = RosenData(
        config['n_dim'], config['data_size'], config['data_split'],
        use_cache=config['use_cache'])
    x_train, y_train = rosen.dataset('train')
    x_val, y_val = rosen.dataset('train')
    x_pool, y_pool = rosen.dataset('pool')

    model = MLP(config['layers'])
    estimator = MCDUE(model, 25)

    # # Training the model
    # model.fit((x_train, y_train), (x_val, y_val), epochs=config['epochs'])
    # torch.save(model.state_dict(), 'model/data/mlp_rosen.ckpt')
    # print(model.evaluate(pool_set))

    model.load_state_dict(torch.load('model/data/mlp_rosen.ckpt'))
    print(model.predict(x_pool[:1]))
    #
    print(estimator.estimate(x_pool, x_train, y_train))











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