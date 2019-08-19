import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.nngp import NNGPRegression
from uncertainty_estimator.mcdue import MCDUE
from uncertainty_estimator.random_estimator import RandomEstimator
from sample_selector.eager import EagerSampleSelector
from oracle.identity import IdentityOracle
from al_trainer import ALTrainer


def run_experiment(config):
    """
    Run active learning for the 10D rosenbrock function data
    It starts from small train dataset and then extends it with points from pool

    We compare three sampling methods:
    - Random datapoints
    - Points with highest uncertainty by MCDUE
    - Points with highest uncertainty by NNGP (proposed method)
    """
    rmses = {}

    for estimator_name in config['estimators']:
        print("\nEstimator:", estimator_name)

        # load data

        rosen = RosenData(
            config['n_dim'], config['data_size'], config['data_split'],
            use_cache=config['use_cache'])
        x_train, y_train = rosen.dataset('train')
        x_val, y_val = rosen.dataset('train')
        x_pool, y_pool = rosen.dataset('pool')

        # Build neural net and set random seed
        set_random(config['random_seed'])
        model = MLP(config['layers'])

        estimator = build_estimator(estimator_name, model)  # to estimate uncertainties
        oracle = IdentityOracle(y_pool)  # generate y for X from pool
        sampler = EagerSampleSelector()  # sample X and y from pool by uncertainty estimations

        # Active learning training
        trainer = ALTrainer(
            model, estimator, sampler, oracle, config['al_iterations'],
            config['update_size'], verbose=config['verbose'])
        rmses[estimator_name] = trainer.train(x_train, y_train, x_val, y_val, x_pool)

    visualize(rmses)


def set_random(random_seed):
    # Setting seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


def build_estimator(name, model):
    if name == 'nngp':
        estimator = NNGPRegression(model)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def visualize(rmses):
    print(rmses)
    plt.figure(figsize=(12, 9))
    plt.xlabel('Active learning iteration')
    plt.ylabel('Validation RMSE')
    for estimator_name, rmse in rmses.items():
        plt.plot(rmse, label=estimator_name, marker='.')

    plt.title('RMS Error by active learning iterations')
    plt.legend()

    plt.show()


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
        '--data-split', type=int, default=[0.1, 0.1, 0.1, 0.7], help='Size of dataset')
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
    run_experiment(config)


config = {
    'estimators': ['nngp', 'mcdue', 'random'],
    'random_seed': None,
    'n_dim': 9,
    'data_size': 2400,
    'data_split': [0.2, 0.1, 0.1, 0.6],
    'update_size': 99,
    'al_iterations': 9,
    'verbose': False,
    'use_cache': True,
    'layers': [9, 128, 64, 32, 1]
}
