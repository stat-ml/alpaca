import random
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.mlp import MLP
from dataloader.rosen import RosenData
from uncertainty_estimator.nngp import NNGP
from uncertainty_estimator.mcdue import MCDUE
from uncertainty_estimator.random_estimator import RandomEstimator
from sample_selector.eager import EagerSampleSelector
from oracle.identity import IdentityOracle
from al_trainer import ALTrainer

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def run_experiment(config):
    """
    Run active learning for the 10D rosenbrock function data
    It starts from small train dataset and then extends it with points from pool

    We compare three sampling methods:
    - Random datapoints
    - Points with highest uncertainty by MCDUE
    - Points with highest uncertainty by NNGP (proposed method)
    """
    sess = None
    rmses = {}

    for estimator_name in config['estimators']:
        print("\nEstimator:", estimator_name)

        # load data
        X_train, y_train, X_val, y_val, _, _, X_pool, y_pool = RosenData(
            config['n_train'], config['n_val'], config['n_test'], config['n_pool'], config['n_dim']
        ).dataset(use_cache=config['use_cache'])

        # Build neural net and set random seed
        model, sess = build_tf_model(sess, config['n_dim'], config['layers'], config['random_seed'])

        estimator = build_estimator(estimator_name, model)  # to estimate uncertainties
        oracle = IdentityOracle(y_pool)  # generate y for X from pool
        sampler = EagerSampleSelector(oracle)  # sample X and y from pool by uncertainty estimations

        # Active learning training
        trainer = ALTrainer(
            model, estimator, sampler, oracle, config['al_iterations'],
            config['update_size'], verbose=config['verbose'])
        rmses[estimator_name] = trainer.train(X_train, y_train, X_val, y_val, X_pool)

    visualize(rmses)


def build_tf_model(sess, n_dim, layers, random_seed):
    tf.reset_default_graph()
    if random_seed is not None:
        # Setting seeds for reproducibility
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    model = MLP(ndim=n_dim, layers=layers)

    if sess is not None and not sess._closed:
        sess.close()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    model.set_session(sess)

    return model, sess


def build_estimator(name, model):
    if name == 'nngp':
        estimator = NNGP(model)
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
        '--n-train', type=int, default=200, help='Initial size of train dataset')
    parser.add_argument(
        '--n-val', type=int, default=200, help='Initial size of validation dataset')
    parser.add_argument(
        '--n-test', type=int, default=200, help='Initial size of test dataset')
    parser.add_argument(
        '--n-pool', type=int, default=1000, help='Initial size of test dataset')
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
    run_experiment(config)
