import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataloader.rosen import RosenData
from experiment_setup import get_model, set_random, build_estimator
from analysis.metrics import uq_accuracy, uq_ndcg
from uncertainty_estimator.masks import BasicMask, LHSMask, MirrorMask, DecorrelationMask


def experiment_config():
    return {
        'random_seed': 43,
        'n_dim': 10,
        'nn_runs': 100,
        'runs': 2,
        'model_runs': 5,
        'data_size': 2000,
        'data_split': [0.4, 0.6, 0, 0],
        'verbose': True,
        'use_cache': True,
        'layers': [10, 128, 64, 32, 1],
        'epochs': 30_000,
        'acc_percentile': 0.1,
        'patience': 5,
        'target_loss': 750,
    }


def data(config):
    rosen = RosenData(
        config['n_dim'], config['data_size'], config['data_split'],
        use_cache=config['use_cache'])

    x_train, y_train = rosen.dataset('train')
    x_val, y_val = rosen.dataset('train')
    return x_train, y_train, x_val, y_val


def train_the_models(config, x_train, y_train, x_val, y_val, model_paths):
    for i in range(config['model_runs']):
        model = get_model(
            config['layers'], model_paths[i],
            (x_train, y_train), (x_val, y_val), epochs=config['epochs'],
            retrain=True, verbose=False, patience=config['patience'])
        print(f"Loss {model.val_loss}/{config['target_loss']}")


def mask_list(config):
    return {
        'vanilla': None,
        'basic_mask': BasicMask(),
        'lhs': LHSMask(config['nn_runs']),
        'lhs_shuffled': LHSMask(config['nn_runs'], shuffle=True),
        'mirror_random': MirrorMask(),
        'decorrelating': DecorrelationMask(),
        'decorrelating_scaled': DecorrelationMask(scaling=True, dry_run=False)
    }


def mask_performance(masks, config, x_train, y_train, x_val, y_val, model_paths):
    mask_results = []
    for model_run in range(config['model_runs']):
        print(f"===MODEL RUN {model_run+1}====")
        model = get_model(
            config['layers'], model_paths[model_run],
            (x_train, y_train), (x_val, y_val), epochs=config['epochs'])
        predictions = model(x_val).cpu().numpy()
        errors = np.abs(predictions - y_val)

        for name, mask in masks.items():
            estimator = build_estimator(
                'mcdue_masked', model, nn_runs=config['nn_runs'], dropout_mask=mask,
                dropout_rate=0.3)

            for run in range(config['runs']):
                estimations = estimator.estimate(x_val, x_train, y_train)
                acc = uq_accuracy(estimations, errors, config['acc_percentile'])
                ndcg = uq_ndcg(errors, estimations)
                mask_results.append([acc, ndcg, name])

                if hasattr(mask, 'reset'):
                    mask.reset()

    return mask_results


def plot_mask_results(mask_results):
    mask_df = pd.DataFrame(mask_results, columns=['acc', 'ndcg', 'mask'])
    plt.figure()
    plt.xticks(rotation=30)
    plt.ylim(0, 0.8)
    sns.boxplot(data=mask_df, x='mask', y='acc')
    plt.figure()
    plt.xticks(rotation=30)
    plt.ylim(0, 0.9)
    sns.boxplot(data=mask_df, x='mask', y='ndcg')
    plt.show()


if __name__ == '__main__':
    config = experiment_config()
    x_train, y_train, x_val, y_val = data(config)
    model_paths = [f"model/data/rosen_visual_{i}.ckpt" for i in range(config['model_runs'])]
    if not config['use_cache']:
        train_the_models(config, x_train, y_train, x_val, y_val, model_paths)
    masks = mask_list(config)
    mask_results = mask_performance(masks, config, x_train, y_train, x_val, y_val, model_paths)
    plot_mask_results(mask_results)

