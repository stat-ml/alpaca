import sys
import torch
import numpy as np
from fastai.vision import (rand_pad, flip_lr)

sys.path.append('..')
from dataloader.builder import build_dataset
from uncertainty_estimator.masks import DEFAULT_MASKS
from experiments.active_learning import main

torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


def prepare_svhn(config):
    dataset = build_dataset('svhn', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')
    y_set[y_set == 10] = 0
    y_val[y_val == 10] = 0

    shape = (-1, 32, 32, 3)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)
    x_set = np.rollaxis(x_set, 3, 1)
    x_val = np.rollaxis(x_val, 3, 1)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]  # Transformation to augment images

    return x_set, y_set, x_val, y_val, train_tfms


svnh_config = {
    'repeats': 3,
    'start_size': 5_000,
    'step_size': 50,
    'val_size': 10_000,
    'pool_size': 12_000,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs_per_step': 30,
    'patience': 2,
    'model_type': 'resnet',
    'nn_runs': 100,
    'batch_size': 256,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_svhn,
    'name': 'svhn'
}


if __name__ == '__main__':
    main(svnh_config)
