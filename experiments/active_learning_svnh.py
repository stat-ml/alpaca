import sys
import torch
from fastai.vision import (rand_pad, flip_lr)

sys.path.append('..')
from dataloader.builder import build_dataset
from uncertainty_estimator.masks import DEFAULT_MASKS
from experiments.active_learning import main

torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True


def _main():
    dataset = build_dataset('svhn', val_size=5_000)
    print(dataset)



def prepare_svhn(config):
    dataset = build_dataset('svhn', val_size=config['val_size'])
    x_set, y_set = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_set = ((x_set - 128) / 128).reshape(shape)
    x_val = ((x_val - 128) / 128).reshape(shape)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]  # Transformation to augment images

    return x_set, y_set, x_val, y_val, train_tfms


svnh_config = {
    'repeats': 5,
    'start_size': 100,
    'step_size': 20,
    'val_size': 10_000,
    'pool_size': 10_000,
    'steps': 30,
    'methods': ['random', 'error_oracle', 'max_entropy', *DEFAULT_MASKS],
    'epochs_per_step': 30,
    'patience': 2,
    'model_type': 'simple_conv',
    'nn_runs': 100,
    'batch_size': 32,
    'start_lr': 5e-4,
    'weight_decay': 0.2,
    'prepare_dataset': prepare_svhn
}


if __name__ == '__main__':
    _main()
    # main(svnh_config)