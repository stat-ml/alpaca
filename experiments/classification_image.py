from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from fastai.vision import rand_pad, flip_lr, ImageDataBunch, Learner, accuracy
from fastai.callbacks import EarlyStoppingCallback


from dataloader.builder import build_dataset
from experiments.utils.fastai import ImageArrayDS, Inferencer
from model.cnn import AnotherConv
from model.resnet import resnet_linear, resnet_masked
from experiment_setup import build_estimator
from uncertainty_estimator.masks import build_mask, build_masks, DEFAULT_MASKS
from analysis.metrics import uq_ndcg


torch.backends.cudnn.benchmark = True

val_size = 30_000
lr = 1e-3
weight_decay = 1e-3

config = {
    'model_runs': 3,
    'repeat_runs': 3,
    'nn_runs': 150,
    'dropout_uq': 0.5,
    'num_classes': 10
}


def ll(trainer, x, y):
    trainer.eval()
    logits = trainer(x).detach().cpu()
    probs = torch.softmax(logits, axis=-1).numpy()[np.arange(len(x)), y]
    return np.log(probs)


def main():
    data, x_train, y_train, x_val,  y_val = load_data()
    loss_func = torch.nn.CrossEntropyLoss()

    models = {
        'cnn': AnotherConv(),
        'resnet': resnet_masked(pretrained=True),
        'resnet_multiple': resnet_linear(pretrained=True)
    }

    estimation_samples = 5_000
    ndcgs, estimator_type, model_types = [], [], []
    accuracies = []

    for i in range(config['model_runs']):
        print('==models run==', i+1)
        for name, model in models.items():
            callbacks = [partial(EarlyStoppingCallback, patience=3, min_delta=1e-2, monitor='valid_loss')]
            learner = Learner(data, model, loss_func=loss_func, metrics=[accuracy], callback_fns=callbacks)
            learner.fit(100, lr, wd=weight_decay)
            inferencer = Inferencer(model)
            masks = build_masks(DEFAULT_MASKS)

            for j in range(config['repeat_runs']):
                idxs = np.random.choice(len(x_val), estimation_samples, replace=False)
                x_current = x_val[idxs]
                y_current = y_val[idxs]

                # masks
                current_ll = ll(inferencer, x_current, y_current)
                for mask_name, mask in masks.items():
                    print(mask_name)
                    estimator = build_estimator(
                        'bald_masked', inferencer, nn_runs=config['nn_runs'], dropout_mask=mask,
                        dropout_rate=config['dropout_uq'], num_classes=config['num_classes'])
                    uq = estimator.estimate(x_current)
                    estimator.reset()
                    ndcgs.append(uq_ndcg(-current_ll, uq))
                    estimator_type.append(mask_name)
                    estimator.reset()
                    model_types.append(name)
                    accuracies.append(learner.recorder.metrics[-1][0].item())
    #

    try:
        plt.figure(figsize=(12, 8))
        plt.title(f"NDCG on different train samples")

        df = pd.DataFrame({
            'ndcg': ndcgs,
            'estimator_type': estimator_type,
            'model': model_types
        })
        sns.boxplot(data=df, x='estimator_type', y='ndcg', hue='model')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title('Accuracies')
        df = pd.DataFrame({
            'accuracy': accuracies,
            'model': model_types
        })
        sns.boxplot(data=df, y='accuracy', x='model')
        plt.show()
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()



def load_data():
    dataset = build_dataset('cifar_10', val_size=val_size)
    x_train, y_train = dataset.dataset('train')
    x_val, y_val = dataset.dataset('val')

    shape = (-1, 3, 32, 32)
    x_train = ((x_train - 128)/128).reshape(shape)
    x_val = ((x_val - 128)/128).reshape(shape)

    train_tfms = [*rand_pad(4, 32), flip_lr(p=0.5)]
    train_ds = ImageArrayDS(x_train, y_train, train_tfms)
    val_ds = ImageArrayDS(x_val, y_val)
    data = ImageDataBunch.create(train_ds, val_ds, bs=256)
    return data, x_train, y_train, x_val, y_val


if __name__ == '__main__':
    main()

