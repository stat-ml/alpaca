import sys
sys.path.append('..')
# %%
import matplotlib.pyplot as plt

from experiment_setup import get_model, set_random, build_estimator
from uncertainty_estimator.masks import build_masks, DEFAULT_MASKS, BASIC_MASKS
from dataloader.toy import ToyQubicData, ToySinData
from model.dense import Dense
from model.trainer import Trainer
import torch

torch.cuda.set_device(1)
# %%
plt.rcParams['figure.facecolor'] = 'white'
# %%

# %%
config = {
    'nn_runs': 50,
    'verbose': False,
    'use_cache': False,
    'layers': [1, 64, 64, 32, 1],
    'patience': 50,
    'dropout_train': 0,
    'dropout_uq': 0
}
# %% md
### Visualizing on toy data

#### Generate dataset

# %%
dataset = 'sin'

data_class = ToyQubicData

x_train, y_train = data_class(use_cache=config['use_cache']).dataset('train')
x_val, y_val = data_class(use_cache=config['use_cache']).dataset('val')
x_true, y_true = data_class().dataset('ground_truth')

plt.plot(x_true, y_true)
plt.scatter(x_train, y_train, color='red')
plt.scatter(x_val, y_val, color='green')
# %% md

#### Train model

# %%
# model = MLP(config['layers'], l2_reg=1e-5)
# model.fit(
#     (x_train, y_train), (x_train, y_train),
#     patience=config['patience'], validation_step=10, batch_size=15,
#     dropout_rate=config['dropout_train'])
# x_ = np.concatenate((x_true, x_train))
# y_ = model(x_).cpu().numpy()
# plt.figure(figsize=(22, 12))
# plt.plot(x_true, y_true, alpha=0.5)
# plt.scatter(x_train, y_train, color='red')
# plt.scatter(x_, y_, color='green', marker='+')

# %%
model = Dense(config['layers'])
print(model)
# %%
trainer = Trainer(
    model, batch_size=15, dropout_train=config['dropout_train'])
# %%
trainer.fit((x_train, y_train), (x_val, y_val), patience=config['patience'])
# %%
